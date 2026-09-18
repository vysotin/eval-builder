#!/usr/bin/env python3
"""Copy GitHub trees using playwright-cli and the browser's Copy raw file button."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from urllib.parse import quote, urlsplit
import uuid

HERE = Path(__file__).resolve().parent
CLIPBOARD = threading.Lock()


def default_cli():
    # Never silently fall back to a newer global CLI when the pinned install is absent.
    return str(HERE / 'node_modules/.bin/playwright-cli')


def parse_cli_result(output):
    # Playwright 1.59 has no --raw. Decode only its Result section, ignoring
    # subsequent generated-code and page sections (including untrusted page text).
    match = re.search(r'^### Result\s*\n', output, re.MULTILINE)
    if not match:
        raise RuntimeError('Missing playwright-cli Result section: ' + output[:300])
    try:
        value, end = json.JSONDecoder().raw_decode(output[match.end():].lstrip())
        return value
    except ValueError as e:
        raise RuntimeError('Invalid/truncated playwright-cli Result section') from e


def find_chrome(explicit=None):
    candidates = [explicit] if explicit else [
        shutil.which('google-chrome'), shutil.which('google-chrome-stable'),
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        str(Path.home() / 'Applications/Google Chrome.app/Contents/MacOS/Google Chrome'),
        '/opt/google/chrome/chrome',
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser().resolve()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
    raise ValueError('Installed Google Chrome not found. Supply --chrome-path /path/to/chrome. '
                     'No browser will be downloaded or installed.')


def git_hash(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def validate(data, expected):
    if not re.fullmatch(r'[0-9a-f]{7,40}', expected or ''):
        raise ValueError('GitHub did not expose a usable blob hash; refusing unverified content')
    # GitHub copies joined lines without the final LF. Some clipboards normalize CRLF.
    candidates = [data, data + b'\n']
    if b'\r' not in data:
        candidates += [v.replace(b'\n', b'\r\n') for v in candidates]
    for candidate in candidates:
        if git_hash(candidate).startswith(expected):
            return candidate
    raise ValueError(f'Copied bytes do not match GitHub blob {expected} (truncated, stale, binary, or encoding changed)')


def local_path(root, relative):
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith('/') or any(p in ('.', '..') or '\\' in p for p in parts):
        raise ValueError(f'Unsafe repository path: {relative!r}')
    target = root.joinpath(*parts)
    for p in [target, *target.parents]:
        if p == root:
            break
        if p.is_symlink():
            raise ValueError(f'Refusing destination symlink: {p}')
    target.resolve().relative_to(root.resolve())
    return target


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.github-copy-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def clipboard_lock():
    # Shared by all workers AND independent invocations by this OS user.
    with CLIPBOARD, open(Path(tempfile.gettempdir()) / f'github-copy-clipboard-{os.getuid()}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class Browser:
    def __init__(self, args):
        self.args = args
        self.session = 'gc' + uuid.uuid4().hex[:8]
        executable = find_chrome(args.chrome_path)
        self.config_dir = tempfile.TemporaryDirectory(prefix='github-copy-chrome-')
        config = Path(self.config_dir.name) / 'cli.json'
        config.write_text(json.dumps({'browser': {
            'browserName': 'chromium', 'isolated': True,
            'launchOptions': {'executablePath': executable, 'headless': args.headless},
        }}))
        try:
            retry(args, 'browser startup', lambda: self.call('open', 'about:blank', '--config=' + str(config)))
            if args.storage_state:
                self.call('state-load', str(Path(args.storage_state).resolve()))
        except Exception:
            self.close()
            raise

    def call(self, *args, result=False):
        if not shutil.which(self.args.cli):
            raise RuntimeError('CLI executable not found: ' + self.args.cli +
                               '. Run PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install in the project, '
                               'or supply --cli /path/to/playwright-cli.')
        p = subprocess.run([self.args.cli, '-s=' + self.session, *args],
                           capture_output=True, text=True, timeout=90,
                           env={**os.environ, 'PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD': '1'})
        if p.returncode or '### Error' in p.stdout:
            raise RuntimeError((p.stderr + p.stdout)[-2000:])
        if result:
            return parse_cli_result(p.stdout)

    def metadata(self, url):
        self.call('goto', url)
        return self.call('run-code', (HERE / 'browser/metadata.js').read_text(), result=True)

    def copy(self):
        with clipboard_lock():
            info = self.call('run-code', (HERE / 'browser/copy.js').read_text(), result=True)
        pieces = []
        for offset in range(0, info['encoded'], 24000):
            pieces.append(self.call('eval', f'window.__githubCopyBytes.slice({offset},{offset + 24000})', result=True))
        encoded = ''.join(pieces)
        data = base64.b64decode(encoded, validate=True)
        if len(encoded) != info['encoded'] or len(data) != info['bytes']:
            raise ValueError('Truncated browser-to-local transfer')
        return data

    def close(self):
        try:
            self.call('close')
        except Exception as e:
            print(f'Warning: could not close {self.session}: {e}', flush=True)
        finally:
            self.config_dir.cleanup()


def retry(args, label, fn):
    for attempt in range(args.retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == args.retries:
                raise
            delay = min(30, 2 ** (attempt + 1)) + random.random()
            print(f'Retry {attempt + 1}/{args.retries} {label}: {e}; waiting {delay:.1f}s', flush=True)
            time.sleep(delay)


def run(args):
    u = urlsplit(args.url)
    parts = u.path.strip('/').split('/')
    if u.scheme != 'https' or u.netloc != 'github.com' or len(parts) < 2 or (len(parts) > 2 and parts[2] not in ('tree', 'blob')):
        raise ValueError('Expected https://github.com/OWNER/REPO[/tree/REF/PATH]')
    if args.jobs < 1 or args.retries < 0:
        raise ValueError('jobs must be positive and retries nonnegative')
    root = Path(args.destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = 'https://github.com/' + '/'.join(parts[:2])
    browser = Browser(args)
    files, directories, errors, completed = [], [], [], []
    try:
        initial = retry(args, args.url, lambda: browser.metadata(args.url))
        commit, start = initial['commit'], initial['path']
        if not re.fullmatch('[0-9a-f]{40}', commit):
            raise ValueError('Missing immutable commit ID')
        prefix = str(PurePosixPath(start).parent) if initial.get('blob') else start
        if prefix == '.':
            prefix = ''
        prefix = prefix + '/' if prefix else ''

        def relative(path):
            if not path.startswith(prefix):
                raise ValueError('Entry outside requested subtree: ' + path)
            rel = path[len(prefix):]
            local_path(root, rel)
            return rel

        def url(kind, path):
            return f'{base}/{kind}/{commit}/' + quote(path, safe='/')

        if initial.get('blob'):
            files.append(start)
        else:
            pending, seen = [start], set()
            while pending:
                path = pending.pop()
                if path in seen:
                    raise ValueError('Duplicate directory: ' + path)
                seen.add(path)
                def listing():
                    m = browser.metadata(url('tree', path))
                    tree = m.get('tree')
                    if m['commit'] != commit or m['path'] != path or not tree:
                        raise ValueError('Wrong revision, path, or missing listing')
                    if len(tree['items']) != tree['totalCount']:
                        raise ValueError('GitHub directory listing is truncated; cannot claim a complete download')
                    return tree['items']
                entries = retry(args, path or '/', listing)
                for entry in entries:
                    child = entry['path']
                    # GitHub can collapse chains of single-child directories.
                    if not child.startswith(path + '/' if path else '') or child == path:
                        raise ValueError('Invalid child path: ' + child)
                    rel = relative(child)
                    if entry['contentType'] == 'directory':
                        local_path(root, rel).mkdir(parents=True, exist_ok=True)
                        directories.append(rel)
                        pending.append(child)
                    elif entry['contentType'] == 'file':
                        files.append(child)
                    else:
                        errors.append({'path':rel, 'error':'Unsupported entry type: ' + entry['contentType']})
                print(f'Listed {path or "/"}: {len(entries)} entries', flush=True)
    finally:
        browser.close()
    if len(set(files)) != len(files):
        raise ValueError('Duplicate file paths in listing')

    def worker(batch):
        b = Browser(args)
        try:
            results = []
            for path in batch:
                rel = relative(path)
                def download():
                    m = b.metadata(url('blob', path))
                    if m['path'] != path or m['commit'] != commit or not m.get('blob'):
                        raise ValueError('Wrong file page or revision')
                    blob = m['blob']
                    if blob['lfs'] or blob['binary']:
                        raise ValueError('Binary/LFS file cannot be faithfully copied as clipboard text')
                    target = local_path(root, rel)
                    if target.is_file() and git_hash(target.read_bytes()).startswith(blob['hash'] or '!'):
                        data = target.read_bytes()
                        status = 'verified-existing'
                    else:
                        data = validate(b.copy(), blob['hash'])
                        atomic_write(target, data)
                        if target.read_bytes() != data:
                            raise ValueError('Local read-back mismatch')
                        status = 'copied'
                    print(f'{status}: {rel} ({len(data)} bytes)', flush=True)
                    return {'path':rel, 'bytes':len(data), 'gitBlob':git_hash(data), 'githubHash':blob['hash'], 'status':status}
                try:
                    results.append(retry(args, rel, download))
                except Exception as e:
                    errors.append({'path':rel,'error':str(e)})
                    print(f'FAILED {rel}: {e}', flush=True)
            return results
        finally:
            b.close()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(worker, files[i::args.jobs]) for i in range(min(args.jobs, len(files)))]
        for f in as_completed(futures):
            completed.extend(f.result())
    # Audit after ALL workers finish, including collisions on case-insensitive disks
    # and changes made to earlier files while later files were being copied.
    for record in completed:
        try:
            data = local_path(root, record['path']).read_bytes()
            if len(data) != record['bytes'] or git_hash(data) != record['gitBlob']:
                raise ValueError('Final local content differs from verified copy')
        except Exception as e:
            errors.append({'path':record['path'], 'error':str(e)})
    report = {'source':args.url, 'commit':commit, 'destination':str(root), 'directories':sorted(directories),
              'expectedFiles':len(files), 'files':sorted(completed,key=lambda f:f['path']), 'errors':errors,
              'complete':not errors and len(completed)==len(files)}
    report_path = Path(args.report).resolve() if args.report else root.parent / (root.name + '.github-copy-report.json')
    atomic_write(report_path, (json.dumps(report,indent=2) + '\n').encode())
    print(f'{len(completed)}/{len(files)} files verified; {len(errors)} errors. Report: {report_path}', flush=True)
    return 0 if report['complete'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('url')
    p.add_argument('destination', help='Contents of the requested folder go directly here')
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--retries', type=int, default=3, help='Additional attempts per listing/file')
    display = p.add_mutually_exclusive_group()
    display.add_argument('--headed', dest='headless', action='store_false', help='Show installed Chrome windows (default)')
    display.add_argument('--headless', action='store_true', help='Run installed Chrome without visible windows; no separate browser download')
    p.set_defaults(headless=False)
    p.add_argument('--chrome-path', help='Path to an existing Google Chrome executable (otherwise auto-detected)')
    p.add_argument('--storage-state', help='playwright-cli state-save JSON for a private repository')
    p.add_argument('--report', help='Manifest path (default: next to destination)')
    p.add_argument('--cli', default=default_cli(), help='CLI executable (prefers the project-local pinned installation)')
    args = p.parse_args()
    try:
        return run(args)
    except Exception as e:
        p.exit(1, f'Error: {e}\n')


if __name__ == '__main__':
    raise SystemExit(main())
