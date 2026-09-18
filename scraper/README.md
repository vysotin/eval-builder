# GitHub browser copier

Recursively copies a GitHub repository or folder through **playwright-cli**. It visits every folder page, opens each file page, clicks **Copy raw file**, reads the browser clipboard, and writes the verified bytes locally. No git clone, ZIP, GitHub API, HTTP client, raw URL fetch, or browser download is used.

## Run

Requires macOS or Linux, Python 3.10+, Node.js/npm, and Chrome. Install the CLI while package access is available (or provision it before entering the restricted network):

```sh
npm install -g @playwright/cli
python3 github_copy.py \
  https://github.com/vysotin/eval-builder/tree/main/examples \
  ./downloaded-examples --jobs 3
```

The contents of `examples` go directly into `downloaded-examples`. Repository URLs and individual `/blob/` URLs also work. Quote URLs and destination paths containing shell metacharacters or spaces.

```sh
python3 github_copy.py https://github.com/OWNER/REPO ./repo-copy
python3 github_copy.py 'https://github.com/OWNER/REPO/tree/BRANCH/path' './my folder' --headed --retries 5
```

Options:

- `--jobs N`: parallel browser workers (default 1). Navigation runs in parallel; a thread lock and OS file lock serialize clipboard write/click/read across workers and separate script invocations by the same user. Avoid manually changing the clipboard while copying.
- `--retries N`: additional attempts per directory or file (default 3), with exponential backoff and jitter.
- `--headed`: show browser windows.
- `--storage-state FILE`: load an authenticated Playwright storage state into each browser for private repositories. Create it using `playwright-cli -s=login open https://github.com --headed`, sign in manually, then `playwright-cli -s=login state-save /absolute/path/auth.json`. Keep this credential file private.
- `--cli PATH`: use a specific installed executable or the Playwright skill's `playwright_cli.sh` wrapper.
- `--report FILE`: report location; defaults to `DESTINATION.github-copy-report.json` beside the destination.

The script uses named browser sessions and closes only its own sessions. It changes the clipboard. Files at matching destination paths are replaced atomically after verification; unrelated files remain. Rerunning verifies existing files against the selected commit and skips matching content.

## Completeness checks

- Resolve the supplied URL in the browser, then pin all traversal and file pages to the resolved commit. Branch names containing slashes are resolved by GitHub.
- Compare each folder's embedded page entry count with GitHub's total count. Truncated or unsupported listings fail instead of silently omitting files.
- Compare copied bytes with the **Git blob SHA-1 prefix exposed by GitHub** (usually 7 hexadecimal characters). This is an accidental-corruption check, not full-hash cryptographic authentication. No file passes solely because two clipboard reads agree.
- Recover a missing final newline or normalized CRLF only when the resulting bytes match that hash.
- Transfer the captured bytes as bounded base64 chunks; check their lengths and validate before writing. Read back the local file after atomic replacement.
- Write a JSON report with revision, paths, byte counts, locally computed full hashes, GitHub hash prefixes, and errors. Exit 0 only when every discovered supported file was verified and no unsupported entries were found; otherwise exit nonzero. A fatal discovery error exits immediately and does not produce a new successful report.

## Limits

This is a **text-content copier**, not a Git checkout. Clipboard text cannot faithfully transport arbitrary binary files, LFS objects, or every legacy encoding. These fail explicitly. Symlinks and submodules are reported as unsupported; executable bits and Git history are not preserved. Very large files without a working Copy Raw button, folders exceeding GitHub's displayed listing limit, authentication walls, and UI changes also fail explicitly rather than producing an apparently complete copy.

GitHub's embedded page metadata is used only for traversal, revision, and validation; file content comes from the copy button. GitHub can change its UI or embedded metadata schema. Current selectors live in `browser/metadata.js` and `browser/copy.js`.

## Verification

```sh
python3 -m unittest discover -s tests -v
```

Tests cover newline/CRLF/Unicode/empty-file recovery, rejection of truncated and stale content, unsafe destination paths and symlinks, atomic writes, retry exhaustion, recursive traversal, commit pinning, and truncated directory detection. Live-run evidence is stored under `output/playwright/`.

Verified on 2026-09-18: the example folder at commit `dc9bc7945376849a0c821eee67cd1e8b04bacd03` copied 25 files across 13 descendant directories (152,682 bytes) with 3 workers, zero retries, and matching blob hashes. The downloaded files are in `output/playwright/examples-clean/`, with the adjacent `examples-clean.github-copy-report.json` manifest and `clean-run.log`. Repository-root copying and single-file resume were also tested against `octocat/Hello-World`.

CLI reference: https://github.com/microsoft/playwright-cli
