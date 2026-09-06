#!/usr/bin/env bash
# Exercise every evalbuilder CLI tool and the pipeline end to end, offline, using whatever
# `python` / `evalbuilder` is on PATH — meant for environments without uv:
#
#     python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[ui,dev]"
#     bash scripts/cli-smoke.sh            # ~2 minutes, needs no model or key
#
# Runs on the smallest example (weather_bot, scripted agent + offline generator): check →
# discover → agent-map update → pipeline init → run --until dataset → review → --resume
# (deploy on the local target → infer → simulate → teardown → score) → report → dataset/mock
# tools → the phase commands on their own (deploy render/up/status → infer --deployment →
# eval --aggregate → deploy down) → standalone simulate → the UI's job wrapper → the
# Streamlit server's health endpoint. Exit code 0 means every step succeeded.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="${SMOKE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/evalbuilder-smoke.XXXXXX")}"
PORT="${SMOKE_PORT:-8765}"
cd "$ROOT"
fail=0
step() { echo; echo "### $*"; }
run() { "$@"; local rc=$?; if [ $rc -ne 0 ]; then echo "!! FAILED rc=$rc: $*" >&2; fail=1; fi; }

step "interpreter"; run python --version; run python -c "import sys; assert sys.version_info >= (3, 11, 7), sys.version"
step "console script + module entry point"; run evalbuilder --help >/dev/null; run python -m evalbuilder.cli --help >/dev/null
step "check"; run evalbuilder check >/dev/null
step "discover"; run evalbuilder discover examples.weather_bot.agent --source examples/weather_bot/agent.py --eval-dir "$W/eval" >/dev/null
run test -f "$W/eval/agent-map.json"
step "agent-map update"; run evalbuilder agent-map update "$W/eval/agent-map.json" --intents '[{"id":"intent.conditions","status":"hypothesis","evidence":["tool:get_weather"]}]' >/dev/null
step "pipeline init"; run evalbuilder pipeline init "$W/weather.yaml" --name weather-smoke --source examples/weather_bot/agent.py --module examples.weather_bot.agent >/dev/null
python - "$W" <<'PY'
import sys, pathlib, yaml
w = pathlib.Path(sys.argv[1]); p = w / "weather.yaml"; cfg = yaml.safe_load(p.read_text())
scripted = "scripted:examples.weather_bot.agent:default_scripted_model"
cfg["models"] = {"agent": scripted, "judge": scripted, "generator": "scripted:examples.weather_bot.offline:generator_model"}
cfg["coverage"] = {"total_cases": 4, "per_intent": {"happy": 1, "failure": 1}, "per_failure_category": 1, "out_of_intent": 1, "multi_turn_share": 0.2, "per_tool_edge_cases": 1}
cfg["evaluators"] = [{"type": "expected_tools"}, {"type": "contains"}]
cfg["runs"] = {"repeats": 2}; cfg["stages"] = {"simulate": True, "publish": "never", "max_retries": 1}
cfg["review"] = {"auto_approve": False, "approved_by": ""}; cfg["output"] = {"dir": str(w / "out")}
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
step "pipeline run --until dataset"; run evalbuilder pipeline run "$W/weather.yaml" --until dataset >/dev/null
run test -f "$W/out/dataset.json"; run test -f "$W/out/work/mock-rules.json"
run test ! -e "$W/out/coverage.json"   # folded into dataset.coverage.achieved
step "dataset validate / list, mock verify"
run evalbuilder dataset validate "$W/out/dataset.json" >/dev/null
run evalbuilder dataset list "$W/out/dataset.json" >/dev/null
run evalbuilder mock verify "$W/out/dataset.json" >/dev/null
step "review: reject one case, approve the rest"
first=$(python -c "import json; print(json.load(open('$W/out/dataset.json'))['cases'][0]['id'])")
rest=$(python -c "import json; print(','.join(c['id'] for c in json.load(open('$W/out/dataset.json'))['cases'][1:]))")
run evalbuilder review "$W/out/dataset.json" --reject "$first" --note "smoke" >/dev/null
run evalbuilder review "$W/out/dataset.json" --approve "$rest" --note "approved by the smoke script" >/dev/null
step "pipeline run --resume"; run evalbuilder pipeline run "$W/weather.yaml" --resume >/dev/null
step "pipeline report"; run evalbuilder pipeline report "$W/out"
run python -c "import json; r=json.load(open('$W/out/report.json')); assert r['verdict']=='pass', r['verdict_reasons']"
step "artifact layout: deliverables at the root, scratch in work/, nothing duplicated"
run python - "$W/out" <<'PY2'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
names = {p.name for p in out.iterdir()} - {"pipeline.log"}  # only UI-launched jobs write the log
assert names == {
    "agent-map.json", "aggregate.json", "analysis.json", "dataset.json", "deployment.json", "evaluators.yaml",
    "report.json", "results", "scenarios.yaml", "simulation.json", "work",
}, sorted(names)
assert "deploy" in {p.name for p in (out / "work").iterdir()}, sorted(p.name for p in (out / "work").iterdir())
ds = json.loads((out / "dataset.json").read_text())
amap = json.loads((out / "agent-map.json").read_text())
assert ds["coverage"]["plan"]["cells"] and ds["coverage"]["achieved"]["planned"] and amap["applicable_failures"]
assert ds["mocks"]["tools"] == json.loads((out / "work" / "mock-rules.json").read_text())["tools"]
report = json.loads((out / "report.json").read_text())
deployment = json.loads((out / "deployment.json").read_text())
assert deployment["schema"] == "evalbuilder/deployment/v1" and deployment["status"] == "down", deployment["status"]
assert report["deployment"]["target"] == "local" and report["stages"]["infer"]["details"]["execution"]["mode"] == "remote"
assert {report["stages"][s]["status"] for s in ("deploy", "infer", "teardown")} == {"ok"}, report["stages"]
PY2
step "pipeline compact is a no-op on a current directory"
run python -c "from evalbuilder.pipeline.layout import compact_dir; r=compact_dir('$W/out'); assert r=={'folded':[],'moved':[],'removed':[]}, r"
step "phase commands: deploy render (kubernetes files) / up (local) / status → infer --deployment → eval --aggregate → deploy down"
run python -c "
import json, subprocess, sys
out = json.loads(subprocess.run(['evalbuilder', 'deploy', 'render', '$W/weather.yaml', '--target', 'kubernetes'], capture_output=True, text=True, check=True).stdout)
assert set(out['files']) == {'Dockerfile', 'manifests.yaml', 'loader.yaml'}, sorted(out['files'])
assert 'evalbuilder serve' in out['files']['Dockerfile'] and out['image'] == 'evalbuilder-weather-smoke:latest', out['image']
"
run evalbuilder deploy up "$W/weather.yaml" --quiet >/dev/null
run evalbuilder deploy status "$W/out" >/dev/null
run evalbuilder infer "$W/out/dataset.json" --deployment "$W/out" --repeats 2 --workers 2 --out "$W/results" >/dev/null
run python -c "
import json, pathlib
runs = sorted(pathlib.Path('$W/results').glob('run-*.json')); assert len(runs) == 2, runs
run = json.loads(runs[0].read_text()); assert run['execution']['mode'] == 'remote' and run['case_runs'][0]['log'], run['execution']
"
run evalbuilder eval "$W"/results/run-*.json --dataset "$W/out/dataset.json" --evaluators "$W/out/evaluators.yaml" --out "$W/results" --workers 2 --aggregate --config "$W/weather.yaml" >/dev/null
run test -n "$(ls "$W"/results/score-report-*.json 2>/dev/null | head -1)"
run python -c "import json; a=json.load(open('$W/results/aggregate.json')); assert a['verdict']=='pass' and a['repeats']==2, (a['verdict'], a['repeats'])"
run evalbuilder deploy down "$W/out" >/dev/null
if evalbuilder deploy status "$W/out" >/dev/null 2>&1; then echo "!! FAILED: deployment still up after deploy down" >&2; fail=1; fi
step "standalone infer (in-process) / eval / simulate on the pipeline's dataset"
run evalbuilder infer "$W/out/dataset.json" --out "$W/results-local" >/dev/null
runfile=$(ls "$W"/results-local/run-*.json 2>/dev/null | head -1); run test -n "$runfile"
run evalbuilder eval "$runfile" --dataset "$W/out/dataset.json" --evaluators "$W/out/evaluators.yaml" --out "$W/results-local" >/dev/null
run test -n "$(ls "$W"/results-local/score-report-*.json 2>/dev/null | head -1)"
run evalbuilder simulate "$W/out/dataset.json" --scenarios "$W/out/scenarios.yaml" --out "$W/results-local" --no-mine >/dev/null
run test -n "$(ls "$W"/results-local/simulation-*.json 2>/dev/null | head -1)"
step "background job wrapper (what the UI runs)"
python - "$W" <<'PY'
import sys, time, pathlib, yaml
from evalbuilder.pipeline import jobs
w = pathlib.Path(sys.argv[1]); out = w / "jobout"
cfg = yaml.safe_load((w / "weather.yaml").read_text())
cfg["output"] = {"dir": str(out)}; cfg["review"] = {"auto_approve": True, "approved_by": "smoke job"}
(w / "job.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
jobs.start_job(w / "job.yaml", out, mode="full")
for _ in range(600):
    status = jobs.job_status(out)
    if status["status"] == "finished":
        break
    time.sleep(0.5)
assert status["status"] == "finished" and status["exit_code"] == 0, (status, jobs.tail_log(out))
print("job ok:", {k: v["status"] for k, v in status["stages"].items()})
PY
[ $? -ne 0 ] && fail=1
step "UI server health (headless)"
python -m streamlit run src/evalbuilder/ui/app.py --server.port "$PORT" --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none -- --dir "$W/out" > "$W/ui.log" 2>&1 &
uipid=$!
ok=0; for _ in $(seq 1 60); do if curl -fs "http://127.0.0.1:$PORT/_stcore/health" 2>/dev/null | grep -q ok; then ok=1; break; fi; sleep 1; done
if [ $ok -eq 1 ]; then echo "ui healthy on :$PORT"; else echo "!! UI did not become healthy" >&2; fail=1; tail -20 "$W/ui.log"; fi
kill $uipid 2>/dev/null; wait $uipid 2>/dev/null
run evalbuilder ui --help >/dev/null
echo; echo "SMOKE RESULT: $([ $fail -eq 0 ] && echo OK || echo FAILED) (work dir $W)"
exit $fail
