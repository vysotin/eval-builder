#!/usr/bin/env bash
# Exercise every evalbuilder CLI tool and the pipeline end to end, offline, using whatever
# `python` / `evalbuilder` is on PATH — meant for environments without uv:
#
#     python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[ui,dev]"
#     bash scripts/cli-smoke.sh            # ~2 minutes, needs no model or key
#
# Runs on the smallest example (weather_bot, scripted agent + offline generator): check →
# discover → agent-map update → pipeline init → run --until dataset → review → --resume →
# report → dataset/mock tools → standalone run/score/simulate → the UI's job wrapper →
# the Streamlit server's health endpoint. Exit code 0 means every step succeeded.
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
run test -f "$W/out/dataset.json"; run test -f "$W/out/mock-rules.json"
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
step "standalone run / score / simulate on the pipeline's dataset"
run evalbuilder run "$W/out/dataset.json" --mock --out "$W/results" >/dev/null
runfile=$(ls "$W"/results/run-*.json 2>/dev/null | head -1); run test -n "$runfile"
run evalbuilder score "$runfile" --dataset "$W/out/dataset.json" --evaluators "$W/out/evaluators.yaml" --out "$W/results" >/dev/null
run test -n "$(ls "$W"/results/score-report-*.json 2>/dev/null | head -1)"
run evalbuilder simulate "$W/out/dataset.json" --scenarios "$W/out/scenarios.yaml" --out "$W/results" --no-mine >/dev/null
run test -n "$(ls "$W"/results/simulation-*.json 2>/dev/null | head -1)"
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
