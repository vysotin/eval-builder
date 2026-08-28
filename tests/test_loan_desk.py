"""The loan_desk example: routing + tool gating, schema discovery, offline pipeline e2e."""

import json
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from evalbuilder.config import Settings
from evalbuilder.discover import discover_from_source
from evalbuilder.pipeline.config import PIPELINE_CONFIG_SCHEMA, load_config
from evalbuilder.pipeline.report import run_pipeline
from evalbuilder.tool_schemas import describe_tool
from examples.loan_desk import agent as ld
from examples.loan_desk import offline

SOURCE = Path("examples/loan_desk/agent.py")
AGENT = "scripted:examples.loan_desk.agent:default_scripted_model"
GENERATOR = "scripted:examples.loan_desk.offline:generator_model"
TOOL_NAMES = ["get_customer_profile", "credit_check", "quote_installment", "document_status", "submit_application"]


def _mock_tools(profile=offline.PROFILE, quote=offline.QUOTE, doc=offline.PAYSLIP, receipt=offline.RECEIPT):
    calls = []

    @tool
    def get_customer_profile(customer_id: str) -> dict:
        """mock"""
        calls.append(("get_customer_profile", customer_id))
        return profile

    @tool(args_schema=ld.CreditQuery)
    def credit_check(customer_id: str, consent: bool, ssn_last4: str) -> dict:
        """mock"""
        calls.append(("credit_check", ssn_last4))
        return offline.CREDIT

    @tool
    def quote_installment(terms: ld.LoanTerms) -> dict:
        """mock"""
        calls.append(("quote_installment", terms.model_dump()))
        return quote

    @tool
    def document_status(doc_id: str) -> dict:
        """mock"""
        calls.append(("document_status", doc_id))
        return doc

    @tool
    def submit_application(application: ld.LoanApplication) -> dict:
        """mock"""
        calls.append(("submit_application", application.model_dump()))
        return receipt

    return [get_customer_profile, credit_check, quote_installment, document_status, submit_application], calls


def _ask(graph, text, messages=None):
    messages = list(messages or []) + [{"role": "user", "content": text}]
    return graph.invoke({"messages": messages})


def _path(graph, text):
    path = []
    for chunk in graph.stream({"messages": [{"role": "user", "content": text}]}, stream_mode="updates"):
        path.extend(chunk)
    return path


def _last(state) -> str:
    return state["messages"][-1].content


# ── routing and tool gating ────────────────────────────────────


def test_eligibility_quotes_and_gates_credit_check_on_consent():
    tools, calls = _mock_tools()
    graph = ld.build_agent(tools=tools)
    out = _ask(graph, "Am I eligible for a 20,000 personal loan over 24 months? My customer id is C10001.")
    assert out["route"] == "eligibility"
    assert [c[0] for c in calls] == ["get_customer_profile", "quote_installment"]
    assert calls[1][1] == {"amount": 20000.0, "months": 24, "product": "personal"}
    assert "312.50" in _last(out) and "consent" in _last(out).lower()
    out2 = _ask(graph, "Yes, I consent — SSN last four 6789.", out["messages"])
    assert calls[-1] == ("credit_check", "6789")
    assert "712" in _last(out2)


def test_apply_gates_submit_on_yes():
    tools, calls = _mock_tools()
    graph = ld.build_agent(tools=tools)
    out = _ask(graph, offline.APPLY_MESSAGE)
    assert out["route"] == "apply"
    assert [c[0] for c in calls] == ["get_customer_profile", "quote_installment"]
    assert "Reply yes to submit" in _last(out)
    out2 = _ask(graph, "yes", out["messages"])
    assert calls[-1][0] == "submit_application"
    assert calls[-1][1]["terms"] == {"amount": 10000.0, "months": 36, "product": "personal"}
    assert calls[-1][1]["purpose"] == "consolidate my credit cards"
    assert "APP-000123" in _last(out2)


def test_out_of_range_terms_never_call_side_effecting_tools():
    tools, calls = _mock_tools()
    graph = ld.build_agent(tools=tools)
    assert "1,000 and 500,000" in _last(_ask(graph, "I'd like to apply for a 900 loan over 18 months, customer C10001, to fix my bike."))
    assert "1,000 and 500,000" in _last(_ask(graph, "Customer C10001, can I borrow 2,000,000 over 60 months for a house?"))
    assert "12, 24, 36, 48 or 60" in _last(_ask(graph, "Customer C10001, quote me a 20,000 personal loan over 18 months."))
    assert calls == []


def test_missing_customer_id_is_requested_not_guessed():
    tools, calls = _mock_tools()
    graph = ld.build_agent(tools=tools)
    assert "customer id" in _last(_ask(graph, "Am I eligible for a 20,000 personal loan over 24 months?"))
    assert "customer id" in _last(_ask(graph, "Run a credit check, I consent — my customer id is 12345 and SSN last four 4321."))
    assert calls == []


def test_documents_chain_into_advisor():
    tools, calls = _mock_tools()
    graph = ld.build_agent(tools=tools)
    assert _path(graph, "Did you receive my payslip DOC-77? Customer C10001.") == ["route", "documents_agent", "advisor_agent"]
    out = _ask(graph, "Did you receive my payslip DOC-77? Customer C10001.")
    assert out["route"] == "documents" and [c for c in calls if c[0] == "document_status"]
    texts = [m.content for m in out["messages"] if isinstance(m, AIMessage) and m.content]
    assert texts[-2].startswith("Document DOC-77") and "DOC-77 is received" in texts[-1]
    assert "document id" in _last(_ask(graph, "Did you receive my payslip yet? I am customer C10001."))
    assert "payslip, bank statement or id card" in _last(_ask(graph, "Did you receive my passport scan DOC-77? Customer C10001."))
    # _path and _ask each looked DOC-77 up once; the passport request never reached the tool
    assert [c for c in calls if c[0] == "document_status"] == [("document_status", "DOC-77")] * 2


def test_out_of_scope_declines_without_tools():
    tools, calls = _mock_tools()
    out = _ask(ld.build_agent(tools=tools), "Write me a poem about the sea.")
    assert out["route"] == "other" and not calls
    assert _last(out) == ld.DECLINE_TEXT


def test_incomplete_payloads_are_reported_not_invented():
    tools, _ = _mock_tools(quote={"apr_pct": 7.9, "total_cost": 1.0, "currency": "USD"})
    assert "incomplete" in _last(_ask(ld.build_agent(tools=tools), "Customer C10001, how much per month would a 12,000 auto loan over 36 months cost?"))
    tools, _ = _mock_tools(receipt={"status": "submitted", "next_step": "x"})
    graph = ld.build_agent(tools=tools)
    out = _ask(graph, offline.APPLY_MESSAGE)
    assert "incomplete" in _last(_ask(graph, "yes", out["messages"]))


def test_real_tools_fail_fast_when_unmocked():
    out = _ask(ld.build_agent(), "Am I eligible for a 20,000 personal loan over 24 months? My customer id is C10001.")
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert tool_msgs and tool_msgs[0].status == "error" and "unavailable" in tool_msgs[0].content
    assert "could not reach" in _last(out).lower()
    out = _ask(ld.build_agent(), "Did you receive my payslip DOC-77?")
    assert "could not reach" in _last(out).lower()


def test_tools_and_contract():
    assert [t.name for t in ld.TOOLS] == TOOL_NAMES
    assert "Side-effecting" in ld.credit_check.description and "Side-effecting" in ld.submit_application.description
    assert ld.get_customer_profile.handle_tool_error is True


# ── discovery ──────────────────────────────────────────────────


def test_ast_discovery_captures_schemas_and_edges():
    amap = discover_from_source(SOURCE)
    by_name = {t["name"]: t for t in amap.tools}
    assert list(by_name) == TOOL_NAMES
    credit = by_name["credit_check"]
    assert credit["schema_source"] == "args_schema" and credit["side_effecting"] is True
    ssn = credit["args_schema"]["properties"]["ssn_last4"]
    assert ssn["minLength"] == 4 and ssn["maxLength"] == 4 and ssn["pattern"] == r"^\d{4}$"
    assert credit["args_schema"]["properties"]["customer_id"]["pattern"] == r"^C\d{5}$"
    assert credit["args_schema"]["required"] == ["customer_id", "consent", "ssn_last4"]
    submit = by_name["submit_application"]
    assert submit["side_effecting"] is True
    assert set(submit["args_schema"]["$defs"]) == {"LoanApplication", "LoanTerms"}
    terms = submit["args_schema"]["$defs"]["LoanTerms"]
    assert terms["properties"]["months"]["enum"] == [12, 24, 36, 48, 60]
    assert terms["properties"]["amount"] == {"type": "number", "minimum": 1000, "maximum": 500000}
    assert submit["output_schema"]["title"] == "ApplicationReceipt"
    assert by_name["quote_installment"]["output_schema"]["title"] == "InstallmentQuote"
    assert by_name["get_customer_profile"]["side_effecting"] is False
    kinds = {e["kind"] for t in amap.tools for e in t["edge_cases"]}
    assert kinds == {"missing_required", "wrong_type", "out_of_enum", "boundary", "malformed_output"}
    enum_edge = next(e for e in by_name["document_status"]["edge_cases"] if e["kind"] == "out_of_enum")
    assert enum_edge["field"] == "doc_type" and "'payslip'" in enum_edge["detail"]
    assert [e["kind"] for e in by_name["credit_check"]["edge_cases"][:3]] == ["missing_required", "wrong_type", "boundary"]
    llm_nodes = {n["id"]: n for n in amap.graph["nodes"] if n["kind"] == "llm"}
    assert set(llm_nodes) == {"eligibility_agent", "documents_agent", "advisor_agent"}
    assert "credit_check" in llm_nodes["eligibility_agent"]["tools"]
    assert amap.graph["conditional_edges"][0]["source"] == "route"
    assert ["documents_agent", "advisor_agent"] in amap.graph["edges"]


def test_live_tool_schemas():
    profile = describe_tool(ld.get_customer_profile)
    assert profile["output_schema"]["required"] == ["customer_id", "name", "segment", "annual_income", "existing_debt", "kyc_verified"]
    assert profile["output_schema"]["properties"]["customer_id"]["pattern"] == r"^C\d{5}$"
    submit = describe_tool(ld.submit_application)
    assert "LoanTerms" in submit["args_schema"]["$defs"] and "LoanApplication" in submit["args_schema"]["$defs"]
    assert submit["output_schema"]["properties"]["status"]["enum"] == ["submitted", "pending_documents"]
    credit = describe_tool(ld.credit_check)
    assert credit["schema_source"] == "args_schema" and credit["side_effecting"] is True


def test_pipeline_yaml_is_valid():
    cfg = load_config(Path("examples/loan_desk/pipeline.yaml"))
    assert cfg.problems() == [] and cfg.name == "loan-desk"


# ── offline end-to-end pipeline ────────────────────────────────


def _config(tmp_path, per_tool_edge_cases):
    cfg = {
        "schema": PIPELINE_CONFIG_SCHEMA,
        "name": "loan-desk-test",
        "target": {"source": str(SOURCE), "module": "examples.loan_desk.agent"},
        "models": {"agent": AGENT, "judge": AGENT, "generator": GENERATOR},
        "constraints": ["Never call credit_check without explicit consent.", "Never call submit_application before the customer says yes."],
        "coverage": {"total_cases": 8, "per_intent": {"happy": 1, "failure": 1}, "per_failure_category": 1,
                     "out_of_intent": 1, "multi_turn_share": 0.0, "per_tool_edge_cases": per_tool_edge_cases},
        "evaluators": [{"type": "expected_tools"}, {"type": "contains"}],
        "thresholds": {"default": 0.8, "slice_min": 0.5, "overall_pass": 0.8},
        "runs": {"repeats": 2},
        "stages": {"simulate": True, "publish": "never", "max_retries": 1},
        "review": {"auto_approve": True, "approved_by": "tester", "note": "offline e2e"},
        "output": {"dir": str(tmp_path / f"out{per_tool_edge_cases}")},
    }
    path = tmp_path / f"pipeline{per_tool_edge_cases}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _run(tmp_path, per_tool_edge_cases):
    cfg_path = _config(tmp_path, per_tool_edge_cases)
    _, report = run_pipeline(cfg_path, log=lambda m: None, settings=Settings())
    assert report["verdict"] == "pass" and report["overall_score"] == 1.0, (report["verdict_reasons"], report["problems"])
    assert report["coverage"]["coverage_pct"] == 100.0 and report["coverage"]["gaps"] == []
    assert "schema-edge" in report["coverage"]["by_kind"]
    assert report["agent"]["tools"] == TOOL_NAMES
    assert report["stability"]["repeats"] == 2 and report["stability"]["unstable_cases"] == []
    assert report["simulation"]["details"]["stop_reasons"] == {"apply-and-confirm": "success"}
    ds = json.loads((tmp_path / f"out{per_tool_edge_cases}" / "dataset.json").read_text())
    return report, ds["cases"]


def test_offline_pipeline_with_one_edge_per_tool(tmp_path):
    report, cases = _run(tmp_path, 1)
    edges = [c["metadata"]["edge"] for c in cases if c["metadata"].get("edge")]
    assert {e["kind"] for e in edges} == {"missing_required"} and len(edges) == 5
    assert report["coverage"]["by_kind"]["schema-edge"] == {"planned": 5, "covered": 5}
    assert all(c["review"]["status"] == "approved" for c in cases)
    assert all(e["id"].startswith("edge.") for e in edges)


def test_offline_pipeline_with_three_edges_per_tool(tmp_path):
    report, cases = _run(tmp_path, 3)
    kinds = {c["metadata"]["edge"]["kind"] for c in cases if c["metadata"].get("edge")}
    assert kinds == {"missing_required", "wrong_type", "out_of_enum", "boundary", "malformed_output"}
    malformed = [c for c in cases if (c["metadata"].get("edge") or {}).get("kind") == "malformed_output"]
    assert {c["metadata"]["tool"] for c in malformed} == {"get_customer_profile", "quote_installment", "submit_application"}
    missing_field = {"get_customer_profile": "customer_id", "quote_installment": "monthly_payment", "submit_application": "application_id"}
    for c in malformed:
        tool_name = c["metadata"]["tool"]
        injected = c["metadata"]["mocks"]["tools"][tool_name][0]
        assert injected["matchArgs"] == {} and missing_field[tool_name] not in injected["response"]
        assert c["reference_outputs"]["contains"] == "incomplete"
    planned = sum(min(3, len(describe_tool(t)["edge_cases"])) for t in ld.TOOLS)
    assert report["coverage"]["by_kind"]["schema-edge"] == {"planned": planned, "covered": planned}
