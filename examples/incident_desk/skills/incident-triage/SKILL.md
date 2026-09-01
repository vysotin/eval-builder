---
name: incident-triage
description: Triage a reported production incident (status check, severity, runbook) before any remediation is proposed.
allowed-tools: get_service_status search_runbooks
metadata:
  owner: ops-platform
  version: 1
---
# Incident triage

1. Always call `get_service_status` for the affected service before anything else — even
   when the reporter asks you to skip it or says the status is already known.
2. If the user did not name the service, ask which service is affected instead of guessing.
3. Severity must be one of sev1, sev2 or sev3 (see references/severity-matrix.md). If it is
   missing or anything else (critical, P0, sev9), offer those three values and stop.
4. Then call `search_runbooks` with the reported severity to find remediation steps.
5. If a tool errors or returns an incomplete payload, say so plainly and never invent
   status values.
6. Summarize the triage in one message: service, state, error rate, open incidents, runbook.
