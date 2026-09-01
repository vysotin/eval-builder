# Severity matrix

| level | meaning | examples |
|---|---|---|
| sev1 | full outage or data loss for most customers | 5xx on every request, checkout down |
| sev2 | degraded service, workaround exists | elevated error rate, slow responses |
| sev3 | minor impact, no customer-visible outage | single endpoint slow, internal tooling |

Anything outside sev1–sev3 (critical, urgent, P0, sev9, high) is not a level: offer the
three values and let the reporter choose.
