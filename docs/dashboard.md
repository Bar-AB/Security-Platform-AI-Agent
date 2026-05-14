# Dashboard Guide

The dashboard gives a real-time view of your organization's security posture.

## Severity Levels

The platform uses four severity levels, aligned with CVSS scores:

| Severity | CVSS Range | Meaning | SLA |
|----------|-----------|---------|-----|
| Critical | 9.0–10.0 | Immediate exploitation risk, patch or mitigate within 24h | 24 hours |
| High | 7.0–8.9 | Likely exploitable, requires urgent attention | 7 days |
| Medium | 4.0–6.9 | Exploitable under specific conditions | 30 days |
| Low | 0.1–3.9 | Minimal risk, fix in next regular release cycle | 90 days |

## Risk Score

Each application is assigned a **Risk Score** from 0 to 10, calculated from:
- Number and severity of open issues
- Time open beyond SLA
- Asset criticality (business impact rating)
- Exposure level (internet-facing vs. internal)

A score above **8.0** is considered high risk and triggers an alert to the application owner.

## Dashboard Filters

Use the filter bar at the top of the dashboard to narrow findings:

- **Severity**: Filter by Critical / High / Medium / Low (multi-select).
- **Status**: Open, In Progress, Resolved.
- **Category**: Injection, XSS, Broken Auth, Exposed Data, Misconfiguration, Dependency.
- **Application**: Filter by one or more application names.
- **Date range**: Discovered within a time window.
- **Assigned to**: Filter by team member responsible for remediation.

Filters are combinable. URL state is preserved — share filtered views with teammates via the browser URL.

## Pipeline Security Tab

The **Pipeline** tab shows security findings from CI/CD runs:

- **Tool**: The scanner that found the issue (Semgrep, Trivy, Gitleaks, OWASP ZAP, etc.).
- **Stage**: The pipeline stage where it was detected (build, test, deploy).
- **Branch**: The Git branch the pipeline ran on.
- **Commit**: The SHA of the commit that triggered the finding.

Pipeline findings auto-close when the issue is no longer detected in a subsequent scan on the same branch.

## Understanding the Issues Table

Columns in the main issues table:
- **ID**: Unique issue identifier (e.g. ISS-001).
- **Title**: Short description of the vulnerability.
- **CVE**: Linked CVE identifier if applicable.
- **Application**: The service or app where the issue was found.
- **Category**: Vulnerability class.
- **Severity**: Color-coded severity badge.
- **Status**: Current remediation status.
- **Discovered**: Date the issue was first detected.

Click any row to view the full issue detail including description, remediation steps, and history.

## Exporting Data

Export the current filtered view as CSV or PDF via the **Export** button (top right).
Exports respect all active filters.
