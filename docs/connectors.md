# Connectors Guide

Connectors allow the security platform to ingest vulnerability data from external services.
Navigate to **Settings → Integrations** to manage connectors.

Available connectors: **Jira**, **GitHub**, **AWS Security Hub**, and **Slack**.

## Jira Connector

The Jira connector syncs security issues as Jira tickets for team triage.

### Setup Steps
1. Go to Settings → Integrations → Jira.
2. Enter your Jira site URL (e.g. `https://yourcompany.atlassian.net`).
3. Generate an API token in your Jira profile under **Security → API tokens**.
4. Enter your email and API token, then click **Connect**.
5. Select the target Jira project where tickets will be created.
6. Choose a severity threshold — only issues at or above this level create tickets.

### Supported Features
- Auto-create tickets for new critical and high issues.
- Sync issue status between platform and Jira (open/resolved).
- Attach CVE references and remediation notes to tickets.

### Troubleshooting
- **"401 Unauthorized"**: The API token may have expired. Regenerate it in Jira.
- **"Project not found"**: Ensure the Jira project key matches exactly (case-sensitive).
- **Tickets not syncing**: Verify the webhook URL is whitelisted in your Jira firewall rules.

---

## GitHub Connector

The GitHub connector scans repositories for secrets, vulnerable dependencies, and code vulnerabilities.

### Setup Steps
1. Go to Settings → Integrations → GitHub.
2. Click **Install GitHub App** — this redirects to GitHub.
3. Select the organization or repositories to grant access.
4. Click **Install & Authorize**.
5. The platform will begin scanning within 15 minutes of installation.

### Supported Features
- Dependency vulnerability scanning (via `package.json`, `requirements.txt`, `pom.xml`, etc.).
- Secret detection (API keys, tokens, credentials committed to code).
- Code scanning results ingested from GitHub Advanced Security (GHAS).
- Pull request annotations for new findings.

### Troubleshooting
- **App not appearing in GitHub**: Clear browser cache and retry the OAuth flow.
- **Scans not running**: Ensure the GitHub App has `read` permissions on repository contents.
- **Missing repos**: Re-install the app and select all repositories explicitly.

---

## AWS Security Hub Connector

The AWS connector ingests findings from Security Hub, GuardDuty, and Inspector.

### Setup Steps
1. Go to Settings → Integrations → AWS.
2. Enter your AWS Account ID and the region of your Security Hub instance.
3. Create an IAM role in your AWS account with the following policy:
   ```json
   {
     "Effect": "Allow",
     "Action": ["securityhub:GetFindings", "guardduty:ListFindings"],
     "Resource": "*"
   }
   ```
4. Enter the IAM role ARN and click **Verify & Connect**.
5. Select which finding types to import (Security Hub, GuardDuty, Inspector, or all).

### Troubleshooting
- **"Access Denied"**: The IAM role trust policy must allow `sts:AssumeRole` from the platform's AWS account.
- **No findings appearing**: Confirm Security Hub is enabled in the specified region.
- **Duplicate findings**: Disable native Security Hub cross-region aggregation to avoid double-ingestion.

---

## Slack Connector

The Slack connector sends real-time alerts for new critical issues to a Slack channel.

### Setup Steps
1. Go to Settings → Integrations → Slack.
2. Click **Add to Slack** — this redirects to Slack's OAuth page.
3. Select the workspace and the channel for alerts.
4. Click **Allow**.
5. Set the minimum severity for notifications (default: Critical).

### Troubleshooting
- **Bot not posting**: Ensure the bot has `chat:write` permission in the target channel.
- **Missing alerts**: Check the severity filter — Medium and Low are off by default.
