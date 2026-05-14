import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SecurityCharts:
    def severity_distribution(self, issues: list[dict]) -> str:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in issues:
            sev = issue.get("severity", "").lower()
            if sev in counts:
                counts[sev] += 1

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c"]
        ax.bar(list(counts.keys()), list(counts.values()), color=colors)
        ax.set_title("Issue Severity Distribution")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return self._save_fig(fig, "severity_distribution.png")

    def top_vulnerable_apps(self, applications: list[dict]) -> str:
        apps = sorted(applications, key=lambda a: a.get("risk_score", 0), reverse=True)[:5]
        names = [a["name"] for a in apps]
        scores = [a["risk_score"] for a in apps]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(names, scores, color="#1565c0")
        ax.set_xlim(0, 10)
        ax.set_xlabel("Risk Score")
        ax.set_title("Top Vulnerable Applications")
        plt.tight_layout()
        return self._save_fig(fig, "top_vulnerable_apps.png")

    def _save_fig(self, fig: plt.Figure, filename: str) -> str:
        fig.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return filename
