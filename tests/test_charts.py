import base64
import pytest
from unittest.mock import patch


class TestSecurityCharts:
    @pytest.fixture
    def charts(self):
        from agent.charts import SecurityCharts
        return SecurityCharts()

    def test_severity_distribution_returns_base64(self, charts):
        issues = [{"severity": "critical"}, {"severity": "high"}, {"severity": "medium"}]
        result = charts.severity_distribution(issues)
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

    def test_severity_distribution_counts_all_levels(self, charts):
        issues = [
            {"severity": "critical"},
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "low"},
        ]
        with patch.object(charts, "_fig_to_base64", return_value="fake_b64") as mock_b64:
            result = charts.severity_distribution(issues)
            assert result == "fake_b64"
            fig = mock_b64.call_args[0][0]
            ax = fig.axes[0]
            bars = ax.patches
            heights = [b.get_height() for b in bars]
            assert heights[0] == 2  # critical
            assert heights[1] == 1  # high
            assert heights[3] == 1  # low

    def test_top_vulnerable_apps_returns_base64(self, charts):
        apps = [
            {"name": "App A", "risk_score": 9.8},
            {"name": "App B", "risk_score": 7.2},
        ]
        result = charts.top_vulnerable_apps(apps)
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    def test_top_vulnerable_apps_sorted_descending(self, charts):
        apps = [
            {"name": "Low", "risk_score": 2.0},
            {"name": "High", "risk_score": 9.8},
            {"name": "Mid", "risk_score": 5.5},
        ]
        with patch.object(charts, "_fig_to_base64", return_value="fake_b64") as mock_b64:
            charts.top_vulnerable_apps(apps)
            fig = mock_b64.call_args[0][0]
            ax = fig.axes[0]
            labels = [t.get_text() for t in ax.get_yticklabels()]
            assert labels[0] == "High"
