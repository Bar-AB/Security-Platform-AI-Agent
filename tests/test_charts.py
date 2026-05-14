import os
import pytest
from unittest.mock import patch, MagicMock


class TestSecurityCharts:
    @pytest.fixture
    def charts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from agent.charts import SecurityCharts
        return SecurityCharts()

    def test_severity_distribution_creates_file(self, charts, tmp_path):
        issues = [
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "high"},
            {"severity": "medium"},
        ]
        path = charts.severity_distribution(issues)
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_severity_distribution_counts_all_levels(self, charts):
        issues = [
            {"severity": "critical"},
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "low"},
        ]
        with patch.object(charts, "_save_fig", return_value="severity_distribution.png") as mock_save:
            charts.severity_distribution(issues)
            fig = mock_save.call_args[0][0]
            ax = fig.axes[0]
            bars = ax.patches
            heights = [b.get_height() for b in bars]
            assert heights[0] == 2  # critical
            assert heights[1] == 1  # high
            assert heights[3] == 1  # low

    def test_top_vulnerable_apps_creates_file(self, charts, tmp_path):
        apps = [
            {"name": "App A", "risk_score": 9.8},
            {"name": "App B", "risk_score": 7.2},
            {"name": "App C", "risk_score": 6.3},
        ]
        path = charts.top_vulnerable_apps(apps)
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_top_vulnerable_apps_sorted_descending(self, charts):
        apps = [
            {"name": "Low", "risk_score": 2.0},
            {"name": "High", "risk_score": 9.8},
            {"name": "Mid", "risk_score": 5.5},
        ]
        with patch.object(charts, "_save_fig", return_value="top_vulnerable_apps.png") as mock_save:
            charts.top_vulnerable_apps(apps)
            fig = mock_save.call_args[0][0]
            ax = fig.axes[0]
            labels = [t.get_text() for t in ax.get_yticklabels()]
            assert labels[0] == "High"
