from mock_server.models import Severity


class TestMockServerData:
    def test_issues_count(self) -> None:
        from mock_server.data import MOCK_ISSUES

        assert len(MOCK_ISSUES) >= 8

    def test_applications_count(self) -> None:
        from mock_server.data import MOCK_APPLICATIONS

        assert len(MOCK_APPLICATIONS) >= 4

    def test_pipeline_issues_count(self) -> None:
        from mock_server.data import MOCK_PIPELINE_ISSUES

        assert len(MOCK_PIPELINE_ISSUES) >= 4

    def test_issue_fields_populated(self) -> None:
        from mock_server.data import MOCK_ISSUES

        issue = MOCK_ISSUES[0]
        assert issue.id
        assert issue.title
        assert issue.application
        assert issue.description

    def test_risk_score_range(self) -> None:
        from mock_server.data import MOCK_APPLICATIONS

        for app in MOCK_APPLICATIONS:
            assert 0.0 <= app.risk_score <= 10.0

    def test_has_critical_issues(self) -> None:
        from mock_server.data import MOCK_ISSUES

        criticals = [i for i in MOCK_ISSUES if i.severity == Severity.CRITICAL]
        assert len(criticals) >= 2

    def test_security_issue_serializable(self) -> None:
        from mock_server.data import MOCK_ISSUES

        data = MOCK_ISSUES[0].model_dump()
        assert isinstance(data, dict)
        assert "severity" in data

    def test_filter_by_severity(self) -> None:
        from mock_server.data import MOCK_ISSUES
        from mock_server.models import Severity

        result = [i for i in MOCK_ISSUES if i.severity == Severity.CRITICAL]
        assert all(i.severity == Severity.CRITICAL for i in result)
        assert len(result) >= 2

    def test_filter_by_status(self) -> None:
        from mock_server.data import MOCK_ISSUES
        from mock_server.models import IssueStatus

        result = [i for i in MOCK_ISSUES if i.status == IssueStatus.OPEN]
        assert all(i.status == IssueStatus.OPEN for i in result)

    def test_filter_pipeline_by_branch(self) -> None:
        from mock_server.data import MOCK_PIPELINE_ISSUES

        result = [i for i in MOCK_PIPELINE_ISSUES if i.branch == "main"]
        assert all(i.branch == "main" for i in result)
        assert len(result) >= 2
