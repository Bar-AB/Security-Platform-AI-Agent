from enum import Enum
from pydantic import BaseModel


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(str, Enum):
    INJECTION = "injection"
    XSS = "xss"
    BROKEN_AUTH = "broken_auth"
    EXPOSED_DATA = "exposed_data"
    MISCONFIG = "misconfig"
    DEPENDENCY = "dependency"


class IssueStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


class SecurityIssue(BaseModel):
    id: str
    title: str
    severity: Severity
    category: Category
    status: IssueStatus
    cve_id: str | None = None
    application: str
    description: str
    discovered_at: str


class Application(BaseModel):
    id: str
    name: str
    risk_score: float
    issue_count: int
    critical_count: int
    last_scan: str


class PipelineIssue(BaseModel):
    id: str
    pipeline: str
    stage: str
    severity: Severity
    title: str
    tool: str
    commit_sha: str
    branch: str
    detected_at: str
