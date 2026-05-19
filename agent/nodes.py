import json
import logging
from datetime import date
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from agent.charts import SecurityCharts
from agent.prompts import CLASSIFIER_PROMPT, FORMATTER_PROMPT
from agent.state import AgentState, QueryClassification
from mcp_client.tools import SecurityMCPTools
from rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


class AgentNodes:
    def __init__(
        self,
        llm: BaseChatModel,
        mcp_tools: SecurityMCPTools,
        retriever: RAGRetriever,
    ) -> None:
        self._llm = llm
        self._mcp_tools = mcp_tools
        self._retriever = retriever
        self._formatter = FORMATTER_PROMPT | llm
        self._charts = SecurityCharts()

    def classify_query(self, state: AgentState) -> dict:
        query = cast(str, state["messages"][-1].content)
        try:
            classifier = self._llm.with_structured_output(QueryClassification)
            prompt_value = CLASSIFIER_PROMPT.invoke({"query": query})
            result = cast(QueryClassification, classifier.invoke(prompt_value))
            logger.info("Classified '%s' as '%s'", query[:50], result.query_type)
            return {"query_type": result.query_type, "docs_query": result.docs_query}
        except Exception:
            logger.exception("Classification failed, defaulting to 'mixed'")
            return {"query_type": "mixed", "docs_query": query}

    @staticmethod
    def _build_mcp_system() -> SystemMessage:
        today = date.today().isoformat()
        return SystemMessage(
            f"You are a security analyst. Today's date is {today}. "
            "You have three tools: get_security_issues, get_applications, get_pipeline_issues.\n\n"
            "TOOL ROUTING — follow these rules exactly:\n\n"
            "FILTERING:\n"
            "- By service/app ('issues in payment-service', 'what affects auth?'): "
            "call BOTH get_security_issues(application='<name>') AND "
            "get_pipeline_issues(pipeline='<name>-ci'). "
            "Known apps: payment-service, auth-service, user-service, api-gateway, frontend-app. "
            "Their pipelines follow the pattern '<app>-ci' (e.g. auth-service → auth-service-ci).\n"
            "- By CVE ('show CVE-2021-44228'): get_security_issues(cve_id='CVE-...').\n"
            "- By technology keyword ('AWS incidents', 'log4j findings', 'JWT issues'): "
            "use keyword=<term> on whichever tool is relevant.\n"
            "- By pipeline name ('auth-service-ci findings'): get_pipeline_issues(pipeline='auth-service-ci').\n"
            "- By pipeline stage ('SAST findings', 'dependency scan results', 'secret scan'): "
            "get_pipeline_issues(stage='sast'|'dependency-scan'|'secret-scan'|'container-scan'|'dast').\n"
            "- By scanner tool ('Trivy findings', 'what did Semgrep find?', 'Gitleaks results'): "
            "get_pipeline_issues(tool='Trivy'|'Semgrep'|'Gitleaks'|'OWASP ZAP').\n"
            "- By branch ('issues on main', 'feature branch findings'): "
            "get_pipeline_issues(branch='main'|'develop'|'feature'). Prefix match is supported.\n\n"
            "DATE QUERIES (use today's date to calculate):\n"
            "- 'last month': discovered_after=first day of last month, discovered_before=last day of last month.\n"
            "- 'last week': discovered_after=7 days ago.\n"
            "- 'in November 2024': discovered_after='2024-11-01', discovered_before='2024-11-30'.\n"
            "- 'Q4 2024': discovered_after='2024-10-01', discovered_before='2024-12-31'.\n"
            "Use discovered_after/discovered_before for get_security_issues, "
            "detected_after/detected_before for get_pipeline_issues. "
            "If the result is empty, say so — never fabricate data.\n\n"
            "RANKING:\n"
            "- 'top N riskiest/most vulnerable apps': get_applications(limit=N). "
            "Results are pre-sorted by risk score descending.\n"
            "- 'top N highest risk/most severe issues' or 'top N incidents': "
            "get_security_issues(limit=N). Results are pre-sorted by severity (critical first). "
            "IMPORTANT: security issues do NOT have a risk_score field — use severity as the risk proxy.\n"
            "- 'top N pipeline findings': get_pipeline_issues(limit=N).\n\n"
            "MULTIPLE SEVERITIES:\n"
            "- 'high and critical issues': make TWO tool calls — one with severity='critical', "
            "one with severity='high' — then combine results in your answer.\n\n"
            "COUNT/TOTAL QUERIES:\n"
            "- When the user asks 'how many', 'how much', 'total', or 'count', always state the "
            "exact number clearly in your answer (e.g. 'There are 3 issues in auth-service: 1 security issue and 2 pipeline findings.').\n\n"
            "GENERAL FINDINGS:\n"
            "- For 'all issues', 'critical issues', or any broad security query: "
            "call BOTH get_security_issues AND get_pipeline_issues.\n\n"
            "Never guess or fabricate data. If filters return no results, say so clearly."
        )

    def mcp_node(self, state: AgentState) -> dict:
        query = cast(str, state["messages"][-1].content)
        tools = self._mcp_tools.as_langchain_tools()
        try:
            llm_with_tools = self._llm.bind_tools(tools)
            response = llm_with_tools.invoke([self._build_mcp_system(), *state["messages"]])
            if response.tool_calls:
                tool_results = self._execute_tool_calls(response.tool_calls, tools)
                self._try_generate_chart(tool_results)
                return {"mcp_result": tool_results}
            return {"mcp_result": response.content}
        except Exception:
            logger.exception("MCP node failed for query: %s", query[:50])
            return {"mcp_result": "Error fetching security data. The platform may be unavailable."}

    def rag_node(self, state: AgentState) -> dict:
        query = cast(str, state["messages"][-1].content)
        retrieval_query = state.get("docs_query") or query
        try:
            docs = self._retriever.retrieve(retrieval_query)
            if not docs:
                return {"rag_result": "No relevant documentation found."}
            return {"rag_result": self._retriever.format_for_prompt(docs)}
        except Exception:
            logger.exception("RAG node failed for query: %s", query[:50])
            return {"rag_result": "Error retrieving documentation."}

    _AGGREGATION_KEYWORDS = frozenset(["how many", "how much", "in total", "total", "count", "how much there"])

    def format_response(self, state: AgentState) -> dict:
        query = cast(str, state["messages"][-1].content)
        mcp_result = state.get("mcp_result") or "N/A"
        rag_result = state.get("rag_result") or "N/A"
        try:
            # Pure data queries: format deterministically so no items get dropped by LLM summarization.
            # Exception: aggregation queries need the LLM to synthesize a count answer.
            is_aggregation = any(kw in query.lower() for kw in self._AGGREGATION_KEYWORDS)
            if rag_result == "N/A" and mcp_result != "N/A" and not is_aggregation:
                return {"final_response": self._format_mcp_as_markdown(mcp_result)}
            response = self._formatter.invoke({
                "query": query,
                "mcp_result": mcp_result,
                "rag_result": rag_result,
            })
            return {"final_response": response.content}
        except Exception:
            logger.exception("Formatter failed")
            return {"final_response": "Sorry, I could not generate a response."}

    def _format_mcp_as_markdown(self, mcp_result: str) -> str:
        sections: list[str] = []
        for chunk in mcp_result.split("\n\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            header, _, body = chunk.partition("\n")
            tool_name = header.strip("[]").replace("_", " ").title()
            try:
                items: list[dict] = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                sections.append(chunk)
                continue
            if not items:
                sections.append(f"**{tool_name}:** No results found.")
                continue
            lines = [f"**{tool_name}** — {len(items)} result{'s' if len(items) != 1 else ''}:\n"]
            for idx, item in enumerate(items, 1):
                fields = "  \n".join(
                    f"  - **{k.replace('_', ' ').title()}**: {v}"
                    for k, v in item.items()
                    if v is not None
                )
                lines.append(f"**{idx}.** {fields}\n")
            sections.append("\n".join(lines))
        return "\n\n---\n\n".join(sections)

    def _try_generate_chart(self, mcp_result: str) -> None:
        severity_items: list[dict] = []
        app_items: list[dict] = []
        for chunk in mcp_result.split("\n\n"):
            chunk = chunk.strip()
            if not chunk:
                continue
            _, _, body = chunk.partition("\n")
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, list) or not data:
                continue
            if "severity" in data[0]:
                severity_items.extend(data)
            elif "risk_score" in data[0]:
                app_items.extend(data)
        if severity_items:
            path = self._charts.severity_distribution(severity_items)
            logger.info("Chart saved to %s", path)
        if app_items:
            path = self._charts.top_vulnerable_apps(app_items)
            logger.info("Chart saved to %s", path)

    def _execute_tool_calls(self, tool_calls: list, tools: list) -> str:
        tool_map = {t.name: t for t in tools}
        results: list[str] = []
        for call in tool_calls:
            tool = tool_map.get(call["name"])
            if not tool:
                continue
            try:
                output = tool.invoke(call["args"])
                results.append(f"[{call['name']}]\n{output}")
            except Exception:
                logger.exception("Tool call failed: %s", call["name"])
                results.append(f"[{call['name']}] Error: tool call failed.")
        return "\n\n".join(results)
