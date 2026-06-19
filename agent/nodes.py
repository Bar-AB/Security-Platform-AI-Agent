import json
import logging
from datetime import date
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.charts import SecurityCharts
from agent.prompts import CLASSIFIER_PROMPT, FORMATTER_PROMPT, VALIDATOR_PROMPT
from agent.state import AgentState, GroundednessResult, QueryClassification
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
        messages = state["messages"]
        query = cast(str, messages[-1].content)
        history = self._format_history(messages)
        wants_chart = any(kw in query.lower() for kw in self._CHART_KEYWORDS)
        try:
            classifier = self._llm.with_structured_output(QueryClassification)
            prompt_value = CLASSIFIER_PROMPT.invoke({"history": history, "query": query})
            result = cast(QueryClassification, classifier.invoke(prompt_value))
            query_type = result.query_type
            if query_type == "chart" and any(
                kw in query.lower() for kw in self._DATA_ENTITY_KEYWORDS
            ):
                query_type = "data"
                logger.info("Overriding 'chart' → 'data': query contains data entities")
            standalone = result.standalone_query or query
            logger.info(
                "Classified '%s' as '%s' (standalone: '%s')",
                query[:50],
                query_type,
                standalone[:60],
            )
            return {
                "query_type": query_type,
                "docs_query": result.docs_query,
                "standalone_query": standalone,
                "wants_chart": wants_chart,
                **self._fresh_results(query_type),
            }
        except Exception:
            logger.exception("Classification failed, defaulting to 'mixed'")
            return {
                "query_type": "mixed",
                "docs_query": query,
                "standalone_query": query,
                "wants_chart": wants_chart,
                **self._fresh_results("mixed"),
            }

    @staticmethod
    def _fresh_results(query_type: str) -> dict:
        # Wipe per-turn outputs so a node that doesn't run this turn can't leak last turn's
        # result (state is persisted per thread). The "chart" route is exempt: it reuses the
        # previous turn's mcp_result ("now chart that").
        if query_type == "chart":
            return {"rag_result": "N/A", "rag_distances": [], "rag_chunks_returned": 0}
        return {"mcp_result": "N/A", "rag_result": "N/A", "rag_distances": [], "rag_chunks_returned": 0}

    @staticmethod
    def _format_history(messages: list[BaseMessage], max_messages: int = 6) -> str:
        # Recent turns feed the classifier so it can resolve follow-up references; truncated to
        # keep the prompt small.
        prior = messages[:-1][-max_messages:]
        lines: list[str] = []
        for msg in prior:
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"{role}: {content[:500]}")
        return "\n".join(lines) if lines else "(no prior conversation)"

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
            "- For 'all issues' or unspecific queries: call BOTH get_security_issues AND get_pipeline_issues. "
            "If the user says 'security issues', call ONLY get_security_issues.\n\n"
            "Never guess or fabricate data. If filters return no results, say so clearly."
        )

    def mcp_node(self, state: AgentState) -> dict:
        query = state.get("standalone_query") or cast(
            str, state["messages"][-1].content
        )
        tools = self._mcp_tools.as_langchain_tools()
        try:
            llm_with_tools = self._llm.bind_tools(tools)
            response = llm_with_tools.invoke(
                [self._build_mcp_system(), HumanMessage(query)]
            )
            if response.tool_calls:
                tool_results = self._execute_tool_calls(response.tool_calls, tools)
                if state.get("wants_chart"):
                    self._try_generate_chart(tool_results)
                return {"mcp_result": tool_results}
            return {"mcp_result": response.content}
        except Exception:
            logger.exception("MCP node failed for query: %s", query[:50])
            return {
                "mcp_result": "Error fetching security data. The platform may be unavailable."
            }

    def rag_node(self, state: AgentState) -> dict:
        query = cast(str, state["messages"][-1].content)
        retrieval_query = state.get("docs_query") or query
        try:
            docs = self._retriever.retrieve(retrieval_query)
            if not docs:
                return {
                    "rag_result": "No relevant documentation found.",
                    "rag_distances": [],
                    "rag_chunks_returned": 0,
                }
            distances = [round(doc.metadata.get("distance", 0.0), 3) for doc in docs]
            return {
                "rag_result": self._retriever.format_for_prompt(docs),
                "rag_distances": distances,
                "rag_chunks_returned": len(docs),
            }
        except Exception:
            logger.exception("RAG node failed for query: %s", query[:50])
            return {
                "rag_result": "Error retrieving documentation.",
                "rag_distances": [],
                "rag_chunks_returned": 0,
            }

    _GROUNDEDNESS_THRESHOLD: float = 0.7

    _AGGREGATION_KEYWORDS = frozenset(
        ["how many", "how much", "in total", "total", "count", "how much there"]
    )
    _CHART_KEYWORDS = frozenset(
        ["chart", "graph", "visualize", "visualization", "plot"]
    )
    # If the LLM misclassifies a data+chart query as "chart", these words signal fresh data is needed.
    _DATA_ENTITY_KEYWORDS = frozenset(
        [
            "issues",
            "issue",
            "applications",
            "apps",
            "pipeline",
            "findings",
            "vulnerabilities",
            "distribution",
            "breakdown",
            "critical",
            "high",
            "medium",
            "low",
            "open",
            "resolved",
            "payment-service",
            "auth-service",
            "user-service",
            "api-gateway",
            "frontend-app",
        ]
    )

    # rag_node returns these strings when there is nothing useful to pass to the formatter.
    _EMPTY_RESULTS = frozenset({
        "N/A",
        "No relevant documentation found.",
        "Error retrieving documentation.",
    })

    def format_response(self, state: AgentState) -> dict:
        query = state.get("standalone_query") or cast(
            str, state["messages"][-1].content
        )
        mcp_result = state.get("mcp_result") or "N/A"
        rag_result = state.get("rag_result") or "N/A"
        try:
            # Render pure data deterministically so no rows get dropped; aggregation queries
            # still go through the LLM to synthesize a count.
            is_aggregation = any(
                kw in query.lower() for kw in self._AGGREGATION_KEYWORDS
            )
            chart_note = (
                "Chart generated and opened in your default image viewer.\n\n"
                if state.get("wants_chart")
                else ""
            )
            # No context from either source — don't let the LLM answer from training data.
            if mcp_result in self._EMPTY_RESULTS and rag_result in self._EMPTY_RESULTS:
                return self._emit(
                    "I don't have information about that in the platform's documentation "
                    "or security data. Try asking about connectors, dashboards, security "
                    "issues, applications, or pipeline findings."
                )
            if rag_result == "N/A" and mcp_result != "N/A" and not is_aggregation:
                return self._emit(
                    chart_note + self._format_mcp_as_markdown(mcp_result)
                )
            response = self._formatter.invoke(
                {
                    "query": query,
                    "mcp_result": mcp_result,
                    "rag_result": rag_result,
                }
            )
            sources_footer = self._build_sources_footer(rag_result)
            return self._emit(
                chart_note + cast(str, response.content) + sources_footer
            )
        except Exception:
            logger.exception("Formatter failed")
            return self._emit("Sorry, I could not generate a response.")

    @staticmethod
    def _emit(text: str) -> dict:
        # Append the answer as an AIMessage so the next turn's classifier sees it for follow-ups.
        return {"final_response": text, "messages": [AIMessage(content=text)]}

    def chart_node(self, state: AgentState) -> dict:
        mcp_result = state.get("mcp_result") or ""
        if not mcp_result or mcp_result == "N/A":
            return self._emit(
                "No data available to chart. Please run a data query first."
            )
        self._try_generate_chart(mcp_result)
        formatted = self._format_mcp_as_markdown(mcp_result)
        return self._emit(f"Chart generated from previous results.\n\n{formatted}")

    @staticmethod
    def _build_sources_footer(rag_result: str) -> str:
        # The LLM formatter doesn't reliably echo the retriever's "[source — section]" prefixes,
        # so extract them and append a deterministic Sources line.
        if rag_result == "N/A":
            return ""
        sources: list[str] = []
        for part in rag_result.split("\n\n---\n\n"):
            first_line = part.strip().split("\n", 1)[0].strip()
            if not (first_line.startswith("[") and first_line.endswith("]")):
                continue
            if first_line not in sources:
                sources.append(first_line)
        if not sources:
            return ""
        return "\n\n**Sources:** " + ", ".join(sources)

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
            lines = [
                f"**{tool_name}** — {len(items)} result{'s' if len(items) != 1 else ''}:\n"
            ]
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

    def validate_response(self, state: AgentState) -> dict:
        response = state.get("final_response", "")
        mcp_result = state.get("mcp_result") or "N/A"
        rag_result = state.get("rag_result") or "N/A"

        # Nothing to validate against — skip
        if mcp_result == "N/A" and rag_result == "N/A":
            return {"validation_score": 1.0, "validation_flagged": False}

        # Guard: if format_response emitted nothing, there's nothing to validate
        if not response:
            return {"validation_score": 1.0, "validation_flagged": False}

        context = f"MCP Data:\n{mcp_result}\n\nDocumentation:\n{rag_result}"
        try:
            validator = self._llm.with_structured_output(GroundednessResult)
            result = cast(
                GroundednessResult,
                validator.invoke(VALIDATOR_PROMPT.invoke({"context": context, "response": response})),
            )
            logger.info(
                "Groundedness score=%.2f is_grounded=%s flagged=%d claim(s) — %s",
                result.score,
                result.is_grounded,
                len(result.flagged_claims),
                result.reasoning,
            )
            # Guard on score directly — don't trust is_grounded alone; an inconsistent
            # LLM response (e.g. score=0.2, is_grounded=True) should still trigger the warning.
            is_flagged = not result.is_grounded or result.score < self._GROUNDEDNESS_THRESHOLD
            if is_flagged:
                flagged_lines = "\n".join(f"  • {c}" for c in result.flagged_claims)
                warning = (
                    f"\n\n---\n> ⚠️ **Validation warning** (confidence: {result.score:.0%}): "
                    f"some claims could not be verified against the retrieved data."
                )
                if result.flagged_claims:
                    warning += f"\n> Unverified:\n{flagged_lines}"
                warned_text = response + warning
                # Replace the AIMessage committed by format_response so conversation history
                # stays consistent with what the user sees. LangGraph's add_messages reducer
                # replaces an existing message when the returned message shares its ID.
                last_msg = state["messages"][-1]
                updated_msg = AIMessage(content=warned_text, id=last_msg.id)
                return {
                    "messages": [updated_msg],
                    "final_response": warned_text,
                    "validation_score": result.score,
                    "validation_flagged": True,
                }
            return {"validation_score": result.score, "validation_flagged": False}
        except Exception:
            logger.exception("Response validation failed — skipping")
            return {"validation_score": 1.0, "validation_flagged": False}

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
