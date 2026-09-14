import asyncio
import json
import logging
from collections import defaultdict
from datetime import date
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.charts import SecurityCharts
from agent.guardrails import InputGuardrail
from agent.prompts import CLASSIFIER_PROMPT, FORMATTER_PROMPT, SYNTHESIS_PROMPT, VALIDATOR_PROMPT
from agent.state import AgentState, GroundednessResult, QueryClassification
from mcp_client.tools import SecurityMCPTools
from rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


class AgentNodes:
    _GROUNDEDNESS_THRESHOLD: float = 0.7
    _MAX_ACTIVE_ENTITIES = 10
    _MAX_INJECTED_ENTITIES = 2
    _CLASSIFIER_RECENT_TURNS = 4
    _CLASSIFIER_RECENT_CHARS = 1500
    _CLASSIFIER_OLDER_CHARS = 200
    _SYNTHESIS_HISTORY_CHARS = 3000

    _AGGREGATION_KEYWORDS = frozenset(
        [
            "how many",
            "how much",
            "in total",
            "total",
            "count",
            "how much there",
            "zero vulnerabilities",
            "zero issues",
            "zero open issues",
            "no vulnerabilities",
            "no issues",
            "no open issues",
            "which have zero",
            "have no issues",
            "have zero",
        ]
    )
    _GROUP_BY_FIELDS: dict[str, str] = {
        "by severity": "severity",
        "per severity": "severity",
        "severity breakdown": "severity",
        "severity distribution": "severity",
        "count by severity": "severity",
        "by application": "application",
        "per application": "application",
        "by service": "application",
        "per service": "application",
        "application breakdown": "application",
        "by category": "category",
        "per category": "category",
        "category breakdown": "category",
        "by status": "status",
        "per status": "status",
        "status breakdown": "status",
    }
    _FIELD_SPECIFIC_KEYWORDS = frozenset(
        [
            "cvss",
            "cvss score",
            "risk score",
            "exploit",
            "epss",
            "patch date",
            "remediation date",
            "fix date",
            "cvss vector",
            "base score",
        ]
    )

    _COMPARISON_KEYWORDS = frozenset(
        [
            "compare",
            "vs",
            "versus",
            "both",
            "each",
            "side by side",
            "contrast",
            "how do they",
            "how do both",
        ]
    )

    _CHART_KEYWORDS = frozenset(["chart", "graph", "visualize", "visualization", "plot"])
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

    _SEVERITY_ORDER: list[str] = ["critical", "high", "medium", "low"]

    _EMPTY_RESULTS = frozenset(
        {
            "N/A",
            "No relevant documentation found.",
            "Error retrieving documentation.",
        }
    )

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
        self._guardrail = InputGuardrail()

    def classify_query(self, state: AgentState) -> dict:
        messages = state["messages"]
        query = cast(str, messages[-1].content)

        guard = self._guardrail.check(query)
        if guard.blocked:
            safe_msg = guard.safe_message
            return {
                **self._reset_turn_results("mixed"),
                "query_type": "blocked",
                "docs_query": "",
                "standalone_query": "",
                "wants_chart": False,
                "group_by_field": None,
                "active_entities": state.get("active_entities") or [],
                "final_response": safe_msg,
                "messages": [AIMessage(content=safe_msg)],
            }

        history = self._format_history(
            messages,
            recent_n=self._CLASSIFIER_RECENT_TURNS,
            recent_max_content=self._CLASSIFIER_RECENT_CHARS,
            max_content=self._CLASSIFIER_OLDER_CHARS,
        )
        wants_chart = any(kw in query.lower() for kw in self._CHART_KEYWORDS)
        current_entities: list[str] = state.get("active_entities") or []
        raw_entities_str = ", ".join(current_entities) if current_entities else "(none)"
        entities_str = InputGuardrail.sanitize_for_xml_context(raw_entities_str, "active_entities")
        try:
            classifier = self._llm.with_structured_output(QueryClassification)
            prompt_value = CLASSIFIER_PROMPT.invoke(
                {"history": history, "query": query, "active_entities": entities_str}
            )
            result = cast(QueryClassification, classifier.invoke(prompt_value))
            query_type = result.query_type
            if query_type == "chart" and any(
                kw in query.lower() for kw in self._DATA_ENTITY_KEYWORDS
            ):
                query_type = "data"
                logger.info("Overriding 'chart' → 'data': query contains data entities")
            standalone = result.standalone_query or query
            standalone = self._enforce_multi_entity(
                standalone, query, result.active_entities, current_entities
            )
            deduped_entities = list(dict.fromkeys(result.active_entities + current_entities))
            merged_entities = deduped_entities[: self._MAX_ACTIVE_ENTITIES]
            logger.info(
                "Classified '%s' as '%s' (standalone: '%s', entities: %s)",
                query[:50],
                query_type,
                standalone[:60],
                merged_entities,
            )
            return {
                **self._reset_turn_results(query_type),
                "query_type": query_type,
                "docs_query": result.docs_query,
                "standalone_query": standalone,
                "wants_chart": wants_chart,
                "group_by_field": self._detect_group_by(standalone),
                "active_entities": merged_entities,
            }
        except Exception:
            logger.exception("Classification failed, defaulting to 'mixed'")
            return {
                **self._reset_turn_results("mixed"),
                "query_type": "mixed",
                "docs_query": query,
                "standalone_query": query,
                "wants_chart": wants_chart,
                "group_by_field": self._detect_group_by(query),
                "active_entities": current_entities,
            }

    @staticmethod
    def _reset_turn_results(query_type: str) -> dict:
        if query_type == "chart":
            return {
                "rag_result": "N/A",
                "rag_distances": [],
                "rag_chunks_returned": 0,
                "chart_image": None,
                "group_by_field": None,
            }
        return {
            "mcp_result": "N/A",
            "rag_result": "N/A",
            "rag_distances": [],
            "rag_chunks_returned": 0,
            "chart_image": None,
            "group_by_field": None,
        }

    @staticmethod
    def _format_history(
        messages: list[BaseMessage],
        max_messages: int = 10,
        max_content: int = 500,
        recent_n: int = 0,
        recent_max_content: int = 500,
    ) -> str:
        prior = messages[:-1][-max_messages:]
        recent_cutoff = len(prior) - recent_n if recent_n > 0 else len(prior)
        lines: list[str] = []
        for i, msg in enumerate(prior):
            if isinstance(msg, HumanMessage):
                role = "User"
            elif isinstance(msg, AIMessage):
                role = "Assistant"
            else:
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            limit = recent_max_content if i >= recent_cutoff else max_content
            lines.append(f"{role}: {content[:limit]}")
        return "\n".join(lines) if lines else "(no prior conversation)"

    @staticmethod
    def _enforce_multi_entity(
        standalone: str,
        raw_query: str,
        new_entities: list[str],
        current_entities: list[str],
    ) -> str:
        if not any(kw in raw_query.lower() for kw in AgentNodes._COMPARISON_KEYWORDS):
            return standalone
        missing = [e for e in current_entities if e not in new_entities and e not in standalone]
        if not missing:
            return standalone
        to_inject = missing[: AgentNodes._MAX_INJECTED_ENTITIES]
        logger.info("Multi-entity guard: injecting %s into standalone_query", to_inject)
        return standalone + " and " + " and ".join(to_inject)

    @staticmethod
    def _detect_group_by(query: str) -> str | None:
        q = query.lower()
        for phrase, field in AgentNodes._GROUP_BY_FIELDS.items():
            if phrase in q:
                return field
        return None

    def synthesis_node(self, state: AgentState) -> dict:
        query = state.get("standalone_query") or cast(str, state["messages"][-1].content)
        raw_history = self._format_history(
            state["messages"], max_content=self._SYNTHESIS_HISTORY_CHARS
        )
        history = InputGuardrail.sanitize_for_xml_context(raw_history, "conversation_history")
        try:
            chain = SYNTHESIS_PROMPT | self._llm
            response = chain.invoke({"history": history, "query": query})
            return {
                **self._emit(cast(str, response.content)),
                "mcp_result": "N/A",
                "rag_result": "N/A",
            }
        except Exception:
            logger.exception("Synthesis node failed for query: %s", query[:50])
            return {
                **self._emit(
                    "Sorry, I could not generate a response from the conversation history."
                ),
                "mcp_result": "N/A",
                "rag_result": "N/A",
            }

    @staticmethod
    def _build_mcp_system() -> SystemMessage:
        today = date.today().isoformat()
        return SystemMessage(
            f"You are a security analyst. Today's date is {today}. "
            "You have three tools: get_security_issues, get_applications, get_pipeline_issues.\n\n"
            "TOOL ROUTING — follow these rules exactly:\n\n"
            "FILTERING:\n"
            "- By exact ID ('ISS-003', 'PIPE-006', 'show me ISS-001'): "
            "use get_security_issues(id='ISS-003') for ISS-* IDs, "
            "get_pipeline_issues(id='PIPE-006') for PIPE-* IDs. "
            "Always prefer id lookup when the user provides an exact ID.\n"
            "- By service/app ('issues in payment-service', 'what affects auth?'): "
            "call BOTH get_security_issues(application='<name>') AND "
            "get_pipeline_issues(pipeline='<name>-ci'). "
            "Known apps: payment-service, auth-service, user-service, api-gateway, frontend-app. "
            "Their pipelines follow the pattern '<app>-ci' (e.g. auth-service → auth-service-ci).\n"
            "- By CVE ('show CVE-2021-44228'): get_security_issues(cve_id='CVE-...').\n"
            "- By technology keyword ('AWS incidents', 'log4j findings', 'JWT issues'): "
            "use keyword=<term> on whichever tool is relevant.\n"
            "- By pipeline name ('auth-service-ci findings'): "
            "get_pipeline_issues(pipeline='auth-service-ci').\n"
            "- By pipeline stage ('SAST findings', 'dependency scan results', 'secret scan'): "
            "get_pipeline_issues(stage='sast'|'dependency-scan'|'secret-scan'|'container-scan'|'dast').\n"
            "- By scanner tool ('Trivy findings', 'what did Semgrep find?', 'Gitleaks results'): "
            "get_pipeline_issues(tool='Trivy'|'Semgrep'|'Gitleaks'|'OWASP ZAP').\n"
            "- By branch ('issues on main', 'feature branch findings'): "
            "get_pipeline_issues(branch='main'|'develop'|'feature'). Prefix match is supported.\n\n"
            "DATE QUERIES (use today's date to calculate):\n"
            "- 'last month': discovered_after=first day of last month, discovered_before=last day "
            "of last month.\n"
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
            "IMPORTANT: security issues do NOT have a risk_score field — use severity as the risk "
            "proxy.\n"
            "- 'top N pipeline findings': get_pipeline_issues(limit=N).\n\n"
            "MULTIPLE SEVERITIES:\n"
            "- 'high and critical issues': make TWO tool calls — one with severity='critical', "
            "one with severity='high' — then combine results in your answer.\n\n"
            "COUNT/TOTAL QUERIES:\n"
            "- When the user asks 'how many', 'how much', 'total', or 'count', always state the "
            "exact number clearly in your answer (e.g. 'There are 3 issues in auth-service: 1 "
            "security issue and 2 pipeline findings.').\n"
            "- Always include ALL statuses (open, in_progress, resolved) in counts unless the user "
            "explicitly asks for open or active issues only.\n"
            "- Never add a limit parameter to any tool call unless the user explicitly specified a "
            "maximum number of results (e.g. 'show me the top 5').\n\n"
            "GENERAL FINDINGS:\n"
            "- For 'all issues' or unspecific queries: call BOTH get_security_issues AND "
            "get_pipeline_issues. "
            "If the user says 'security issues', call ONLY get_security_issues.\n\n"
            "Never guess or fabricate data. If filters return no results, say so clearly.\n\n"
            "COMPARING MULTIPLE SERVICES: When the query asks to compare, contrast, or show "
            "side-by-side data for two or more named services, make SEPARATE tool calls for EACH "
            "service. For example, to compare auth-service and payment-service: call "
            "get_security_issues(application='auth-service'), "
            "get_security_issues(application='payment-service'), "
            "get_pipeline_issues(pipeline='auth-service-ci'), and "
            "get_pipeline_issues(pipeline='payment-service-ci') as four separate calls.\n\n"
            "ZERO-VULNERABILITY QUERIES: If the user asks which apps have zero issues or "
            "vulnerabilities, call get_applications() with no filter to get all apps, then "
            "identify those with issue_count == 0 in your response. Do not say you cannot "
            "filter by zero — fetch all and filter yourself."
        )

    async def mcp_node(self, state: AgentState) -> dict:
        query = state.get("standalone_query") or cast(str, state["messages"][-1].content)
        tools = self._mcp_tools.as_langchain_tools()

        if state.get("group_by_field"):
            try:
                iss = await self._mcp_tools.get_security_issues()
                pipe = await self._mcp_tools.get_pipeline_issues()
                tool_results = f"[get_security_issues]\n{iss}\n\n[get_pipeline_issues]\n{pipe}"
                return {"mcp_result": tool_results, "chart_image": None}
            except Exception:
                logger.exception("Direct fetch failed for group-by query — falling back to LLM")

        try:
            llm_with_tools = self._llm.bind_tools(tools)
            response = await llm_with_tools.ainvoke([self._build_mcp_system(), HumanMessage(query)])
            if response.tool_calls:
                tool_results = await self._execute_tool_calls_async(response.tool_calls)
                chart_b64 = (
                    self._try_generate_chart(tool_results) if state.get("wants_chart") else None
                )
                return {"mcp_result": tool_results, "chart_image": chart_b64}
            return {"mcp_result": response.content, "chart_image": None}
        except Exception:
            logger.exception("MCP node failed for query: %s", query[:50])
            return {
                "mcp_result": "Error fetching security data. The platform may be unavailable.",
                "chart_image": None,
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

    @staticmethod
    def _count_by_field(mcp_result: str, field: str) -> str | None:
        counts: dict[str, list[str]] = defaultdict(list)
        try:
            for chunk in mcp_result.split("\n\n"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                _, _, body = chunk.partition("\n")
                if not body:
                    continue
                items = json.loads(body)
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    value = item.get(field)
                    if value is None:
                        continue
                    id_ = item.get("id", "")
                    title = item.get("title", id_)
                    counts[str(value)].append(f"{title} ({id_})" if id_ else title)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not counts:
            return None
        keys = (
            [k for k in AgentNodes._SEVERITY_ORDER if k in counts]
            + sorted(k for k in counts if k not in AgentNodes._SEVERITY_ORDER)
            if field == "severity"
            else sorted(counts)
        )
        lines: list[str] = [
            f"Based on the provided data, here are the counts of issues by {field}:\n"
        ]
        summary: list[str] = []
        for key in keys:
            items_for_key = counts[key]
            lines.append(f"**{key.replace('_', ' ').title()}:**")
            for i, title in enumerate(items_for_key, 1):
                lines.append(f"{i}. {title}")
            lines.append("")
            summary.append(f"{len(items_for_key)} {key}")
        total = sum(len(v) for v in counts.values())
        lines.append(f"In summary: there are {', '.join(summary)}, and **{total} issues total**.")
        return "\n".join(lines)

    def format_response(self, state: AgentState) -> dict:
        query = state.get("standalone_query") or cast(str, state["messages"][-1].content)
        mcp_result = state.get("mcp_result") or "N/A"
        rag_result = state.get("rag_result") or "N/A"
        try:
            is_aggregation = any(kw in query.lower() for kw in self._AGGREGATION_KEYWORDS)
            if mcp_result in self._EMPTY_RESULTS and rag_result in self._EMPTY_RESULTS:
                return self._emit(
                    "I don't have information about that in the platform's documentation "
                    "or security data. Try asking about connectors, dashboards, security "
                    "issues, applications, or pipeline findings."
                )
            group_by_field = state.get("group_by_field")
            if group_by_field and rag_result == "N/A" and mcp_result != "N/A":
                det = self._count_by_field(mcp_result, group_by_field)
                if det:
                    return self._emit(det)
            is_field_specific = any(kw in query.lower() for kw in self._FIELD_SPECIFIC_KEYWORDS)
            if (
                rag_result == "N/A"
                and mcp_result != "N/A"
                and not is_aggregation
                and not is_field_specific
            ):
                return self._emit(self._format_mcp_as_markdown(mcp_result))
            response = self._formatter.invoke(
                {
                    "query": query,
                    "mcp_result": InputGuardrail.sanitize_for_xml_context(mcp_result, "mcp_data"),
                    "rag_result": InputGuardrail.sanitize_for_xml_context(rag_result, "rag_data"),
                }
            )
            sources_footer = self._build_sources_footer(rag_result)
            return self._emit(cast(str, response.content) + sources_footer)
        except Exception:
            logger.exception("Formatter failed")
            return self._emit("Sorry, I could not generate a response.")

    @staticmethod
    def _emit(text: str) -> dict:
        return {"final_response": text, "messages": [AIMessage(content=text)]}

    def chart_node(self, state: AgentState) -> dict:
        mcp_result = state.get("mcp_result") or ""
        if not mcp_result or mcp_result == "N/A":
            return {
                **self._emit("No data available to chart. Please run a data query first."),
                "chart_image": None,
            }
        chart_b64 = self._try_generate_chart(mcp_result)
        formatted = self._format_mcp_as_markdown(mcp_result)
        msg = "Chart generated from previous results.\n\n" + formatted
        return {**self._emit(msg), "chart_image": chart_b64}

    @staticmethod
    def _build_sources_footer(rag_result: str) -> str:
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

    def _try_generate_chart(self, mcp_result: str) -> str | None:
        try:
            return self._generate_chart(mcp_result)
        except Exception:
            logger.exception("Chart generation failed — data will still be returned")
            return None

    def _generate_chart(self, mcp_result: str) -> str | None:
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
            b64 = self._charts.severity_distribution(severity_items)
            logger.info("Chart generated (%d bytes base64)", len(b64))
            return b64
        if app_items:
            b64 = self._charts.top_vulnerable_apps(app_items)
            logger.info("Chart generated (%d bytes base64)", len(b64))
            return b64
        return None

    def validate_response(self, state: AgentState) -> dict:
        response = state.get("final_response", "")
        mcp_result = state.get("mcp_result") or "N/A"
        rag_result = state.get("rag_result") or "N/A"
        query_type = state.get("query_type", "")
        is_synthesis = query_type == "synthesis" and mcp_result == "N/A" and rag_result == "N/A"

        if mcp_result == "N/A" and rag_result == "N/A" and not is_synthesis:
            return {"validation_score": 1.0, "validation_flagged": False}

        if not response:
            return {"validation_score": 1.0, "validation_flagged": False}

        if is_synthesis:
            raw_history = self._format_history(
                state["messages"][:-1], max_content=self._SYNTHESIS_HISTORY_CHARS
            )
            safe_history = InputGuardrail.sanitize_for_xml_context(raw_history, "context")
            context = f"Conversation History:\n{safe_history}"
        else:
            context = (
                f"MCP Data:\n{InputGuardrail.sanitize_for_xml_context(mcp_result, 'context')}\n\n"
                f"Documentation:\n{InputGuardrail.sanitize_for_xml_context(rag_result, 'context')}"
            )
        try:
            validator = self._llm.with_structured_output(GroundednessResult)
            result = cast(
                GroundednessResult,
                validator.invoke(
                    VALIDATOR_PROMPT.invoke({"context": context, "response": response})
                ),
            )
            logger.info(
                "Groundedness score=%.2f is_grounded=%s flagged=%d claim(s) — %s",
                result.score,
                result.is_grounded,
                len(result.flagged_claims),
                result.reasoning,
            )
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
            logger.exception("Response validation failed — confidence unknown")
            return {"validation_score": None, "validation_flagged": False}

    async def _execute_tool_calls_async(self, tool_calls: list) -> str:
        tool_map = {
            "get_security_issues": self._mcp_tools.get_security_issues,
            "get_applications": self._mcp_tools.get_applications,
            "get_pipeline_issues": self._mcp_tools.get_pipeline_issues,
        }

        async def _run_one(call: dict) -> str | None:
            fn = tool_map.get(call["name"])
            if not fn:
                return None
            try:
                output = await fn(**call["args"])
                return f"[{call['name']}]\n{output}"
            except Exception:
                logger.exception("Async tool call failed: %s", call["name"])
                return f"[{call['name']}] Error: tool call failed."

        results = await asyncio.gather(*(_run_one(call) for call in tool_calls))
        return "\n\n".join(r for r in results if r is not None)
