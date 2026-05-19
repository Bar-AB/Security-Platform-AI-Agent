from langchain_core.prompts import ChatPromptTemplate

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier for a security platform assistant.

Classify the user's query into exactly one type:
- "data": User wants live security data (issues, applications, pipeline findings, counts, severities).
- "doc": User wants to know HOW to use the platform (setup, connectors, dashboard, filters).
- "mixed": User wants BOTH data AND documentation (e.g. explain a vulnerability category AND show examples).

Also produce a docs_query: a refined search string for the documentation knowledge base.
- "data" queries: set docs_query to the original query (it will not be used).
- "doc" queries: rewrite as a concise keyword search focused on setup, configuration, or how-to aspects.
- "mixed" queries: extract only the conceptual/documentation part; strip data-specific language.

Examples:
- "Show me critical issues" → type: data, docs_query: "Show me critical issues"
- "How do I connect Jira?" → type: doc, docs_query: "Jira connector setup configuration"
- "What is SQL injection and how many do we have?" → type: mixed, docs_query: "SQL injection vulnerability explanation\""""),
    ("human", "{query}"),
])

FORMATTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful security platform assistant. Answer the user's question
using only the provided context. Be specific. If showing security issues, list ALL of them
completely — do not omit, truncate, or summarize. If referencing documentation, cite the source.

MCP Data:
{mcp_result}

Documentation:
{rag_result}"""),
    ("human", "{query}"),
])
