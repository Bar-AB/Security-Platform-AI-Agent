import logging
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    blocked: bool
    matched_pattern: str | None = None
    safe_message: str = "I can't process that request."


class InputGuardrail:
    _INJECTION_PATTERNS: tuple[re.Pattern, ...] = tuple(
        re.compile(p, re.IGNORECASE)
        for p in [
            r"ignore\s+(the\s+)?(previous|all|prior|above)\s+instructions?",
            r"disregard\s+(the\s+)?(above|previous|prior|all)",
            r"forget\s+(your\s+)?(instructions?|training|rules?|guidelines?)",
            r"you\s+are\s+now\s+(a|an)\b",
            r"act\s+as\s+(a|an|if)\b",
            r"pretend\s+(you\s+are|to\s+be)",
            r"roleplay\s+as",
            r"\bjailbreak\b",
            r"reveal\s+(your\s+)?(system\s+)?prompt",
            r"show\s+(me\s+)?your\s+(instructions?|system\s+prompt)",
            r"what\s+are\s+your\s+(instructions?|rules?|system\s+prompt)",
            r"<\s*system\s*>",
            r"\[INST\]",
            r"###\s*SYSTEM",
            r"\|im_start\|",
            r"override\s+(your\s+)?(instructions?|training|rules?|safety)",
            r"bypass\s+(your\s+)?(instructions?|training|safety|filter)",
            r"(?:you\s+are|act\s+as)\s+DAN\b",
        ]
    )

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def check(self, query: str) -> GuardrailResult:
        for pattern in self._INJECTION_PATTERNS:
            if pattern.search(query):
                self._logger.warning(
                    "Prompt injection attempt blocked | pattern=%r | query_excerpt=%r",
                    pattern.pattern,
                    query[:100],
                )
                return GuardrailResult(blocked=True, matched_pattern=pattern.pattern)
        return GuardrailResult(blocked=False)

    @staticmethod
    def sanitize_for_xml_context(text: str, *tag_names: str) -> str:
        result = text
        for tag in tag_names:
            result = result.replace(f"</{tag}>", f"&lt;/{tag}&gt;")
        return result
