import asyncio
import builtins
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import main
from agent.factory import AgentFactory


class TestMain:
    @pytest.fixture
    def async_only_agent(self) -> MagicMock:
        agent = MagicMock()
        agent.invoke.side_effect = TypeError("No synchronous function provided")
        agent.ainvoke = AsyncMock(return_value={"final_response": "Found 3 critical issues."})
        return agent

    def _run(self, agent: MagicMock, inputs: list[str]) -> list[str]:
        printed: list[str] = []
        with (
            patch.object(AgentFactory, "build", return_value=agent),
            patch.object(builtins, "input", side_effect=inputs),
            patch.object(builtins, "print", side_effect=lambda *a, **_: printed.append(str(a))),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ):
            main.main()
        return printed

    def test_uses_async_graph_invocation(self, async_only_agent: MagicMock) -> None:
        printed = self._run(async_only_agent, ["show critical issues", "exit"])
        async_only_agent.ainvoke.assert_awaited_once()
        assert any("Found 3 critical issues." in line for line in printed)

    def test_reuses_one_event_loop_across_turns(self, async_only_agent: MagicMock) -> None:
        loops: list[asyncio.AbstractEventLoop] = []

        async def _record_loop(*_: object, **__: object) -> dict:
            loops.append(asyncio.get_running_loop())
            return {"final_response": "ok"}

        async_only_agent.ainvoke = AsyncMock(side_effect=_record_loop)
        self._run(async_only_agent, ["first question", "second question", "exit"])
        assert len(loops) == 2
        assert loops[0] is loops[1]

    def test_agent_failure_message_does_not_guess_cause(self, async_only_agent: MagicMock) -> None:
        async_only_agent.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        printed = self._run(async_only_agent, ["show critical issues", "exit"])
        assert not any("mock server" in line for line in printed)
        assert any("Something went wrong" in line for line in printed)
