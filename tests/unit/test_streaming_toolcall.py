from __future__ import annotations

import pytest

from refaat.domain.models import ToolCallDelta, ToolProposal
from refaat.errors import DependencyUnavailableError
from refaat.programs.toolcall import _ToolCallAccumulator


def _reduce(deltas: list[ToolCallDelta]) -> ToolProposal | None:
    accumulator = _ToolCallAccumulator()
    for delta in deltas:
        accumulator.feed(delta)
    return accumulator.result()


def test_reducer_joins_name_and_split_argument_fragments() -> None:
    proposal = _reduce(
        [
            ToolCallDelta(tool="lookup"),
            ToolCallDelta(arguments_fragment='{"top'),
            ToolCallDelta(arguments_fragment='ic": "roll'),
            ToolCallDelta(arguments_fragment='back"}'),
            ToolCallDelta(done=True),
        ]
    )
    assert proposal == ToolProposal(tool="lookup", arguments={"topic": "rollback"})


def test_reducer_empty_argument_buffer_is_empty_dict() -> None:
    proposal = _reduce([ToolCallDelta(tool="ping"), ToolCallDelta(done=True)])
    assert proposal == ToolProposal(tool="ping", arguments={})


def test_reducer_non_dict_json_collapses_to_empty_dict() -> None:
    proposal = _reduce([ToolCallDelta(tool="ping", arguments_fragment="[1, 2, 3]"), ToolCallDelta(done=True)])
    assert proposal == ToolProposal(tool="ping", arguments={})


def test_reducer_malformed_json_raises_dependency_error() -> None:
    with pytest.raises(DependencyUnavailableError):
        _reduce([ToolCallDelta(tool="ping", arguments_fragment='{"topic": '), ToolCallDelta(done=True)])


def test_reducer_without_a_tool_name_returns_none() -> None:
    assert _reduce([ToolCallDelta(arguments_fragment='{"topic": "x"}'), ToolCallDelta(done=True)]) is None


def test_reducer_keeps_first_tool_name_and_ignores_later_names() -> None:
    proposal = _reduce(
        [
            ToolCallDelta(tool="lookup"),
            ToolCallDelta(tool="search", arguments_fragment='{"topic": "x"}'),
            ToolCallDelta(done=True),
        ]
    )
    assert proposal is not None
    assert proposal.tool == "lookup"
    assert proposal.arguments == {"topic": "x"}


def test_reducer_without_a_terminal_done_returns_none() -> None:
    # A stream that ends without a terminal done is incomplete: no proposal, so
    # nothing is validated, authorized, or executed.
    assert (
        _reduce(
            [
                ToolCallDelta(tool="lookup"),
                ToolCallDelta(arguments_fragment='{"topic": "rollback"}'),
            ]
        )
        is None
    )


def test_reducer_ignores_fragments_after_terminal_done() -> None:
    proposal = _reduce(
        [
            ToolCallDelta(tool="lookup"),
            ToolCallDelta(arguments_fragment='{"topic": "rollback"}'),
            ToolCallDelta(done=True),
            ToolCallDelta(arguments_fragment='{"topic": "TAMPERED"}'),
            ToolCallDelta(tool="evil"),
        ]
    )
    assert proposal == ToolProposal(tool="lookup", arguments={"topic": "rollback"})


def test_reducer_folds_a_co_located_closing_fragment_and_done() -> None:
    # A terminal delta that also carries the closing fragment must not drop it.
    proposal = _reduce(
        [
            ToolCallDelta(tool="lookup"),
            ToolCallDelta(arguments_fragment='{"topic": '),
            ToolCallDelta(arguments_fragment='"rollback"}', done=True),
        ]
    )
    assert proposal == ToolProposal(tool="lookup", arguments={"topic": "rollback"})
