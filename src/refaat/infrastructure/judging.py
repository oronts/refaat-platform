from __future__ import annotations

import re
from collections.abc import Sequence

from refaat.domain.models import ContextPacket, GenerationRequest, JudgeVerdict
from refaat.errors import DependencyUnavailableError
from refaat.ports import Generator

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict answer-quality judge. Score how faithfully the answer is supported by the "
    "evidence on a scale from 0.0 to 1.0. Reply with 'score: <value>' on the first line, then one "
    "sentence of rationale. Judge only support by the evidence, never style."
)
_LABELLED_SCORE = re.compile(r"score\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE)
_BARE_SCORE = re.compile(r"(?<![\w.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\w.])")


class GeneratorJudge:
    """A JudgeModel backed by a text :class:`Generator`.

    Active only when injected: the offline extractive generator cannot judge, so
    ModelJudge stays on its deterministic Faithfulness proxy unless a real model
    is supplied. The judge parses a bounded numeric score from the generation and
    raises :class:`DependencyUnavailableError` when the generation carries no
    parseable score, so ModelJudge degrades to the proxy and no judge score is
    ever fabricated.
    """

    def __init__(self, generator: Generator, *, threshold: float = 0.6) -> None:
        self._generator = generator
        self._threshold = threshold

    @property
    def model_id(self) -> str:
        return f"judge:{self._generator.model_id}"

    def assess(self, query: str, answer: str, evidence: Sequence[str]) -> JudgeVerdict:
        joined = "\n\n".join(evidence) if evidence else "None"
        context = ContextPacket(
            query=query,
            hits=(),
            graph_facts=tuple(evidence),
            session_summary=None,
            profile={},
            supplemental_harness={},
            token_count=0,
            routes=(),
        )
        prompt = f"Question:\n{query}\n\nAnswer to judge:\n{answer}\n\nEvidence:\n{joined}"
        generation = self._generator.generate(
            GenerationRequest(query=prompt, context=context, system_prompt=_JUDGE_SYSTEM_PROMPT)
        )
        return self._verdict(generation.answer)

    def _verdict(self, text: str) -> JudgeVerdict:
        match = _LABELLED_SCORE.search(text) or _BARE_SCORE.search(text)
        if match is None:
            raise DependencyUnavailableError("Judge generation carried no parseable score")
        score = max(0.0, min(1.0, float(match.group(1))))
        rationale = text.strip() or "no rationale"
        return JudgeVerdict(score=score, passed=score >= self._threshold, rationale=rationale)
