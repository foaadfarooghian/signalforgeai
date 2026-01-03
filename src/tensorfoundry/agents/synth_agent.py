"""Stress-test SynthAgent: multi-step research synthesis with citations + critic checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import re
from tensorfoundry.logging.emitter import JsonlEmitter
from tensorfoundry.models.registry import get_provider_for_model


REQUIRED_KEYS = {"answer", "citations", "assumptions", "risks", "confidence"}

@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    text: str


class SynthAgent:
    """
    Stress-test agent:
    - multi-step extraction -> clustering -> conflict resolution -> answer drafting
    - emits tool-style events per step
    - includes a critic pass that enforces schema + citation coverage
    """

    def __init__(self, *, emitter: JsonlEmitter) -> None:
        self.emitter = emitter

    def run(self, task: str, *, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        root = self.emitter.emit(
            event_type="task_received",
            stage="system",
            payload={"task": task, "num_sources": len(sources)},
        )

        srcs = [Source(**s) for s in sources]

        plan = [
            "Extract claims per source",
            "Cluster claims by topic",
            "Resolve conflicts / uncertainty",
            "Draft answer with citations",
            "Critic check and refine",
        ]
        self.emitter.emit(
            event_type="plan_created",
            stage="planner",
            trace_id=root.trace_id,
            parent_span_id=root.span_id,
            payload={"plan": plan},
        )

        # ---- Execute steps (tool-like) ----
        claims = self._extract_claims(task=task, sources=srcs, trace_id=root.trace_id, parent_span_id=root.span_id)
        groups = self._cluster_claims(claims=claims, trace_id=root.trace_id, parent_span_id=root.span_id)
        resolved = self._resolve_conflicts(groups=groups, trace_id=root.trace_id, parent_span_id=root.span_id)
        draft = self._draft_answer(task=task, resolved=resolved, trace_id=root.trace_id, parent_span_id=root.span_id)

        # ---- Critic pass (one shot) ----
        final = self._critic_and_refine(
            task=task,
            draft=draft,
            sources=srcs,
            trace_id=root.trace_id,
            parent_span_id=root.span_id,
        )

        final = _ensure_contract(final)

        self.emitter.emit(
            event_type="task_completed",
            stage="system",
            trace_id=root.trace_id,
            parent_span_id=root.span_id,
            payload={"result_summary": final.get("answer", "")[:120], "plan": plan},
            outcome={"status": "success", "reason": "synth_completed", "confidence": final.get("confidence", 0.0)},
        )

        return {"trace_id": root.trace_id, "result": final}

    # -----------------------------
    # Step 1: extract claims
    # -----------------------------
    def _extract_claims(
        self,
        *,
        task: str,
        sources: List[Source],
        trace_id: str,
        parent_span_id: str,
    ) -> List[Dict[str, Any]]:
        self.emitter.emit(
            event_type="tool_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool": "extract_claims", "num_sources": len(sources)},
        )

        model_id = self._model_id()
        provider = get_provider_for_model(model_id)

        # Keep prompt short and deterministic-ish
        prompt = (
            "You are extracting factual claims for later synthesis.\n"
            f"Task: {task}\n\n"
            "For each source, return JSON list of objects: "
            "{source_id, claims:[{claim, quote}]}.\n"
            "Rules: claims must be supported by quote from the source text.\n\n"
        )
        for s in sources:
            prompt += f"SOURCE {s.source_id} ({s.title}):\n{s.text}\n\n"

        out = provider.generate(prompt=prompt, model_id=model_id, task_type="synth_extract")
        # We keep raw parsing simple: store the model text and let scorer judge structure.
        # You can tighten later with json parsing + retries.
        claims = [{"raw": out.text}]

        self.emitter.emit(
            event_type="model_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"model": model_id, "input_summary": "extract_claims", "output_summary": "claims json-ish"},
            metrics={"latency_ms": out.metrics.latency_ms, "cost_usd": out.metrics.cost_usd, "extra": out.metrics.extra},
            outcome={"result": {"text": out.text[:1200], "text_full": out.text}},
        )
        return claims

    # -----------------------------
    # Step 2: cluster
    # -----------------------------
    def _cluster_claims(
        self,
        *,
        claims: List[Dict[str, Any]],
        trace_id: str,
        parent_span_id: str,
    ) -> Dict[str, Any]:
        self.emitter.emit(
            event_type="tool_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool": "cluster_claims"},
        )

        model_id = self._model_id()
        provider = get_provider_for_model(model_id)

        prompt = (
            "Cluster extracted claims into 2-5 topics.\n"
            "Return JSON: {topics:[{topic, supporting_claims:[...]}]}.\n\n"
            f"CLAIMS:\n{claims[0].get('raw','')}\n"
        )
        out = provider.generate(prompt=prompt, model_id=model_id, task_type="synth_cluster")

        self.emitter.emit(
            event_type="model_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"model": model_id, "input_summary": "cluster_claims", "output_summary": "topics json-ish"},
            metrics={"latency_ms": out.metrics.latency_ms, "cost_usd": out.metrics.cost_usd, "extra": out.metrics.extra},
            outcome={"result": {"text": out.text[:1200], "text_full": out.text}},
        )
        return {"raw": out.text}

    # -----------------------------
    # Step 3: resolve conflicts
    # -----------------------------
    def _resolve_conflicts(
        self,
        *,
        groups: Dict[str, Any],
        trace_id: str,
        parent_span_id: str,
    ) -> Dict[str, Any]:
        self.emitter.emit(
            event_type="tool_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool": "resolve_conflicts"},
        )

        model_id = self._model_id()
        provider = get_provider_for_model(model_id)

        prompt = (
            "Resolve conflicts between sources. If sources disagree, explicitly mark uncertainty.\n"
            "Return JSON: {resolved:[{topic, position, uncertainty, citations:[{source_id, quote}]}]}.\n\n"
            f"TOPICS:\n{groups.get('raw','')}\n"
        )
        out = provider.generate(prompt=prompt, model_id=model_id, task_type="synth_resolve")

        self.emitter.emit(
            event_type="model_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"model": model_id, "input_summary": "resolve_conflicts", "output_summary": "resolved json-ish"},
            metrics={"latency_ms": out.metrics.latency_ms, "cost_usd": out.metrics.cost_usd, "extra": out.metrics.extra},
            outcome={"result": {"text": out.text[:1200], "text_full": out.text}},
        )
        return {"raw": out.text}


    def _is_contract(self, obj: Dict[str, Any]) -> bool:
        return REQUIRED_KEYS.issubset(obj.keys())

    def _coerce_contract(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(obj)
        out.setdefault("answer", "")
        out.setdefault("citations", [])
        out.setdefault("assumptions", [])
        out.setdefault("risks", [])
        out.setdefault("confidence", 0.3)

        if not isinstance(out["answer"], str):
            out["answer"] = str(out["answer"])
        if not isinstance(out["citations"], list):
            out["citations"] = []
        if not isinstance(out["assumptions"], list):
            out["assumptions"] = []
        if not isinstance(out["risks"], list):
            out["risks"] = []
        if not isinstance(out["confidence"], (int, float)):
            out["confidence"] = 0.3
        return out

    # -----------------------------
    # Step 4: draft answer
    # -----------------------------
    def _draft_answer(
        self,
        *,
        task: str,
        resolved: Dict[str, Any],
        trace_id: str,
        parent_span_id: str,
    ) -> Dict[str, Any]:
        self.emitter.emit(
            event_type="tool_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool": "draft_answer"},
        )

        model_id = self._model_id()
        provider = get_provider_for_model(model_id)

        # ✅ Provide resolved as JSON, not resolved.get("raw","")
        resolved_json = json.dumps(resolved, ensure_ascii=False)

        prompt = (
            "Return ONLY valid JSON (no markdown, no commentary) with EXACT keys:\n"
            "{\n"
            '  "answer": string,\n'
            '  "citations": [{"source_id": string, "quote": string}],\n'
            '  "assumptions": [string],\n'
            '  "risks": [string],\n'
            '  "confidence": number\n'
            "}\n\n"
            "Rules:\n"
            "- Every major claim must be supported by a citation.\n"
            "- Use only source_id values present in the RESOLVED content.\n"
            "- If uncertainty exists, mention it and reduce confidence.\n\n"
            f"Task: {task}\n\n"
            f"RESOLVED_JSON:\n{resolved_json}\n"
        )

        out = provider.generate(prompt=prompt, model_id=model_id, task_type="synth_draft")

        self.emitter.emit(
            event_type="model_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"model": model_id, "input_summary": "draft_answer", "output_summary": "contract_json"},
            metrics={"latency_ms": out.metrics.latency_ms, "cost_usd": out.metrics.cost_usd, "extra": out.metrics.extra},
            outcome={"result": {"text": out.text[:2000], "text_full": out.text, "prompt_full": prompt}},  # log preview + full
        )

        parsed = _safe_parse_json(out.text)
        if isinstance(parsed, dict) and all(k in parsed for k in REQUIRED_KEYS):
            return self._coerce_contract(parsed)

        # Fallback: keep raw for debugging but don't break the contract
        fallback = self._coerce_contract({})
        fallback["risks"].append("draft_parse_failed")
        # Keep a short raw snippet for visibility without polluting answer
        fallback["answer"] = "Draft answer generation failed to produce valid JSON."
        return fallback

    # -----------------------------
    # Step 5: critic
    # -----------------------------


    def _critic_and_refine(
        self,
        *,
        task: str,
        draft: Dict[str, Any],
        sources: List[Source],
        trace_id: str,
        parent_span_id: str,
    ) -> Dict[str, Any]:
        self.emitter.emit(
            event_type="critic_started",
            stage="critic",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool": "critic_check"},
        )

        model_id = self._model_id()
        provider = get_provider_for_model(model_id)

        sources_by_id = {s.source_id: s for s in sources}

        # Optional: remove "Irrelevant" sources entirely
        filtered_sources = [s for s in sources if (s.title or "").strip().lower() != "irrelevant"]
        filtered_by_id = {s.source_id: s for s in filtered_sources}
        valid_ids = set(filtered_by_id.keys())

        sources_block = ""
        for s in filtered_sources:
            sources_block += f"SOURCE {s.source_id} ({s.title}):\n{s.text}\n\n"

        # Always pass a valid JSON string of the draft
        draft_json = json.dumps(draft, ensure_ascii=False)

        prompt = (
            "You are a strict JSON validator for an agent output contract.\n"
            "Return ONLY a single JSON object that conforms EXACTLY to this schema:\n"
            "{\n"
            '  "answer": string,\n'
            '  "citations": [{"source_id": string, "quote": string}],\n'
            '  "assumptions": [string],\n'
            '  "risks": [string],\n'
            '  "confidence": number\n'
            "}\n\n"
            "Rules:\n"
            "- Output MUST be valid JSON. No markdown. No explanations. No extra keys.\n"
            "- Each citation.quote MUST be copied verbatim from the corresponding SOURCE text (exact substring).\n"
            "- citation.quote MUST NOT contain '[' or ']'.\n"
            "- Use ONLY these source_id values: " + ", ".join(sorted(valid_ids)) + "\n"
            "- Do NOT cite any source titled 'Irrelevant'.\n"
            "- If the draft is missing/invalid citations, repair them using verbatim quotes from SOURCES.\n"
            "- If sources conflict, reflect uncertainty and lower confidence.\n\n"
            f"Task: {task}\n\n"
            f"SOURCES:\n{sources_block}\n"
            f"DRAFT_JSON:\n{draft_json}\n"
                )

        out = provider.generate(prompt=prompt, model_id=model_id, task_type="synth_critic")

        self.emitter.emit(
            event_type="model_called",
            stage="critic",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"model": model_id, "input_summary": "critic_check", "output_summary": "contract_json"},
            metrics={"latency_ms": out.metrics.latency_ms, "cost_usd": out.metrics.cost_usd, "extra": out.metrics.extra},
            outcome={"result": {"text": out.text[:2000], "text_full": out.text, "prompt_full": prompt}},
        )

        # Must parse to a dict; otherwise return the original draft (not critic prose)

        refined = _safe_parse_json(out.text)
        if isinstance(refined, dict) and self._is_contract(refined):
            refined = self._coerce_contract(refined)

            # Drop invalid ids + ungrounded quotes + bracket placeholders
            cleaned = []
            dropped = 0
            for c in refined["citations"]:
                if not isinstance(c, dict):
                    dropped += 1
                    continue
                sid = c.get("source_id")
                quote = c.get("quote") or ""
                if sid not in valid_ids:
                    dropped += 1
                    continue
                if "[" in quote or "]" in quote:
                    dropped += 1
                    continue
                if not _quote_in_source(quote, filtered_by_id[sid].text):
                    dropped += 1
                    continue
                cleaned.append({"source_id": sid, "quote": quote})

            if dropped:
                refined["risks"].append("critic_dropped_invalid_or_ungrounded_citations")
            refined["citations"] = cleaned

            # If we ended up with no citations, flag it (optionally: trigger a repair retry)
            if not refined["citations"]:
                refined["risks"].append("no_grounded_citations_after_filtering")
                refined["confidence"] = min(float(refined.get("confidence", 0.3)), 0.3)

            return refined
            
        return draft


    def _model_id(self) -> str:
        import os
        return os.getenv("TENSORFOUNDRY_MODEL_ID") or "ollama:ministral-3:8b"


def _ensure_contract(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        x.setdefault("answer", "")
        x.setdefault("citations", [])
        x.setdefault("assumptions", [])
        x.setdefault("risks", [])
        x.setdefault("confidence", 0.3)
        return x
    return {
        "answer": "No valid result produced.",
        "citations": [],
        "assumptions": [],
        "risks": ["final_missing_or_invalid"],
        "confidence": 0.1,
    }

def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    import json
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _quote_in_source(quote: str, source_text: str) -> bool:
    q = _norm(quote)
    t = _norm(source_text)
    if not q or len(q) < 6:
        return False
    return q.lower() in t.lower()
