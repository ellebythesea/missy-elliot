"""Centralized prompts for the Missy Elliott method app."""

from __future__ import annotations

from typing import Optional, Sequence


# Exact description as requested
MISSY_METHOD_PROMPT = (
    "Use the Missy Elliott method: design the video in reverse. Start from the payoff, "
    "but write the script by validating the opening three seconds first—then the next "
    "three, then the next. At every 3‑second beat ask: Would I keep watching? Why should "
    "anyone care? Keep hooks tight and specific, mimicking the viewer’s experience so you "
    "consistently maintain attention across each beat until the end."
)


def build_user_prompt(
    topic: str,
    style: Optional[str],
    length_s: int,
    cta: Optional[str] = None,
    facts: Optional[Sequence[str]] = None,
) -> str:
    """Create the user prompt with the fixed 30-second arc and CTA/fact hooks."""

    raw_style = (style or "").strip()
    if not raw_style or raw_style.lower() == "other":
        style_display = "lighthearted and comedic"
        style_directive = "keep it quick, warm, and a little mischievous"
    else:
        style_display = raw_style.lower()
        style_directive = f"make every line feel {raw_style.lower()}"

    topic_text = (topic or "").strip()
    topic_lines = [ln.strip() for ln in topic_text.splitlines() if ln.strip()]
    topic_summary = " ".join(topic_lines) if topic_lines else topic_text

    embedded_facts = [ln for ln in topic_lines if any(ch.isdigit() for ch in ln)]

    cleaned_facts = [f.strip() for f in (facts or []) if f.strip()]
    if embedded_facts:
        if cleaned_facts:
            cleaned_facts.extend([f for f in embedded_facts if f not in cleaned_facts])
        else:
            cleaned_facts = embedded_facts

    if cleaned_facts:
        facts_instruction = (
            "Integrate every one of these payoff-field facts somewhere in the script, quoting each number or named detail plainly and exactly once: "
            + "; ".join(cleaned_facts)
            + ". Do not paraphrase away the numbers, and never invent new data."
        )
    else:
        facts_instruction = (
            "Ground each beat in believable, specific, verifiable details. When the payoff text includes numbers or named sources, carry them through verbatim. Never invent statistics."
        )

    cta_text = (cta or "").strip()
    if cta_text:
        cta_guideline = (
            f"- End with the provided CTA: \"{cta_text}\". Paraphrase it into 2-3 inspiring, action-oriented sentences (25-35 words total) inside the Final CTA beat, preserving every concrete detail."
        )
    else:
        cta_guideline = (
            "- End with a CTA you invent that naturally follows the story—make it specific (e.g., text a friend, sign a pledge, volunteer) and never default to vague \"learn more\" language. Write 2-3 inspiring, action-oriented sentences (25-35 words total)."
        )

    instructions = "\n".join(
        [
            f"You are a campaign storyteller crafting a {length_s}-second video script about \"{topic_summary}\" in a {style_display} tone.",
            "",
            "Break the story into these exact beats and include each label with its timestamp:",
            "- Hook (0-6s) — earn the scroll-stopping moment with a bold, curiosity-spiking opener.",
            "- Spark (6-12s) — reveal the catalyst or stakes that make the hook matter right now.",
            "- Proof (12-18s) — show the concrete evidence, stat, or lived moment that makes the story undeniable.",
            "- Turn (18-24s) — pivot toward the hopeful path forward, hinting at how momentum builds.",
            "- Final CTA (24-30s) — deliver the CTA with urgency, clarity, and emotional payoff.",
            "",
            "Format requirements:",
            "- For Hook, Spark, Proof, and Turn, write one voiceover line containing 3-4 fluid sentences (35-45 words total).",
            "- For Final CTA, write one voiceover line containing 2-3 sentences (25-35 words) that builds urgency toward the call to action.",
            "- Immediately after each voiceover line, add a new line formatted as `> *Visuals: ...*` describing kinetic supporting footage. Do not mention on-screen text or captions.",
            "- Keep a single blank line between beats and do not add bullet lists, headings, or commentary outside the beats.",
            "",
            "Narrative requirements:",
            "- Every beat must explicitly acknowledge or escalate what came before so the script reads as one continuous narrative.",
            "- Use sharp humor, puns, and fluid, non-repetitive metaphors tailored to the topic.",
            "- Voiceover must flow as complete sentences—never bullet fragments.",
            "- Prioritize precise numbers, percentages, dollar amounts, or timeframes to prove each claim; attribute them to credible sources where possible.",
            f"- Always {style_directive}, keeping engagement high without misleading claims.",
            f"- {facts_instruction}",
            "- Avoid repetitive phrasing (no overusing \"imagine\" or similar openers).",
            cta_guideline,
            "",
            "Return only the formatted beats in plain text using the labels exactly as written above.",
        ]
    )

    return instructions
