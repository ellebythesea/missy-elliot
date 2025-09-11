"""Centralized prompts for the Missy Elliott method app."""

from __future__ import annotations

from typing import Optional


# Exact description as requested
MISSY_METHOD_PROMPT = (
    "Use the Missy Elliott method: design the video in reverse. Start from the payoff, "
    "but write the script by validating the opening three seconds first—then the next "
    "three, then the next. At every 3‑second beat ask: Would I keep watching? Why should "
    "anyone care? Keep hooks tight and specific, mimicking the viewer’s experience so you "
    "consistently maintain attention across each beat until the end."
)


def build_user_prompt(topic: str, style: Optional[str], length_s: int) -> str:
    """Create the user prompt from inputs, enforcing a timed 3s breakdown."""
    style_fragment = (style if style and style != "Other" else "any")
    return (
        f"Generate a viral video script for '{topic}' in {style_fragment} style, "
        f"length about {length_s} seconds, using the Missy Elliott method to ensure hooks every 3 seconds.\n"
        "Structure the script as a timed breakdown with clear 3-second beats like: \n"
        "0-3s: [hook]\n3-6s: [next beat]\n6-9s: [next beat]\n... and so on until the target length.\n"
        "For each beat, include:\n"
        "- What is said (voiceover or dialogue)\n"
        "- On-screen text (if any)\n"
        "- Suggested visuals or actions (short)\n"
        "Keep beats punchy, specific, and audience-focused. End with a crisp CTA.\n"
        "Tone and style requirements:\n"
        "- Politically impactful and educational.\n"
        "- Click-baity hooks that create curiosity gaps without misleading.\n"
        "- Surprise and delight with credible facts, stats, or discoveries.\n"
        "- Keep claims accurate and responsibly framed; avoid personal attacks or demeaning language.\n"
        "- Where relevant, mention reputable sources or how to verify claims.\n"
        "- Close with a constructive, non-harassing civic action (learn more, verify, vote, contact reps)."
    )
