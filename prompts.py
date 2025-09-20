"""Centralized prompts for the Missy Elliott method app."""

from __future__ import annotations

from typing import Optional


REBUTTAL_TONES = {
    "Witty": "Clever, sharp, and playful. Uses humor to land the point without diluting the seriousness.",
    "Sarcastic": "Dry and cutting. Highlights the absurdity of the opponent’s statement with pointed irony.",
    "Empowered": "Confident, inspiring, and forward-looking. Centers community power and determination.",
    "Logical": "Calm, fact-forward, and methodical. Walks through the evidence and dismantles the claim with receipts.",
    "Fallacy": "Instructional tone that calls out the exact logical fallacy or misinformation tactic at play and redirects viewers to the truth.",
}


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


QUOTE_EXTRACTION_SYSTEM_PROMPT = (
    "You are a rapid-response researcher. Extract the most extreme or controversial"
    " quotes directly from transcripts. Return concise, verbatim quotes that opponents"
    " can use in rebuttal videos."
)


def extract_key_quotes_prompt(transcript: str) -> str:
    """Prompt for extracting 3-5 high-impact quotes from a transcript."""
    return (
        "From the transcript below, extract 3-5 key quotes that are extreme, controversial, or"
        " politically damaging. Each quote must be verbatim from the transcript."
        " Respond with a numbered list only.\n\n"
        f"Transcript:\n{transcript}"
    )


def build_script_prompt(
    quote: str,
    breakdown: str,
    personal: str,
    final: str,
    cta: str,
    rebuttal_style: str,
) -> str:
    """Prompt for generating a rebuttal script with an optional style overlay."""
    base_prompt = (
        "Generate a 30-60 second video script. Structure:\n"
        f"1. [Opening clip] - Quote: \"{quote}\"\n"
        f"2. [Candidate rebuttal] - Breakdown: {breakdown}\n"
        f"3. [Personal tone] - {personal}\n"
        f"4. [Call to action] - {final}\n"
        f"5. [End screen] - Display: {cta}\n"
        "Make it energetic, rhythmic, with a 2026 election focus (Nov 3, early voting Oct).\n"
    )
    if rebuttal_style and rebuttal_style != "None":
        base_prompt += f"Apply a {rebuttal_style} tone to the rebuttal section."
    return base_prompt.strip()


SYSTEM_PROMPT = (
    "You are a political content strategist creating viral rebuttal videos. Scripts should be sharp,"
    " confident, and end with a strong CTA to vote."
)


REBUTTAL_SYSTEM_PROMPT = (
    "You craft rapid-response political rebuttals that are punchy, credible, and built for vertical video."
    " Deliver concise copy that feels ready to record as a direct-to-camera response."
)


def build_rebuttal_variants_prompt(transcript: str) -> str:
    tones_description = "\n".join([f"- {tone}: {desc}" for tone, desc in REBUTTAL_TONES.items()])
    return (
        "You are given a transcript of an opponent's clip. Craft five distinct rebuttals after the clip plays.\n"
        "Each rebuttal should be 3-4 punchy sentences (max ~90 words) and speak directly to voters.\n"
        "Use the tone guidance below and keep copy camera-ready (no hashtags, no stage directions).\n"
        f"\nTranscript:\n{transcript}\n\n"
        "Tone guidance:\n"
        f"{tones_description}\n\n"
        "Return a single JSON object where each key is the tone name and the value is the rebuttal text."
        " Use plain strings (no markdown, no extra formatting)."
    )


def build_vote_script_prompt(
    transcript: str,
    speaker: str,
    core_rebuttal: str,
    personal_story: str,
    call_to_action: str,
    onscreen_message: str,
) -> str:
    return (
        "Create a 30-60 second vertical video script using 3-second beats (0-3s, 3-6s, ...).\n"
        "The opponent clip plays first, then the candidate appears with the rebuttal.\n"
        "Provide only the spoken lines with timestamps, no visual callouts.\n"
        "Weave in the edited rebuttal copy, the personal story, and end with the provided CTA and on-screen message.\n"
        "Requirements:\n"
        "1. Start with the opponent clip. Use the speaker name and reproduce the transcript text verbatim without shortening or paraphrasing."
        "   You may span multiple beats for the clip if needed, but do not alter the wording.\n"
        "2. Every beat after the opponent clip must begin with `You:` followed by the candidate's line.\n"
        "3. Maintain strict 3-second beat timestamps and keep lines concise and record-ready.\n"
        "4. Ensure the flow feels like: opponent clip → You: clapback → You: story → You: CTA → You: end screen.\n"
        "Call out the final call to action exactly once as a graphic overlay using this format: Graphic overlay: \"<text>\"."
        "Close with an end screen beat that uses the on-screen message exactly or with minimal polish.\n\n"
        f"Speaker: {speaker or 'Unknown'}\n"
        f"Opponent transcript (edited):\n{transcript}\n\n"
        f"Core rebuttal copy to integrate:\n{core_rebuttal}\n\n"
        f"Personal connection story:\n{personal_story}\n\n"
        f"Final CTA (use exactly, do not rewrite):\n{call_to_action}\n\n"
        f"End screen text:\n{onscreen_message}"
    )
