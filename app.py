"""
Missy Elliott Style Viral Video Script Generator

A minimal Streamlit app that collects a video's payoff/topic and a few optional
inputs, then calls OpenAI to generate a timed, 3-second-beat script using the
"Missy Elliott method" (designing the video in reverse, focusing on the
viewer-validated hook every three seconds).

Run with:
    streamlit run app.py

Environment variable required:
    OPENAI_API_KEY (from .env or environment)
"""

from __future__ import annotations

import os
import hmac
import hashlib
import base64
import time
import datetime as dt
import tempfile
from pathlib import Path
# (no URL parsing needed for upload flow)

import streamlit as st
import streamlit.components.v1 as components
import json
import uuid

# Optional: load .env if python-dotenv is installed
try:  # pragma: no cover - optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# The OpenAI Python SDK has two common interfaces depending on the installed
# version. We support both the new Client-based API and the legacy ChatCompletion
# for compatibility.
try:
    from openai import OpenAI  # New-style SDK (>=1.0)
    _HAS_NEW_OPENAI = True
except Exception:  # pragma: no cover - only hit on older SDKs
    import openai  # Legacy SDK (<1.0)
    _HAS_NEW_OPENAI = False

# Local configuration and prompts
from config import MODEL as DEFAULT_MODEL, TEMPERATURE
from prompts import MISSY_METHOD_PROMPT, build_user_prompt

# Optional cookie manager for password remember-me without username field
try:
    import extra_streamlit_components as stx  # type: ignore
    _HAS_STX = True
except Exception:
    _HAS_STX = False


APP_TITLE = "Viral Video Script Generator"


def get_openai_client(api_key: str):
    """Return an object with a unified `chat` completion interface.

    - For new SDKs: returns an OpenAI() client instance.
    - For legacy SDKs: returns the imported `openai` module after setting api_key.
    """
    if _HAS_NEW_OPENAI:
        return OpenAI(api_key=api_key)
    # Legacy fallback
    openai.api_key = api_key
    return openai


def call_openai(
    client,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
) -> str:
    """Create a chat completion using either new or legacy SDKs.

    Returns the assistant message content as a string.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if _HAS_NEW_OPENAI:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    # Legacy API path
    resp = client.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    # Some legacy SDKs return a dict-like object
    choice = resp["choices"][0]
    # `message` can be an object or dict depending on SDK version
    message = choice["message"] if isinstance(choice, dict) else choice.message
    return message["content"] if isinstance(message, dict) else message.content

    

# ----------------------
# Logical Fallacy helpers
# ----------------------

# URL helpers no longer needed after switching to file uploads for analysis




def _transcribe_via_whisper(client, audio_path: Path) -> str | None:
    """Transcribe audio using OpenAI Whisper with new or legacy SDKs."""
    try:
        # New SDK
        if _HAS_NEW_OPENAI:
            with open(audio_path, "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
            # New SDK returns an object with text
            text = getattr(resp, "text", None)
            if text:
                return text.strip()
        else:  # Legacy fallback
            with open(audio_path, "rb") as f:
                try:
                    resp = client.Audio.transcriptions.create(model="whisper-1", file=f)
                    text = resp.get("text") if isinstance(resp, dict) else getattr(resp, "text", None)
                    if text:
                        return text.strip()
                except Exception:
                    f.seek(0)
                    resp = client.Audio.transcribe("whisper-1", f)
                    text = resp.get("text") if isinstance(resp, dict) else getattr(resp, "text", None)
                    if text:
                        return text.strip()
    except Exception:
        return None
    return None


def _enrich_context_with_serp(api_key: str, query: str) -> list[dict]:
    """Use SerpAPI (Google Search) to fetch top organic results.

    Returns a list of {name, description, url} dicts (up to 3), or empty.
    """
    try:
        from serpapi import GoogleSearch  # type: ignore

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": 5,
            "hl": "en",
            "safe": "active",
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        out: list[dict] = []
        for item in results.get("organic_results", [])[:3]:
            out.append(
                {
                    "name": item.get("title"),
                    "description": item.get("snippet"),
                    "url": item.get("link"),
                }
            )
        return out
    except Exception:
        return []


def _analyze_transcript(client, transcript: str, speaker: str, context: str, fallacy_filter: str) -> str:
    """Use gpt-4o-mini to extract quotes, issues, and good responses."""
    max_chars = 12000
    tx = transcript.strip()
    if len(tx) > max_chars:
        tx = tx[:max_chars] + "..."

    system_prompt = (
        "You analyze speech for misinformation, divisive rhetoric, and logical fallacies. "
        "Identify exact quotes (verbatim) and classify issues. Be concise and precise."
    )

    filter_line = (
        "Analyze all fallacies and issues." if fallacy_filter == "All (auto-detect)" else f"Focus on: {fallacy_filter}."
    )

    user_prompt = (
        f"Speaker: {speaker or 'Unknown'}\n"
        f"Context: {context or 'None provided'}\n"
        f"Guidance: {filter_line}\n"
        "Output 3–7 findings. For each, use exactly this format on separate blocks:\n"
        "Quote: [exact quote]\n"
        "Issue: [logical fallacy/divisive rhetoric/lie — brief explanation]\n"
        "Good Response: [concise, factual counterargument]\n\n"
        "Transcript to analyze:\n" + tx
    )

    try:
        result = call_openai(client, system_prompt, user_prompt)
        return result.strip()
    except Exception as e:
        return f"Analysis failed: {e}"


def _parse_analysis_findings(text: str) -> list[dict]:
    """Parse LLM output with lines: Quote:, Issue:, Good Response: into structured blocks.

    Returns a list of dicts with keys: quote, issue, good.
    """
    blocks: list[dict] = []
    if not text:
        return blocks
    # Split on blank lines into candidate blocks
    parts = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    cur: dict | None = None
    for part in parts:
        # A part may contain multiple lines; handle each line
        for line in part.splitlines():
            l = line.strip()
            low = l.lower()
            if low.startswith("quote:"):
                if cur and (cur.get("quote") or cur.get("issue") or cur.get("good")):
                    blocks.append(cur)
                cur = {"quote": l.split(":", 1)[1].strip(), "issue": "", "good": ""}
            elif low.startswith("issue:"):
                cur = cur or {"quote": "", "issue": "", "good": ""}
                cur["issue"] = l.split(":", 1)[1].strip()
            elif low.startswith("good response:"):
                cur = cur or {"quote": "", "issue": "", "good": ""}
                cur["good"] = l.split(":", 1)[1].strip()
            else:
                # Continuation lines: append to last non-empty field
                if cur:
                    if cur.get("good"):
                        cur["good"] = (cur["good"] + " " + l).strip()
                    elif cur.get("issue"):
                        cur["issue"] = (cur["issue"] + " " + l).strip()
                    elif cur.get("quote"):
                        cur["quote"] = (cur["quote"] + " " + l).strip()
        # End of part: close block if we have at least one field
        if cur and (cur.get("quote") or cur.get("issue") or cur.get("good")):
            blocks.append(cur)
            cur = None
    if cur and (cur.get("quote") or cur.get("issue") or cur.get("good")):
        blocks.append(cur)
    # Normalize quotes: strip surrounding quotes
    for b in blocks:
        q = (b.get("quote") or "").strip()
        if q.startswith("[") and q.endswith("]"):
            q = q[1:-1].strip()
        q = q.strip('"')
        b["quote"] = q
    return [b for b in blocks if any(b.values())]


def _generate_hashtags(findings: list[dict]) -> str:
    """Generate 8–13 relevant hashtags based on findings content.

    Uses simple keyword heuristics from the issue/good text.
    """
    import re

    text = " \n".join(
        [
            (f.get("issue") or "") + " " + (f.get("good") or "")
            for f in findings
        ]
    ).lower()

    tags: list[str] = []

    def add(*items: str) -> None:
        for it in items:
            if it not in tags:
                tags.append(it)

    # General baselines
    add("#FactCheck", "#EvidenceBased", "#DataMatters", "#CriticalThinking")

    # Keyword buckets
    if any(k in text for k in ["lie", "false", "fake", "fabricat", "mislead", "exaggerat", "inflate"]):
        add("#Misinformation", "#Disinformation", "#TruthMatters", "#Debunked")

    if any(k in text for k in ["drug", "opioid", "fentanyl", "overdose", "addiction", "narcotic"]):
        add("#DrugFacts", "#Overdose", "#HarmReduction", "#PublicHealth", "#Addiction")

    if any(k in text for k in ["unodc", "united nations", "u.n.", "un "]):
        add("#UNODC", "#GlobalHealth")

    if any(k in text for k in ["statistic", "evidence", "data", "numbers", "source", "citation"]):
        add("#EvidenceBased", "#DataIntegrity", "#MediaLiteracy", "#ContextMatters")

    if any(k in text for k in ["health", "public health", "global"]):
        add("#PublicHealth", "#GlobalHealth")

    if any(k in text for k in ["fallacy", "strawman", "ad hominem", "whatabout", "false cause", "slippery"]):
        add("#LogicalFallacy")

    # Ensure 8–13 tags by padding with defaults and trimming if necessary
    defaults = [
        "#TruthMatters",
        "#MediaLiteracy",
        "#ContextMatters",
        "#Accountability",
        "#StayInformed",
        "#CivicDialogue",
        "#GlobalHealth",
        "#PublicHealth",
        "#HarmReduction",
    ]
    for d in defaults:
        if len(tags) >= 13:
            break
        if d not in tags:
            tags.append(d)

    if len(tags) < 8:
        # Final fallback to reach minimum count
        extras = ["#EvidenceMatters", "#DataDriven", "#CheckTheFacts", "#StayEvidenceBased"]
        for e in extras:
            if len(tags) >= 8:
                break
            if e not in tags:
                tags.append(e)

    return " ".join(tags[:13])


def _split_issue(issue: str) -> tuple[str, str]:
    """Split an issue string into (name, description).

    Heuristics handle common separators like em dash, hyphen, or colon.
    """
    s = (issue or "").strip()
    if not s:
        return "", ""
    # Prefer em dash separators
    for sep in [" — ", " —", "— ", "—"]:
        if sep in s:
            left, right = s.split(sep, 1)
            return left.strip(), right.strip()
    # Fallback to spaced hyphen
    if " - " in s:
        left, right = s.split(" - ", 1)
        return left.strip(), right.strip()
    # Fallback to colon
    if ": " in s:
        left, right = s.split(": ", 1)
        return left.strip(), right.strip()
    # Last resort: first word as name
    parts = s.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return s, ""


def _render_copy_button(label: str, text: str) -> None:
    """Render a simple HTML copy button using components (unique ID each time)."""
    btn_id = f"copybtn-{uuid.uuid4().hex}"
    safe_text = json.dumps(text or "")
    safe_label = json.dumps(label)
    components.html(
        f"""
        <div style='margin: 0.25rem 0 0.5rem 0;'>
          <button id='{btn_id}' style='padding:6px 10px; border-radius:6px; border:1px solid #ccc; cursor:pointer;'>
            {label}
          </button>
        </div>
        <script>
          const btn = document.getElementById('{btn_id}');
          if (btn) {{
            const original = {safe_label};
            btn.addEventListener('click', async () => {{
              try {{
                await navigator.clipboard.writeText({safe_text});
                btn.innerText = 'Copied!';
                setTimeout(() => btn.innerText = original, 1200);
              }} catch (e) {{
                btn.innerText = 'Copy failed';
                setTimeout(() => btn.innerText = original, 1500);
              }}
            }});
          }}
        </script>
        """,
        height=60,
    )


def _render_capped_image(img_source, max_height: int = 300, caption: str | None = None) -> None:
    """Render an image capped to a max height using an HTML <img>.

    img_source can be a filesystem path (str/Path) or raw bytes.
    """
    try:
        if isinstance(img_source, (str, Path)):
            with open(img_source, "rb") as f:
                data = f.read()
        else:  # assume bytes-like
            data = img_source
        b64 = base64.b64encode(data).decode()
        components.html(
            f"""
            <div style='display:flex; justify-content:center;'>
              <img src='data:image/png;base64,{b64}' alt='Preview image' style='max-height:{max_height}px; width:auto; object-fit:contain; border-radius:6px; border:1px solid #eee;'>
            </div>
            """,
            height=max(60, max_height + 20),
        )
        if caption:
            st.caption(caption)
    except Exception:
        st.info("Image preview unavailable (failed to render).")


# ----------------------
# Local media helpers (upload)
# ----------------------

def _ffprobe_duration_seconds(path: Path) -> int | None:
    """Use ffprobe to get media duration in seconds. Returns None if unavailable."""
    try:
        import subprocess, shlex
        cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(str(path))}"
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
        if out:
            return int(float(out))
    except Exception:
        return None
    return None


def _extract_frame_screenshot(video_path: Path, time_s: float = 0.5) -> Path | None:
    """Extract a single frame as PNG from a media file using ffmpeg."""
    try:
        import subprocess, shlex
        out_dir = Path(tempfile.mkdtemp(prefix="lf_frame_"))
        out_path = out_dir / "frame.png"
        cmd = (
            f"ffmpeg -y -ss {time_s} -i {shlex.quote(str(video_path))} -frames:v 1 "
            f"-q:v 2 {shlex.quote(str(out_path))}"
        )
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path if out_path.exists() else None
    except Exception:
        return None


def _extract_audio_from_media(media_path: Path) -> Path | None:
    """Extract audio track to MP3 using ffmpeg. Returns path or None."""
    try:
        import subprocess, shlex
        out_dir = Path(tempfile.mkdtemp(prefix="lf_audio_local_"))
        out_path = out_dir / "audio.mp3"
        cmd = f"ffmpeg -y -i {shlex.quote(str(media_path))} -vn -acodec libmp3lame -q:a 2 {shlex.quote(str(out_path))}"
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path if out_path.exists() else None
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="centered")
    st.title(APP_TITLE)
    st.caption("Design in reverse. Analyze rhetoric. Ship better videos.")
    # Info expander moved below auth so it doesn't appear on the password page

    # Optional password gate with monthly remember-me cookie and NO username field.
    app_password = os.getenv("APP_PASSWORD", "")
    if not app_password:
        try:
            app_password = st.secrets.get("APP_PASSWORD", "")
        except Exception:
            pass

    # Cookie settings for remember-me auth
    cookie_name = os.getenv("APP_AUTH_COOKIE_NAME", "missy_auth")
    signature_key = os.getenv("APP_AUTH_SIGNATURE_KEY", "change-me")
    try:
        expiry_days = int(os.getenv("APP_AUTH_EXPIRY_DAYS", "30"))
    except ValueError:
        expiry_days = 30

    if app_password:
        is_unlocked = bool(st.session_state.get("_auth_ok"))

        def _make_token(expiry_ts: int) -> str:
            msg = str(expiry_ts).encode()
            sig = hmac.new(signature_key.encode(), msg, hashlib.sha256).digest()
            sig_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
            return f"{expiry_ts}.{sig_b64}"

        def _verify_token(token: str) -> bool:
            try:
                expiry_s, sig = token.split(".", 1)
                expiry_ts = int(expiry_s)
                if time.time() > expiry_ts:
                    return False
                msg = expiry_s.encode()
                expected = base64.urlsafe_b64encode(
                    hmac.new(signature_key.encode(), msg, hashlib.sha256).digest()
                ).decode().rstrip("=")
                return hmac.compare_digest(sig, expected)
            except Exception:
                return False

        if not is_unlocked:
            if _HAS_STX:
                cm = stx.CookieManager()
                _ = cm.get_all()  # initialize component
                token = cm.get(cookie_name)

                # CookieManager may need one render to hydrate cookies. Pause once.
                if token is None:
                    if not st.session_state.get("_cookie_checked"):
                        st.session_state["_cookie_checked"] = True
                        st.stop()
                    else:
                        # If still None after hydration attempt, treat as absent.
                        token = ""

                # If we have a valid token, unlock without prompting.
                if token and _verify_token(token):
                    st.session_state["_auth_ok"] = True
                    st.rerun()

                # Otherwise, show the password form.
                with st.form("password-form"):
                    pwd = st.text_input("Enter app password", type="password").strip()
                    submit_pwd = st.form_submit_button("Unlock")
                if submit_pwd:
                    if hmac.compare_digest(pwd, app_password.strip()):
                        exp_dt = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=expiry_days)
                        exp_ts = int(exp_dt.timestamp())
                        # Set cookie and mark session as unlocked for immediate access
                        cm.set(cookie_name, _make_token(exp_ts), expires_at=exp_dt)
                        st.session_state["_auth_ok"] = True
                        st.success("Unlocked")
                        st.rerun()
                    else:
                        st.error("Incorrect password. Please try again.")
                st.stop()
            else:
                # Fallback: simple in-session password (requires each new browser session)
                if "_auth_ok" not in st.session_state:
                    st.session_state["_auth_ok"] = False
                if not st.session_state["_auth_ok"]:
                    with st.form("password-form"):
                        pwd = st.text_input("Enter app password", type="password").strip()
                        submit_pwd = st.form_submit_button("Unlock")
                    if submit_pwd:
                        if hmac.compare_digest(pwd, app_password.strip()):
                            st.session_state["_auth_ok"] = True
                            st.success("Unlocked")
                            st.rerun()
                        else:
                            st.error("Incorrect password. Please try again.")
                    st.stop()
        else:
            # Already unlocked: offer logout if cookies are enabled
            if _HAS_STX:
                cm = stx.CookieManager()
                if st.sidebar.button("Logout"):
                    try:
                        cm.delete(cookie_name)
                    except Exception:
                        cm.set(cookie_name, "", expires_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1))
                    st.session_state.pop("_auth_ok", None)
                    st.rerun()

    # Read API key once and cache in session for UX transparency
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # Streamlit Cloud or local secrets.toml support
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass

    # Removed standalone Missy Elliott explainer; brief descriptions are shown near the mode dropdown instead.

    # Mode switch
    mode = st.selectbox(
        "Mode",
        options=[
            "Missy Elliott",
            "Logical Fallacy",
        ],
        index=1,
        key="_mode",
        help="Generate 3‑second‑beat scripts or analyze a video for logical fallacies.",
    )
    # Show concise per‑mode explanation below the dropdown
    mode_desc = {
        "Missy Elliott": "Generate 3-second-beat video scripts in Missy Elliott's style.",
        "Logical Fallacy": "Analyze a short video for logical fallacies, divisive rhetoric, or misleading claims.",
    }
    st.caption(mode_desc.get(mode, ""))

    if mode == "Missy Elliott":
        with st.form(key="missy-form", clear_on_submit=False):
            topic = st.text_area(
                "Payoff or main topic*",
                placeholder="Describe the payoff or main topic \nE.g., This is why you need to vote for Proposition 50 on Nov 4th, 2025.",
                help="The core payoff or idea the video leads to. You can write a short paragraph.",
                height=120,
            ).strip()

            length_s = st.number_input(
                "Approximate length (seconds)",
                min_value=6,
                max_value=600,
                value=30,
                step=3,
                help="Used to size the number of 3-second beats.",
            )

            style = st.selectbox(
                "Video style (optional)",
                options=["Educational", "Recipe", "Comedy", "Motivational", "Other"],
                index=0,
            )

            submitted = st.form_submit_button("Generate Script ✨")

        if submitted:
            if not topic:
                st.error("Please enter the video's payoff or main topic.")
                st.stop()

            if not api_key:
                st.error("Missing OPENAI_API_KEY environment variable. Set it and rerun the app.")
                with st.expander("How to set OPENAI_API_KEY"):
                    st.code(
                        """
export OPENAI_API_KEY='sk-...'
streamlit run app.py
                        """.strip(),
                        language="bash",
                    )
                    st.markdown(
                        "- Or set it in Streamlit Cloud: App settings → Secrets or Environment variables\n"
                        "- Or add it to a local `.env` file (see `.env.example`)"
                    )
                st.stop()

            with st.spinner("Generating your Missy Elliott style script..."):
                try:
                    client = get_openai_client(api_key)
                    system_prompt = MISSY_METHOD_PROMPT
                    user_prompt = build_user_prompt(topic, style, int(length_s))
                    script = call_openai(
                        client,
                        system_prompt,
                        user_prompt,
                        model=DEFAULT_MODEL,
                        temperature=TEMPERATURE,
                    )
                except Exception as e:  # Broad catch to show a friendly error
                    st.error("There was an error generating the script. Please try again.")
                    st.caption(f"Details: {e}")
                    return

            st.subheader("Generated Script")

            def render_copy_button(text: str, label: str = "Copy script") -> None:
                btn_id = f"copybtn-{uuid.uuid4().hex}"
                safe_text = json.dumps(text or "")
                safe_label = json.dumps(label)
                components.html(
                    f"""
                    <div style='margin: 0.25rem 0 0.5rem 0;'>
                      <button id='{btn_id}' style='padding:6px 10px; border-radius:6px; border:1px solid #ccc; cursor:pointer;'>
                        {label}
                      </button>
                    </div>
                    <script>
                      const btn = document.getElementById('{btn_id}');
                      if (btn) {{
                        const original = {safe_label};
                        btn.addEventListener('click', async () => {{
                          try {{
                            await navigator.clipboard.writeText({safe_text});
                            btn.innerText = 'Copied!';
                            setTimeout(() => btn.innerText = original, 1200);
                          }} catch (e) {{
                            btn.innerText = 'Copy failed';
                            setTimeout(() => btn.innerText = original, 1500);
                          }}
                        }});
                      }}
                    </script>
                    """,
                    height=60,
                )

            render_copy_button(script)
            st.markdown(script or "(No content returned)")
            render_copy_button(script)

            st.divider()
            st.caption("Pro tip: Iterate by tightening the first 3–6 seconds until it's irresistible.")
    else:
        with st.form(key="logical-fallacy-form", clear_on_submit=False):
            uploaded = st.file_uploader(
                "Upload video or audio (≤5 minutes)",
                type=["mp4", "mov", "mkv", "webm", "m4v", "mp3", "wav", "m4a", "aac"],
                accept_multiple_files=False,
                help="Upload a short clip. We'll grab a frame to confirm it's ready.",
            )

            # Prepare temp file if uploaded to enable preview screenshot
            temp_media_path: Path | None = None
            if uploaded is not None:
                tmp_dir = Path(tempfile.mkdtemp(prefix="lf_upload_"))
                temp_media_path = tmp_dir / uploaded.name
                with open(temp_media_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                # Extract and show a screenshot if it's a video container
                screenshot = _extract_frame_screenshot(temp_media_path)
                if screenshot and screenshot.exists():
                    _render_capped_image(screenshot, max_height=300, caption="Preview frame")
                else:
                    st.info("Preview frame unavailable (audio-only or ffmpeg not found).")

            speaker = st.text_input("Speaker's name", placeholder="e.g., John Doe")
            ctx = st.text_area(
                "Context (optional)",
                placeholder="Recent event, topic, or any helpful context",
                height=100,
            )
            fallacy = st.selectbox(
                "Logical fallacy focus",
                options=[
                    "All (auto-detect)",
                    "Ad Hominem",
                    "Strawman",
                    "False Dichotomy",
                    "Appeal to Emotion",
                    "Slippery Slope",
                ],
                index=0,
            )
            submitted = st.form_submit_button("Analyze Video")

        if submitted:
            if uploaded is None:
                st.error("Please upload a short video or audio file.")
                st.stop()

            if not api_key:
                st.error("Missing OPENAI_API_KEY environment variable. Set it and rerun the app.")
                st.stop()

            # Ensure temp path exists (recompute if needed)
            if temp_media_path is None:
                tmp_dir = Path(tempfile.mkdtemp(prefix="lf_upload_"))
                temp_media_path = tmp_dir / uploaded.name
                with open(temp_media_path, "wb") as f:
                    f.write(uploaded.getbuffer())

            # Duration check (<= 5 minutes)
            dur = _ffprobe_duration_seconds(temp_media_path)
            if dur is None:
                st.info("Could not determine media duration; proceeding cautiously.")
            elif dur > 300:
                st.warning("Media is longer than 5 minutes. Please choose a shorter clip.")
                st.stop()

            client = get_openai_client(api_key)

            with st.spinner("Transcribing audio..."):
                # If audio-only extension, transcribe directly; else extract audio first
                audio_exts = {".mp3", ".wav", ".m4a", ".aac"}
                if temp_media_path.suffix.lower() in audio_exts:
                    audio_path = temp_media_path
                else:
                    audio_path = _extract_audio_from_media(temp_media_path)
                if not audio_path or not Path(audio_path).exists():
                    st.error("Failed to extract audio for transcription. Ensure ffmpeg is installed.")
                    st.stop()

                transcript_text = _transcribe_via_whisper(client, audio_path)

            if not transcript_text:
                st.error("Transcription failed. Try another file or check dependencies.")
                st.stop()

            # Optional: SerpAPI context enrichment
            gkey = os.getenv("SERPAPI_API_KEY", "")
            enriched_bits = []
            if gkey:
                query = f"{speaker} {ctx}".strip() or speaker or ctx or ""
                if query:
                    enriched_bits = _enrich_context_with_serp(gkey, query)

            if enriched_bits:
                with st.expander("Context enrichment (Search)"):
                    for item in enriched_bits:
                        line = f"- {item.get('name') or ''}: {item.get('description') or ''}"
                        if item.get("url"):
                            line += f" — {item['url']}"
                        st.markdown(line)

            with st.spinner("Analyzing transcript for issues..."):
                analysis = _analyze_transcript(client, transcript_text, speaker, ctx, fallacy)

            st.subheader("Analysis Results")
            # Format analysis into blocks per spec:
            # 1) Bolded quote
            # 2) Bolded fallacy name - explanation
            # 3) Response (not bolded)
            # 4) Horizontal rule between findings
            findings = _parse_analysis_findings(analysis)
            if findings:
                formatted_sections = []
                for f in findings:
                    quote = f.get("quote", "").strip()
                    issue = f.get("issue", "").strip()
                    good = f.get("good", "").strip()
                    section_lines = []
                    if quote:
                        section_lines.append(f"**\"{quote}\"**")
                    if issue:
                        name, desc = _split_issue(issue)
                        if name and desc:
                            section_lines.append(f"**{name}** - {desc}")
                        else:
                            # If we can't split, just bold the whole issue
                            section_lines.append(f"**{issue}**")
                    if good:
                        section_lines.append(good)
                    formatted_sections.append("\n\n".join(section_lines))

                body = ("\n\n---\n\n".join(formatted_sections)).strip()
                hashtags = _generate_hashtags(findings)
                formatted_analysis = (body + ("\n\n" + hashtags if hashtags else "")).strip()
                st.markdown(formatted_analysis)
                _render_copy_button("Copy analysis", formatted_analysis)
            else:
                st.markdown(analysis or "(No analysis returned)")
                if analysis:
                    _render_copy_button("Copy analysis", analysis)

            # Also generate a 30-second Missy Elliott style response video script
            st.subheader("30s Response Script")
            with st.spinner("Generating 30-second response script..."):
                sys = MISSY_METHOD_PROMPT
                resp_user_prompt = (
                    "Generate a 30-second response video script that addresses the misleading or problematic "
                    "statements identified below. Use tight 3-second beats exactly as in the Missy Elliott method "
                    "(0-3s, 3-6s, ...). Keep tone factual, constructive, and audience-friendly. If useful, reference "
                    "the speaker or context.\n\n"
                    f"Speaker: {speaker or 'Unknown'}\n"
                    f"Context: {ctx or 'None provided'}\n"
                    "Findings to address (use as input; do not repeat labels):\n" + (analysis or "")
                )
                try:
                    response_script = call_openai(client, sys, resp_user_prompt, model=DEFAULT_MODEL, temperature=TEMPERATURE)
                except Exception as e:
                    response_script = f"Script generation failed: {e}"
            st.markdown(response_script or "(No content returned)")
            _render_copy_button("Copy 30s response script", response_script or "")


if __name__ == "__main__":
    main()
