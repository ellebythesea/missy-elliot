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
import re
from urllib.parse import urlparse, parse_qs

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

YOUTUBE_HOSTS = {"www.youtube.com", "youtube.com", "m.youtube.com", "youtu.be"}
INSTAGRAM_HOSTS = {"www.instagram.com", "instagram.com", "m.instagram.com"}


def _is_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False


def _is_youtube(url: str) -> bool:
    try:
        return urlparse(url).netloc.replace(" ", "").lower() in YOUTUBE_HOSTS
    except Exception:
        return False


def _extract_youtube_id(url: str) -> str | None:
    try:
        p = urlparse(url)
        host = p.netloc.lower()
        if host == "youtu.be":
            vid = p.path.lstrip("/")
            return vid or None
        if "watch" in p.path:
            q = parse_qs(p.query)
            vid = q.get("v", [None])[0]
            return vid
        # Shorts or other formats
        m = re.search(r"/(shorts|embed)/([\w-]{6,})", p.path)
        if m:
            return m.group(2)
    except Exception:
        pass
    return None


def _get_video_duration_seconds(url: str) -> int | None:
    """Use yt-dlp to fetch metadata and return duration in seconds (or None)."""
    try:
        import yt_dlp  # type: ignore

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "ignoreerrors": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return None
            # Some extractors nest info in 'entries'
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            return int(info.get("duration")) if info.get("duration") else None
    except ImportError:
        st.warning("Package 'yt-dlp' is not installed. Install it to enable duration checks.")
    except Exception:
        return None
    return None


def _try_youtube_transcript(url: str) -> str | None:
    """Try using youtube-transcript-api. Returns transcript text or None."""
    vid = _extract_youtube_id(url)
    if not vid:
        return None
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

        # Prefer English; allow auto-generated
        segments = YouTubeTranscriptApi.get_transcript(vid, languages=["en", "en-US"])
        text = " ".join([s.get("text", "").strip() for s in segments if s.get("text")])
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return None


def _download_audio_with_ytdlp(url: str) -> Path | None:
    """Download audio to a temp file using yt-dlp and return the path."""
    try:
        import yt_dlp  # type: ignore

        tempdir = Path(tempfile.mkdtemp(prefix="lf_audio_"))
        outtmpl = str(tempdir / "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                return None
            # After postprocess, resulting file extension is mp3
            base = info.get("id") or "audio"
            audio_path = next((p for p in tempdir.glob(f"{base}.*") if p.suffix.lower() in {".mp3", ".m4a", ".aac", ".wav"}), None)
            return audio_path
    except ImportError:
        st.warning("Package 'yt-dlp' is not installed. Install it to enable audio download.")
    except Exception:
        return None
    return None


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
                _ = cm.get_all()  # initialize
                token = cm.get(cookie_name)
                if not (token and _verify_token(token)):
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
        index=0,
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
            url = st.text_input("YouTube or Instagram URL*", placeholder="https://www.youtube.com/watch?v=...")
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
            if not url or not _is_url(url):
                st.error("Please enter a valid YouTube or Instagram URL.")
                st.stop()

            # Duration check (<= 5 minutes)
            dur = _get_video_duration_seconds(url)
            if dur is None:
                st.info("Could not determine video duration; proceeding cautiously.")
            elif dur > 300:
                st.warning("Video is longer than 5 minutes. Please choose a shorter clip.")
                st.stop()

            if not api_key:
                st.error("Missing OPENAI_API_KEY environment variable. Set it and rerun the app.")
                st.stop()

            client = get_openai_client(api_key)

            transcript_text: str | None = None
            with st.spinner("Transcribing audio..."):
                if _is_youtube(url):
                    transcript_text = _try_youtube_transcript(url)
                if not transcript_text:
                    audio_path = _download_audio_with_ytdlp(url)
                    if not audio_path or not audio_path.exists():
                        st.error("Failed to download audio for transcription. Ensure the URL is public.")
                        st.stop()
                    transcript_text = _transcribe_via_whisper(client, audio_path)

            if not transcript_text:
                st.error("Transcription failed. Try another URL or check dependencies.")
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
            st.markdown(analysis or "(No analysis returned)")


if __name__ == "__main__":
    main()
