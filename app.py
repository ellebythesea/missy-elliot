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

import streamlit as st

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


APP_TITLE = "Missy Elliott Style Viral Video Script Generator"


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

    


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="centered")
    st.title(APP_TITLE)
    st.caption("Design every beat in reverse to keep viewers watching.")
    with st.expander("What is the Missy Elliott method?"):
        st.markdown(
            """
            The Missy Elliott method designs videos in reverse: instead of planning
            the end payoff and hoping viewers reach it, you validate attention in
            3‑second beats from the very start. Write the opening 0–3 seconds so
            it's irresistible, then the next 3 seconds, and so on. For each beat,
            ask: “Would I keep watching? Why should anyone care?” This mirrors the
            viewer experience and forces tight hooks, clear payoffs, and steady
            curiosity.
            """
        )

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

    with st.form(key="missy-form", clear_on_submit=False):
        topic = st.text_input(
            "Payoff or main topic*",
            placeholder="e.g., How to save 50% on taxes",
            help="The core payoff or idea the video leads to. Required.",
        ).strip()

        length_s = st.number_input(
            "Approximate length (seconds)",
            min_value=6,
            max_value=600,
            value=60,
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
            st.error(
                "Missing OPENAI_API_KEY environment variable. Set it and rerun the app."
            )
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
        st.markdown(script or "(No content returned)")

        st.divider()
        st.caption(
            "Pro tip: Iterate by tightening the first 3–6 seconds until it's irresistible."
        )


if __name__ == "__main__":
    main()
