# Missy Elliott Style Viral Video Script Generator

A minimal Streamlit app that generates timed, 3-second-beat video scripts using the Missy Elliott method.

## Quick Start
- Python 3.9+
- Create and activate a virtualenv
  - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
  - Windows (Powershell): `py -m venv .venv; .venv\\Scripts\\Activate.ps1`
- Install deps: `pip install -r requirements.txt`
- Set env vars:
  - Copy `.env.example` to `.env`
  - Set `OPENAI_API_KEY` in `.env`
- Run: `streamlit run app.py`

## Configuration
- `OPENAI_MODEL` (default `gpt-4o`) and `OPENAI_TEMPERATURE` (default `0.8`) can be set via environment variables or `.env`.
- See `config.py` and `prompts.py` for centralized settings and prompts.

### Optional password gate
Set `APP_PASSWORD` (via `.env`, host env, or Streamlit Cloud Secrets) to require a password before the app UI loads. If `APP_PASSWORD` is unset, the app is open.

Monthly remember-me (cookie):
- Uses `extra-streamlit-components` (already in `requirements.txt`). No username field; password only.
- Set cookie settings in `.env` or Secrets:
  - `APP_AUTH_COOKIE_NAME=missy_auth`
  - `APP_AUTH_SIGNATURE_KEY=<random-string>`
  - `APP_AUTH_EXPIRY_DAYS=30` (asks for password roughly once per month)
- Click “Logout” in the sidebar to clear the cookie early.

## Files
- `app.py`: Streamlit UI and OpenAI call
- `config.py`: OpenAI model/temperature config via env
- `prompts.py`: Exact system prompt and user prompt builder
- `.env.example`: Copy to `.env` and fill in secrets
- `.gitignore`: Prevents committing `.env` and local artifacts

## GitHub Actions Example
Add your OpenAI API key as a repository secret:
- GitHub → Repo → Settings → Secrets and variables → Actions → New repository secret
- Name: `OPENAI_API_KEY`

Example workflow (`.github/workflows/ci.yml`) that installs dependencies and verifies the secret is plumbed into the job (no external calls):

```yaml
name: CI
on:
  push:
  pull_request:

jobs:
  checks:
    runs-on: ubuntu-latest
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      OPENAI_MODEL: gpt-4o
      OPENAI_TEMPERATURE: "0.8"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Verify environment and imports
        run: |
          python - <<'PY'
          import os
          assert os.getenv('OPENAI_API_KEY'), 'OPENAI_API_KEY not set in env'
          import streamlit, openai
          print('streamlit', streamlit.__version__)
          print('openai', getattr(openai, '__version__', 'legacy'))
          print('OK')
          PY
```

> Note: This example does not make network calls. It only verifies imports and that `OPENAI_API_KEY` is available to the job.

## Docker
Build and run the app in a container.

Build image:

```bash
docker build -t missy-elliott-app .
```

Run (pass your API key):

```bash
docker run -p 8501:8501 -e OPENAI_API_KEY='sk-...' missy-elliott-app
```

Run with a local `.env` file:

```bash
docker run --env-file .env -p 8501:8501 missy-elliott-app
```

## Deployment
- Any host that supports `streamlit run app.py` works (Streamlit Cloud, Fly.io, Railway, etc.).
- Provide `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`, `OPENAI_TEMPERATURE`) as environment variables in your hosting platform.

### Streamlit Cloud
Two easy ways to provide your OpenAI key:
- Environment variables: In your app’s Settings → Advanced → Environment variables, add `OPENAI_API_KEY`.
- Secrets: In app Settings → Secrets, add `OPENAI_API_KEY = "sk-..."`. The app will read `st.secrets["OPENAI_API_KEY"]` if the env var is not set.
  - You can also add `APP_PASSWORD = "your-password"` to require a password.

Local example for secrets: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set your key.

## License
MIT (or your preferred license)

## Dependency Pinning
Keep reproducible installs by pinning exact versions.

Option A — pip-tools (recommended)
- Edit `requirements.in` (top-level, unpinned deps)
- Create a venv and install pip-tools: `make deps-tools`
- Generate a pinned `requirements.txt`: `make lock`
- Commit both files: `requirements.in` (source) and `requirements.txt` (compiled)

Option B — lock snapshot
- After `pip install -r requirements.txt`, run `pip freeze > requirements.lock.txt`
- Have CI/Docker install from the lock file instead of the loose list

Notes
- Dockerfile currently installs from `requirements.txt`. If you adopt Option B, update it to use `requirements.lock.txt`.
