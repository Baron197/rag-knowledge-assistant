# Deploy the live demo to Streamlit Community Cloud (free, no credit card)

The simplest **free** way to put this app online for a public demo. Streamlit
Community Cloud hosts public apps straight from a public GitHub repo — **no credit
card, no Docker, no server setup**. ~5 minutes.

This app is normally two processes (a Streamlit UI over a separate FastAPI
service), but Community Cloud runs a single process. A small built-in adapter
(`ui/demo_backend.py`) handles that: on a single-process host it boots the FastAPI
service in-process in keyless **`fake`** mode, and the unchanged UI talks to it
locally. It's a no-op anywhere a real API is already running (local dev, Docker, a
VM), so nothing about normal use changes.

---

## 1. Prerequisites
- The repo is already **public on GitHub** (`Baron197/rag-knowledge-assistant`). ✓
- A free account at <https://share.streamlit.io> — sign in **with GitHub**, no card.

## 2. Create the app
1. Go to <https://share.streamlit.io> → **Create app** → **Deploy a public app from GitHub**.
2. Fill in:
   - **Repository:** `Baron197/rag-knowledge-assistant`
   - **Branch:** `main`
   - **Main file path:** `ui/streamlit_app.py`
3. Open **Advanced settings** → set **Python version** to **3.11**. (No secrets needed —
   the demo runs keyless.)
4. Click **Deploy**.

## 3. Wait for the build
Community Cloud clones the repo, installs `requirements.txt`, and launches the app
(first build ~3–5 min). On the **first page load** you'll briefly see
*"Starting the demo backend…"* — that's the adapter building the index and starting
the API in-process (a few seconds, once). Then the app is live at:

```
https://<your-app-name>.streamlit.app
```

You land on the **Guide** page; open **Ask** and try an example question.

## 4. Link it from your README
Send me the URL and I'll add the badge, or paste this near the top of the README:

```markdown
**▶ [Try the live demo](https://<your-app-name>.streamlit.app)** — keyless mode, runs in your browser.

[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-app-name>.streamlit.app)
```

---

## Notes
- **Keyless & free:** it runs the `fake` providers, so there's no API key and no
  cost — the retrieval, citations and telemetry are real; the answer *text* is a
  stand-in (see the README's note on fake mode).
- **Sleeping:** a free app sleeps after inactivity and wakes in a few seconds on
  the next visit — fine for a portfolio demo.
- **Resources:** the free tier is ~1 GB RAM. Fake mode is light (no PyTorch), so it
  fits. `hf`/`openai` modes are heavier/paid and not recommended for a public app.
- **Faster builds (optional):** the app installs the full `requirements.txt`;
  `ragas`, `datasets`, `psycopg` and `pgvector` aren't used on the demo path and can
  be trimmed into a smaller requirements file if you want a quicker build.
- **How the adapter is wired:** `ui/streamlit_app.py` calls
  `demo_backend.ensure_local_backend()` once at startup; it only boots an in-process
  API when `RAG_API_URL` isn't already answering, so it never interferes with the
  normal two-process setup.
