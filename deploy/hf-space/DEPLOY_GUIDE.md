# Deploy the live demo to Hugging Face Spaces (free, no credit card)

This runs the whole app — FastAPI service **and** Streamlit UI — in one free
Hugging Face Space. Defaults to the keyless `fake` providers, so it costs nothing
and needs no API key. Total time: ~10 minutes, most of it the first build.

The Space itself only needs **two files** (`Dockerfile` + `README.md`); the app
code is cloned from GitHub at build time.

---

## 1. Create a Hugging Face account
Sign up at <https://huggingface.co/join> — free, **no credit card**.

## 2. Create a new Space
Go to <https://huggingface.co/new-space> and set:
- **Owner:** your username
- **Space name:** `rag-knowledge-assistant`
- **License:** MIT
- **SDK:** **Docker** → **Blank** template
- **Hardware:** *CPU basic · 2 vCPU · 16 GB* (the free default)
- **Visibility:** Public

Click **Create Space**. This creates an empty git repo for the Space.

## 3. Add the two files
Clone the Space repo and copy this folder's files into its **root**:

```bash
git clone https://huggingface.co/spaces/<your-username>/rag-knowledge-assistant
cd rag-knowledge-assistant

# from your local clone of the GitHub repo:
cp "D:/Portfolios/RAG Project/deploy/hf-space/Dockerfile"       ./Dockerfile
cp "D:/Portfolios/RAG Project/deploy/hf-space/space_README.md"  ./README.md
```

> The `Dockerfile` clones the app from GitHub (`Baron197/rag-knowledge-assistant`)
> at build time, so you do **not** copy the source into the Space. `README.md`
> carries the Space's config (the YAML frontmatter at the top).

## 4. Push (authenticate with an access token)
Create a **write** token at <https://huggingface.co/settings/tokens>, then:

```bash
git add Dockerfile README.md
git commit -m "Deploy RAG Knowledge Assistant (keyless demo)"
git push
# Username: <your-username>   Password: <paste the write token>
```

## 5. Watch it build
Open your Space page. The **Building** logs stream live (first build ~3–5 min:
it installs deps, then on boot ingests the corpus and starts both processes).
When the badge turns **Running**, the app is live at:

```
https://<your-username>-rag-knowledge-assistant.hf.space
```

Open it → you land on the **Guide** page; try **Ask** with an example question.

## 6. Link it from your GitHub README
Add a demo line near the top of the main README (send me the URL and I'll add it,
or paste this yourself):

```markdown
**▶ [Try the live demo](https://huggingface.co/spaces/<your-username>/rag-knowledge-assistant)** — keyless mode, runs in your browser.

[![🤗 Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-HF%20Spaces-yellow)](https://huggingface.co/spaces/<your-username>/rag-knowledge-assistant)
```

---

## Optional: switch this Space to real answers (`hf` mode)
Free, still no API key, but slower on the CPU tier (~5–15 s per answer with the
small 0.5B model).

1. In the Space's `Dockerfile`, **uncomment** the `hf mode` `RUN` line.
2. In **Settings → Variables and secrets**, add:
   - `LLM_PROVIDER = hf`
   - `EMBEDDING_PROVIDER = hf`
3. **Restart** the Space (Settings → Factory reboot). The first answer downloads
   the model (~1–2 min); subsequent answers are cached-fast.

To go back, re-comment the line and set the two variables to `fake`.

## Notes
- **Sleeping:** a free Space sleeps after ~48 h of inactivity and wakes in a few
  seconds on the next visit — fine for a portfolio demo.
- **No password:** the demo runs without `APP_PASSWORD` so anyone can click in.
  On the keyless path there's nothing to abuse (no key, no cost). If you enable
  `hf` mode and want to limit load, set `APP_PASSWORD` as a Variable.
- **Leaner build:** the demo installs the full `requirements.txt` for simplicity;
  `ragas`, `datasets`, `psycopg` and `pgvector` aren't used on the `fake` path and
  can be trimmed if you want a faster build.
