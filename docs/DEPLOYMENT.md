# Deployment guide

## 1. Run locally (Windows PowerShell, from the project root)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python scripts\run_notebooks.py          # full pipeline, about 10 minutes (Spark step needs Java 8+)
.\.venv\Scripts\python -m uvicorn app.api.main:app --port 8000      # API  -> http://127.0.0.1:8000/docs
.\.venv\Scripts\python -m streamlit run app\dashboard\app.py         # dashboard -> http://localhost:8501
.\.venv\Scripts\python -m pytest -q tests
```

The committed artefacts (`models/`, `data/app/`, `data/monitoring/monitoring_summary.json`, `monitoring/`) are enough
to run the API and the dashboard **without** re-running the pipeline.

## 2. Publish the code on GitHub

```powershell
git init
git add .
git commit -m "Learner segmentation capstone"
git branch -M main
git remote add origin https://github.com/hst-nxk/BA_project.git
git push -u origin main
```
`.gitignore` excludes the raw OULAD files (454 MB) and regenerable intermediate data. `scripts/download_oulad.py`
re-downloads OULAD.

## 3. Cloud deployment (free tier)

### Option A — Render, API + dashboard (recommended, one blueprint)
1. Sign in at https://render.com with GitHub → **New → Blueprint** → choose the repository. `render.yaml` creates:
   * `learner-segmentation-api` (FastAPI, health check `/health`)
   * `learner-segmentation-dashboard` (Streamlit)
2. When the API is live, copy its URL (e.g. `https://learner-segmentation-api.onrender.com`). Paste it into the
   dashboard service's environment variable `API_URL`, then redeploy the dashboard.
3. Check: `<api-url>/docs`, `<api-url>/health`, and in the dashboard the **Assign a new learner** tab should report
   "source: API …".

Free instances sleep after about 15 minutes idle. Open the API URL once a minute before a live demo. The dashboard
never breaks while the API wakes up, because it falls back to the in-process model.

### Option B — Streamlit Community Cloud (dashboard) + Render (API)
1. https://share.streamlit.io → **New app** → repository, branch `main`, main file `app/dashboard/app.py`.
2. **Advanced settings → Secrets**: `API_URL = "https://learner-segmentation-api.onrender.com"`.
3. Deploy the API on Render as in option A (or **New → Web Service** with the start command from `render.yaml`).

### Option C — Hugging Face Spaces (single container)
1. https://huggingface.co/new-space → SDK **Docker** → push this repository to the Space.
2. The `Dockerfile` starts the API on port 8000 inside the container and the dashboard on port 7860 (the public port).
   The dashboard reaches the API through `http://127.0.0.1:8000`.

## 4. Links (fill in after deployment)

| Component | URL |
|---|---|
| GitHub repository | https://github.com/hst-nxk/BA_project |
| API (Swagger docs) | https://learner-segmentation-api.onrender.com/docs |
| Dashboard | https://learner-segmentation-dashboard.onrender.com |
