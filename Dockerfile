# Single-container image (e.g. Hugging Face Spaces, Docker SDK, port 7860).
# Runs the FastAPI service on 8000 inside the container and the Streamlit dashboard on 7860,
# with the dashboard calling the API through localhost.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app ./app
COPY models ./models
COPY docs ./docs
COPY monitoring ./monitoring
COPY data/app ./data/app
COPY data/monitoring/monitoring_summary.json ./data/monitoring/monitoring_summary.json
ENV API_URL=http://127.0.0.1:8000 PREDICTION_LOG=/tmp/prediction_log.jsonl PYTHONUNBUFFERED=1
EXPOSE 7860
CMD uvicorn app.api.main:app --host 127.0.0.1 --port 8000 & \
    streamlit run app/dashboard/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
