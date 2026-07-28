# Free-tier-friendly container: no GPU, small base image, single process.
# Works as-is on Hugging Face Spaces (Docker SDK), Fly.io's free allowance,
# Railway's free tier, or just `docker run` locally.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Registry, feature store, and data directories need to be writable at runtime.
RUN mkdir -p registry feature_store_db data/dead_letter_queue

EXPOSE 8000

# Expects registry/production.json to already exist (bake it into the image at
# build time by running scripts/run_pipeline.sh before `docker build`, or mount
# a volume with a pre-populated registry/ directory).
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
