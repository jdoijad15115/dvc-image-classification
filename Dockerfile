# Image Classification web app (Flask + gunicorn) serving the DVC-trained model.
# The model file comes from the DVC pipeline / S3 remote (`dvc pull`) before `docker build`.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /srv

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app/ app/
COPY models/model.pkl models/model.pkl
COPY metrics.json params.yaml ./

RUN useradd --create-home appuser && chown -R appuser /srv
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--chdir", "app", "app:app"]
