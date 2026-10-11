# Container for the AQ-GNoME screening app. Build from the repository root:
#   docker build -f app/Dockerfile -t aq-gnome-app .
# The data folder is NOT baked in — mount it at /data (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    AQ_GNOME_DATA_DIR=/data \
    AQ_GNOME_MAX_VARIANTS=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /opt/aq-gnome

# Install the package first so the (slow) dependency layer is cached across app edits.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[app]"
COPY app ./app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
