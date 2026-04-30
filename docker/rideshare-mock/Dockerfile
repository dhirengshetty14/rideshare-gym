FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir uvicorn[standard]

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=2s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request as r; r.urlopen('http://localhost:8000/health').read()"

CMD ["uvicorn", "rideshare_gym.mock_server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
