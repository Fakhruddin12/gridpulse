FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .[api]
COPY configs ./configs
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn", "gridpulse.api:app", "--host", "0.0.0.0", "--port", "8000"]
