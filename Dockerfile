# 1. Base image with Python (NO Chrome - using Selenium Grid sidecar instead)
FROM python:3.11-slim-bookworm

# 2. Install minimal dependencies only
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code

COPY agents/ ./agents/
COPY agentverse/ ./agentverse/
COPY langgraph_logic/ ./langgraph_logic/
COPY models/ ./models/
COPY prompts/ ./prompts/
COPY data_for_monitor/ ./data_for_monitor/
COPY test_func/ ./test_func/

# 6. Expose ports
EXPOSE 8000 8005 8006 8008

# 7. Set environment variables
ENV PYTHONUNBUFFERED=1
# SELENIUM_REMOTE_URL will be set by docker-compose to point to selenium container

# 8. Entry point
CMD ["python", "-m", "agentverse.supplier_orchestrator"]