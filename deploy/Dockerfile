FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser engine ./engine
COPY --chown=appuser:appuser strategies ./strategies
COPY --chown=appuser:appuser live ./live
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser main.py ./main.py

RUN mkdir -p /app/logs && chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1

USER appuser

CMD ["python", "-m", "live.run_portfolio", "--config", "config/live/portfolios/v5c-highrisk-paper.json"]
