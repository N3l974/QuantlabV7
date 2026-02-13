FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY engine ./engine
COPY strategies ./strategies
COPY live ./live
COPY data ./data
COPY config ./config
COPY main.py ./main.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "live.run_portfolio", "--config", "config/live/portfolios/v5c-highrisk-paper.json"]
