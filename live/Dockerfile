FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV BINANCE_API_KEY=""
ENV BINANCE_API_SECRET=""

CMD ["python", "live/executor.py"]
