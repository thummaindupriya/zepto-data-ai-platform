FROM python:3.13-slim

WORKDIR /app

COPY support_assistant/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY support_assistant /app/support_assistant

WORKDIR /app/support_assistant

RUN python ingest.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
