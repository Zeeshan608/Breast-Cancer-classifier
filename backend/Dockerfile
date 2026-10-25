FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

# Render sets PORT env var; default to 10000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
