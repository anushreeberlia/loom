FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# flash_attn requires CUDA to compile but Florence-2 imports it at module level.
# On CPU it falls back to standard attention, so a stub package satisfies the import check.
RUN mkdir -p /usr/local/lib/python3.11/site-packages/flash_attn && \
    echo "" > /usr/local/lib/python3.11/site-packages/flash_attn/__init__.py

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
