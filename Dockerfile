FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements.txt requirements-embeddings.txt ./
# Set INSTALL_EMBEDDINGS=true to include sentence-transformers (adds PyTorch, ~1 GB).
ARG INSTALL_EMBEDDINGS=false
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_EMBEDDINGS" = "true" ]; then \
         pip install --no-cache-dir -r requirements-embeddings.txt \
           --extra-index-url https://download.pytorch.org/whl/cpu; \
       fi

COPY app ./app
COPY static ./static

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
