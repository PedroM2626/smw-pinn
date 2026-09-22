# CPU base; for CUDA use --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
FROM python:3.10-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42

COPY requirements.txt pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt --extra-index-url "$TORCH_INDEX_URL" && \
    pip install -e . --no-deps

# ROM, datasets and checkpoints are NOT baked in (legal + size).
# Mount them at runtime: -v ./data:/app/data -v ./results:/app/results
CMD ["python", "-m", "pytest", "tests/", "-q"]
