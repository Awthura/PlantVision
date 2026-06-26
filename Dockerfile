FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml .
RUN uv sync --no-dev

COPY configs/ configs/
COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/

# data/ is volume-mounted at runtime — see docker run example in README
VOLUME ["/app/data", "/app/outputs", "/app/models"]

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/demo.py", "--server.address=0.0.0.0"]
