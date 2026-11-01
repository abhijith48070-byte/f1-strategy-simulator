FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# default: run the offline test-suite; override to run experiments, e.g.
#   docker run -v $PWD/data:/app/data IMAGE python run_experiment.py
CMD ["sh", "-c", "python -m tests.test_pipeline && python -m tests.test_strategy && python -m tests.test_survival && python -m tests.test_uncertainty && python -m tests.test_forecast"]
