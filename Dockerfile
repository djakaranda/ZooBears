FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY artifacts/cpu_baseline.joblib ./artifacts/cpu_baseline.joblib
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["python", "-m", "compactfacts.api", "--host", "0.0.0.0"]
