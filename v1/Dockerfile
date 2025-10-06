# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose ports
EXPOSE 8000 8501

# Start FastAPI and Streamlit together
CMD uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run dashboard.py --server.port 8501 --server.headless true --server.enableCORS false
