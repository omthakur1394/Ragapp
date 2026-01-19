# Use Python 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create a writable directory for the database (Hugging Face specific fix)
RUN mkdir -p /app/project/chroma_db && chmod -R 777 /app/project/chroma_db

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Run the app
# Note: We point to 'project.api:app' and listen on port 7860
CMD ["uvicorn", "project.api:app", "--host", "0.0.0.0", "--port", "7860"]