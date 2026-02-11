# 1. Use an official Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only requirements first (this makes building faster)
COPY requirements.txt .

# 4. Install the libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your code
COPY . .

# 6. Tell Docker which port your app uses (usually 8080 for GCP)
EXPOSE 8080

# 7. Command to start your app
# Assuming your FastAPI app is in main.py and called 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]