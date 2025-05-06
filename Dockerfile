FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Upgrade pip to avoid compatibility issues
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app code
COPY . .

# Create volume (if your app uses persistent data)
VOLUME /app/data

# Expose both Streamlit ports
EXPOSE 8501 8502

# Create a shell script to run both apps
RUN echo '#!/bin/bash\n\
streamlit run auth_app.py --server.port=8501 &\n\
streamlit run machine_learning.py --server.port=8502' > /app/run.sh

# Make the script executable
RUN chmod +x /app/run.sh

# Set default command to run the shell script
CMD ["/bin/bash", "/app/run.sh"]