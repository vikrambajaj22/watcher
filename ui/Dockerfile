FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

# Copy the Streamlit app
COPY streamlit_app.py .

# Copy static assets
COPY static ./static

# Expose the default port
EXPOSE 8080

# Use a shell entrypoint to set PORT dynamically
CMD ["sh", "-c", "\
    : ${PORT:=8080} && \
    export STREAMLIT_SERVER_PORT=$PORT && \
    export STREAMLIT_SERVER_ADDRESS=0.0.0.0 && \
    export STREAMLIT_SERVER_HEADLESS=true && \
    export IMAGE_DIR='./static/images' && \
    streamlit run streamlit_app.py \
"]