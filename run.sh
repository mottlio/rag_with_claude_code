#!/bin/bash

# Create necessary directories
mkdir -p docs 

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Starting Course Materials RAG System..."
echo "Make sure you have set your ANTHROPIC_API_KEY in .env"

# Change to backend directory and start the server
cd backend && uvicorn app:app --reload --port 8000

# I could not use uv here - because it was impossible to install pytorch with uv - so I
# just used conda, installed all dependencies with requirements.txt and installed a number of packages wusing pip: anthropic, chromadb, dotenv, fastapi, 