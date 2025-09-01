# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for course materials. The system enables users to query course content using semantic search and AI-powered responses through a web interface.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Set up environment variables (required)
# Create .env with: ANTHROPIC_API_KEY=your_api_key_here
```

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uvicorn app:app --reload --port 8000

# Alternative if uv has issues (as noted in run.sh)
cd backend && uvicorn app:app --reload --port 8000
```

### Development Access
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

### Core RAG Pipeline
The system follows a multi-layered architecture with clear separation of concerns:

1. **FastAPI Layer** (`backend/app.py`): HTTP endpoints and request handling
2. **RAG System Orchestrator** (`backend/rag_system.py`): Main coordination logic
3. **AI Generator** (`backend/ai_generator.py`): Two-stage Claude API interaction
4. **Document Processing** (`backend/document_processor.py`): Course content parsing and chunking
5. **Vector Storage** (`backend/vector_store.py`): ChromaDB operations and semantic search
6. **Search Tools** (`backend/search_tools.py`): Tool-based search system for AI
7. **Session Management** (`backend/session_manager.py`): Conversation context

### Key Architectural Patterns

**Dependency Injection**: RAGSystem orchestrates all components, injecting configuration and dependencies.

**Tool-based AI Interaction**: Claude decides autonomously when to search using the CourseSearchTool. Two-stage process: initial tool planning → final synthesis.

**Document Structure**: Course documents in `docs/` follow a specific format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [url]
[lesson content...]
```

**Chunking Strategy**: Sentence-aware chunking with configurable overlap (800 chars, 100 overlap). Context enrichment with lesson prefixes for better semantic search.

### Data Flow
1. Frontend POST to `/api/query`
2. FastAPI → RAG System → AI Generator
3. Claude with tools → Search execution (if needed) → Response synthesis
4. Session update → Response with sources

## Key Configuration

- **Python**: 3.13+ required
- **Vector DB**: ChromaDB with sentence-transformers embeddings
- **AI Model**: Claude Sonnet 4 (configurable in `backend/config.py`)
- **Chunk Settings**: 800 chars with 100 char overlap
- **Frontend**: Served as static files from `/frontend`

## Development Notes

### Document Processing
Course documents are automatically loaded from `docs/` on startup. The system expects structured course scripts with lesson markers and metadata headers.

### Session Management
The system maintains conversation context with configurable history length (default: 2 exchanges).

### Vector Store
ChromaDB persistence in `./chroma_db`. The system avoids re-processing existing courses by checking titles.

### Dependencies Note
The run.sh script mentions PyTorch installation issues with uv, falling back to conda/pip for some packages. This may affect dependency management workflows.