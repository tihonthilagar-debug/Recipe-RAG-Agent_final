# Recipe RAG Agent — No Ollama

This is a fully functional Document Q&A RAG Recipe Generator using the Gemini API.

## Why no Ollama?
The project uses Google's cloud Gemini API for generation and embeddings. Therefore Ollama, local LLM downloads, and local model hosting are NOT required.

## Requirements
- Python 3.9+
- Internet connection
- A Gemini API key
- No Ollama

Google's official Gemini documentation requires an API key and provides the Python `google-genai` SDK. The project uses the same approach.

## Setup on Windows

1. Open this folder in VS Code.
2. Open Terminal.
3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Copy `.env.example` to `.env`.
5. Put your Gemini API key in `.env`:

```text
GEMINI_API_KEY=your_key_here
```

6. Start the app:

```powershell
python -m streamlit run app.py
```

7. Open `http://localhost:8501`.

## Features
- RAG over recipe documents
- PDF/TXT/MD upload
- Chroma vector database
- Natural-language recipe Q&A
- Cuisine preference
- Dietary restrictions
- Available ingredients
- Cooking-time limit
- Step-by-step instructions
- Substitutions
- Estimated nutrition
- Shopping list
- Gemini-powered personalized generation

## GitHub
Create a public repository and upload this folder. Do not commit `.env` or your API key.
