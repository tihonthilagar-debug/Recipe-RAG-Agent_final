# Recipe RAG Agent - Project Report

## Problem Statement
Problem Statement No. 8: Document Q&A RAG Agent for Recipe Generator.

## Proposed Solution
A RAG-based recipe assistant that ingests recipe documents, stores searchable embeddings in Chroma, retrieves relevant recipe context, and uses Gemini to generate personalized cooking guidance.

## Requirements Covered
- Recipe PDF/TXT/Markdown ingestion
- Searchable knowledge base
- Natural-language Q&A
- Available-ingredient personalization
- Dietary restriction personalization
- Cooking-time and cuisine preferences
- Step-by-step instructions
- Ingredient substitutions
- Estimated nutrition
- Shopping list
- Streamlit visualization/interface

## Technology
- Python
- Streamlit
- Google Gemini API
- Gemini Embedding model
- Chroma vector database
- PyPDF
- python-dotenv

## Agentic Design
The application follows sequential roles:
1. Recipe Retrieval Agent
2. Personalization Agent
3. Recipe Generation Agent
4. Nutrition & Shopping List Agent

## Safety
The app states that nutrition values are estimates and that users should verify allergens and medical dietary needs.

## Future Scope
- Voice and image input
- More recipe sources
- User accounts and meal history
- Nutrition database integration
- Multilingual support
- Production deployment
