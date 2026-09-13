# Langflow Workflow

The submission workflow is represented as:

Chat Input
  -> Query/Preference Parser
  -> Recipe Retrieval Agent
  -> Personalization Agent
  -> Recipe Generation Agent
  -> Nutrition & Shopping List Agent
  -> Chat Output

Knowledge path:
Recipe Files -> File Loader -> Text Splitter -> Embedding Model -> Vector Store (Chroma)

Suggested Langflow components:
1. Chat Input
2. File
3. Text Splitter
4. Embedding / Google Generative AI Embeddings
5. Chroma Vector Store
6. Retriever
7. Prompt
8. Google Generative AI / Gemini model
9. Parser
10. Chat Output

The supplied Streamlit implementation provides the same functional pipeline in Python.
