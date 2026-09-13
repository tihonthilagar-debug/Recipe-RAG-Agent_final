import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RecipeRAG

load_dotenv()

st.set_page_config(page_title="Recipe RAG Agent", page_icon="🍳", layout="wide")

st.title("🍳 Recipe RAG Agent")
st.caption("Document Q&A + personalized recipe generation using Gemini API and a local Chroma vector store.")

if not os.getenv("GEMINI_API_KEY"):
    st.warning("GEMINI_API_KEY is not set. Create a Gemini API key and put it in a .env file. See README.md.")
    st.stop()

@st.cache_resource
def get_engine():
    return RecipeRAG()

engine = get_engine()

with st.sidebar:
    st.header("Personalization")
    cuisine = st.selectbox("Cuisine preference", ["Any", "Indian", "Italian", "Mexican", "Asian", "Mediterranean"])
    diet = st.multiselect("Dietary restrictions", ["Vegetarian", "Vegan", "Gluten-free", "Dairy-free", "Nut-free", "Sugar-free"])
    max_time = st.slider("Maximum cooking time (minutes)", 10, 180, 45)
    ingredients = st.text_input("Available ingredients", placeholder="e.g. chicken, rice, tomato")
    st.divider()
    st.subheader("Recipe documents")
    uploads = st.file_uploader("Upload PDF/TXT/MD recipe files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploads:
        count = engine.add_uploaded_files(uploads)
        st.success(f"Indexed {count} new document(s).")

query = st.text_area(
    "Ask a recipe question",
    placeholder="How can I make a high-protein chicken rice bowl in 30 minutes?",
    height=110,
)

col1, col2 = st.columns([1, 1])
with col1:
    generate = st.button("Generate Personalized Recipe", type="primary", use_container_width=True)
with col2:
    reset = st.button("Reset conversation", use_container_width=True)

if reset:
    st.session_state.pop("last_result", None)
    st.rerun()

if generate:
    if not query.strip():
        st.error("Enter a recipe question first.")
    else:
        with st.spinner("Retrieving recipes and generating your personalized result..."):
            try:
                result = engine.answer(
                    query=query,
                    cuisine=cuisine,
                    dietary_restrictions=diet,
                    max_time=max_time,
                    available_ingredients=ingredients,
                )
                st.session_state["last_result"] = result
            except RuntimeError as exc:
                st.error(str(exc))

result = st.session_state.get("last_result")
if result:
    st.markdown("## Personalized Recipe")
    st.markdown(result["answer"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Retrieved recipe sources")
        for src in result["sources"]:
            st.write(f"- {src}")
    with c2:
        st.markdown("### Retrieval details")
        st.write(f"Retrieved chunks: {result['retrieved_chunks']}")
        st.write("Vector store: Chroma")
        st.write("Embedding model: Gemini Embedding")

st.divider()
st.caption("Educational cooking assistant. Verify allergens and nutritional information before relying on them for medical or dietary decisions.")
