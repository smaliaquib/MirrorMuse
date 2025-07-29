import streamlit as st
import requests

# App configuration
st.set_page_config(page_title="MirrorMuse RAG", layout="centered")

# App title and description
st.title("🧠 MirrorMuse RAG System")
st.markdown(
    "Ask any question and get a response using "
    "**Retrieval-Augmented Generation (RAG)** powered by LLMs and vector search."
)

# Query input
query = st.text_area("Enter your query:", height=150, placeholder="E.g., How does LoRA work?")

# Generate button
if st.button("Generate Answer"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid query.")
    else:
        with st.spinner("🧠 Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/rag",  # Replace with correct IP if not localhost
                    json={"query": query},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()
                answer = result.get("answer", "⚠️ No answer returned.")

                st.success("Generated Answer:")
                st.text_area("Response:", value=answer.strip(), height=300, disabled=True)

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to get response from backend:\n`{e}`")
