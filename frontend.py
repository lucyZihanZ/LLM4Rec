import streamlit as st
import pandas as pd
import requests

# Load products for display (front-end only, not used for actual recommendation)
products = pd.read_json('products.json')
product_names = products["name"].unique().tolist()

st.title("RAG LLM Product Recommendation Demo")

# 1. Product name selection
selected_name = st.selectbox("Select or enter a product name", product_names)
user_query = st.text_input("Optionally describe your needs (leave empty to use product description):", value=selected_name)
top_k = st.number_input("Number of recommendations", 1, 10, 3)

# 2. Recommendation mode selector
recommend_mode = st.radio(
    "Recommendation Mode",
    options=["ML Content-based Only", "ML + LLM Enhanced (Gemini)"],
    index=1
)

if st.button("Get Recommendation"):
    params = {
        "name": selected_name,
        "k": top_k,
        "mode": "ml" if recommend_mode.startswith("ML Content") else "llm"
    }
    # Only add user_query if user changed it (not just product name)
    if user_query and user_query != selected_name:
        params["user_query"] = user_query

    try:
        resp = requests.get("http://localhost:8000/rag_augmented_recommend", params=params, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        st.error(f"Failed to contact backend: {e}")
        st.stop()

    if result.get("error"):
        st.error(result["error"])
    elif params["mode"] == "ml":
        st.subheader("ML Recommended Products (Content Similarity)")
        for prod in result.get("recommendations", []):
            st.markdown(f"- **{prod.get('name', 'Unknown')}** | {prod.get('personalized_description', prod.get('description', ''))}")
        st.markdown("**Recommended Product Names:** " + ", ".join(result.get("recommended_names", [])))
    else:
        st.subheader("Augmented Product Recommendation Context")
        st.code(result.get("context_summary", ""), language="markdown")
        st.subheader("LLM-generated Recommendation & Explanation")
        st.code(result.get("llm_generated_response", ""), language="markdown")
        st.markdown("**Recommended Product Names:** " + ", ".join(result.get("recommended_names", [])))


