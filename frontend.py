# In frontend.py
import streamlit as st
import pandas as pd
import os
from rag import RAGProductRecommender
PRODUCT_DATABASE_PATH = 'products.json'
try:
    product_data = pd.read_json(PRODUCT_DATABASE_PATH)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if 'recommender' not in st.session_state:
        st.session_state.recommender = RAGProductRecommender(
            product_database=product_data.to_dict(orient='records'),
            embed_model_name='all-MiniLM-L6-v2',
            gemini_api_key=GEMINI_API_KEY
        )
    recommender = st.session_state.recommender # Get the initialized recommender

except Exception as e:
    st.error(f"Failed to load product data or initialize recommender: {e}")
    st.stop()

products = pd.read_json('products.json')
product_names = products["name"].unique().tolist()

st.title("RAG LLM Product Recommendation Demo")
selected_name = st.selectbox("Select or enter a product name", product_names)
user_query = st.text_input("Optionally describe your needs (leave empty to use product description):", value=selected_name)

top_k = st.number_input("Number of recommendations", 1, 10, 3)
recommend_mode = st.radio(
    "Recommendation Mode",
    options=["ML Content-based Only", "ML + LLM Enhanced (Gemini)"],
    index=1
)
if st.button("Get Recommendation"):
    if recommend_mode.startswith("ML Content"):

        # Assuming query_for_embedding is defined here (from user_query or selected_name's description)
        selected_product_df = recommender.product_database[recommender.product_database["name"] == selected_name]
        selected_product_description = selected_product_df.iloc[0]["description"]
        query_for_embedding = user_query if user_query and user_query != selected_name else selected_product_description

        topk_product_ids = recommender.query_topk_products(query_for_embedding, k=top_k)

        recommended_products_details = []
        recommended_names = []
        for pid in topk_product_ids:
            prod_row = recommender.product_database.loc[recommender.product_database['id'] == pid].iloc[0]
            recommended_products_details.append(prod_row.to_dict())
            recommended_names.append(prod_row['name'])

        st.subheader("ML Recommended Products (Content Similarity)")
        for prod in recommended_products_details:
            st.markdown(f"- **{prod.get('name', 'Unknown')}** | {prod.get('personalized_description', prod.get('description', ''))}")
        st.markdown("**Recommended Product Names:** " + ", ".join(recommended_names))
    else:
        # LLM Enhanced
        # Same logic as above for query_for_embedding
        selected_product_df = recommender.product_database[recommender.product_database["name"] == selected_name]
        selected_product_description = selected_product_df.iloc[0]["description"]
        query_for_embedding = user_query if user_query and user_query != selected_name else selected_product_description

        rag_result = recommender.full_rag_enhanced_recommendation(
            user_query=query_for_embedding,
            top_k=top_k
        )

        # Get recommended names from the retrieved_product_ids
        recommended_names = []
        for pid in rag_result.get("retrieved_product_ids", []):
            prod_row = recommender.product_database.loc[recommender.product_database['id'] == pid]
            if not prod_row.empty:
                recommended_names.append(prod_row.iloc[0]['name'])

        st.subheader("Augmented Product Recommendation Context")
        st.code(rag_result.get("context_summary", ""), language="markdown")
        st.subheader("LLM-generated Recommendation & Explanation")
        st.code(rag_result.get("llm_generated_response", ""), language="markdown")
        st.markdown("**Recommended Product Names:** " + ", ".join(recommended_names))
