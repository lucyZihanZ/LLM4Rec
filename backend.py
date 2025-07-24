from fastapi import FastAPI, Query
from typing import Optional
from ml_rec import ProductRecommender
from rag import RAGProductRecommender 
import pandas as pd


# Make sure your Gemini API key is set correctly.
GEMINI_API_KEY = "AIzaSyB7NlSYpIG5SxaX8fkYCdw8EFpC4_w7Ojg"
app = FastAPI()

mlrec = ProductRecommender('products.json')
products = mlrec.products_df.reset_index()  # DataFrame
ragrec = RAGProductRecommender(product_database=products, gemini_api_key=GEMINI_API_KEY)

@app.get("/rag_augmented_recommend")
def rag_augmented_recommend(
    name: str = Query(..., description="Product name, partial match supported"),
    k: int = 3,
    mode: str = Query("llm", description="'ml' for ML-only, 'llm' for ML+LLM enhanced"),
    user_query: Optional[str] = Query(None, description="Custom user query or preference (optional)")
):
    product_row = mlrec.products_df[mlrec.products_df['name'].str.lower().str.contains(name.lower())]
    print(f"[Backend] product_row: {product_row}")
    if product_row.empty:
        return {
            "error": "Product name not found.",
            "recommendations": [],
            "llm_generated_response": "",
            "context_summary": "",
            "retrieved_product_ids": [],
            "recommended_names": []
        }

    product_id = product_row.index[0]
    recommended_ids = mlrec.get_top_k_recommendations(product_id, k=k)
    ml_enhanced_products = mlrec.enhance_recommendations(recommended_ids)
    ml_recommended_names = [prod['name'] for prod in ml_enhanced_products if 'name' in prod]

    if mode == "ml":
        return {
            "mode": "ml",
            "recommendations": ml_enhanced_products,
            "llm_generated_response": "",
            "context_summary": "",
            "retrieved_product_ids": recommended_ids,
            "recommended_names": ml_recommended_names
        }
    else:
        # query_for_llm = user_query if user_query else product_row.iloc[0]['description']
        # context = ragrec.get_augmented_product_context(recommended_ids)
        # llm_response = ragrec.generate_similar_products_with_llm(context, user_query=query_for_llm)
        # recommended_names = []
        # for pid in recommended_ids:
        #     row2 = mlrec.products[products['id'] == pid]
        #     if not row2.empty:
        #         recommended_names.append(row2.iloc[0]['name'])

        # return {
        #     "mode": "llm",
        #     "recommendations": mlrec.enhance_recommendations(recommended_ids),
        #     "llm_generated_response": llm_response,
        #     "context_summary": context,
        #     "retrieved_product_ids": recommended_ids,
        #     "recommended_names": recommended_names
        # }
            # else:
        print("Available product names:", mlrec.products_df['name'].tolist())
        print("==== [Backend] LLM模式被触发 ====")
        query_for_llm = user_query if user_query else product_row.iloc[0]['description']
        print(f"[Backend] user_query: {user_query}")
        print(f"[Backend] query_for_llm: {query_for_llm}")

        context = ragrec.get_augmented_product_context(recommended_ids)
        print(f"[Backend] context: {context}")

        try:
            llm_response = ragrec.generate_similar_products_with_llm(context, user_query=query_for_llm)
            print(f"[Backend] llm_response: {llm_response}")
        except Exception as e:
            print(f"[Backend][ERROR] Gemini LLM接口调用失败: {e}")
            llm_response = f"LLM调用失败: {e}"

        recommended_names = []
        for pid in recommended_ids:
            row2 = mlrec.products_df[mlrec.products_df['id'] == pid]  # 这里应该用 products_df
            if not row2.empty:
                recommended_names.append(row2.iloc[0]['name'])

        print(f"[Backend] recommended_names: {recommended_names}")

        return {
            "mode": "llm",
            "recommendations": mlrec.enhance_recommendations(recommended_ids),
            "llm_generated_response": llm_response,
            "context_summary": context,
            "retrieved_product_ids": recommended_ids,
            "recommended_names": recommended_names
        }

if __name__ == "__main__":
    import requests

    url = "http://127.0.0.1:8000/rag_augmented_recommend"
    params = {
        "name": "tea",  # 可以改为你 products.json 中的任何产品关键词
        "k": 2,
        "mode": "llm",  # 改为 "ml" 测试纯内容推荐
        "user_query": "I want a relaxing drink with lavender and chamomile"
    }

    response = requests.get(url, params=params)
    print(response.json())
