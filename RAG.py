from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import google.generativeai as genai
from prompt import prompt as PROMPT_TEMPLATE
import os

def api_key():
    """
    Return your api key as a string.
    """
    key = "api-key"
    return key

class RAGProductRecommender:
    def __init__(
        self,
        product_database=None,
        embed_model_name='all-MiniLM-L6-v2',
        gemini_api_key=api_key()
    ):
        # 初始化嵌入模型
        self.embedder = SentenceTransformer(embed_model_name)
        self.product_database = None
        self.product_embeddings = None
        if product_database is not None:
            self.set_product_database(product_database)
        # 初始化 Gemini
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")  # 或"gemini-1.5-pro"
        else:
            self.gemini_model = None

    def set_product_database(self, product_database):
        # Convert list of dicts to DataFrame if needed, then reset index
        if isinstance(product_database, list):
            product_database = pd.DataFrame(product_database)
        self.product_database = product_database.reset_index(drop=True)

        # Encode product descriptions to create embeddings
        if 'description' not in self.product_database.columns:
            raise ValueError("Product database must contain a 'description' column for embedding.")
        
        product_descriptions = self.product_database['description'].tolist()
        if not product_descriptions: # Handle empty list case
            self.product_embeddings = torch.tensor([])
            print("Warning: No product descriptions found to embed.")
        else:
            self.product_embeddings = self.embedder.encode(
                product_descriptions, convert_to_tensor=True
            )
            print(f"Product database set with {len(self.product_database)} products and embeddings generated.")


    def get_augmented_product_context(self, product_ids: list, user_preferences: dict = None) -> str:
        """
        Retrieves and formats product details into a string for LLM context.
        """
        context_string = "Here are some product recommendations with their details:\n\n"
        if self.product_database is None or self.product_database.empty:
            return "No product database available to fetch details."

        for i, pid in enumerate(product_ids, 1):
            row = self.product_database.loc[self.product_database['id'] == pid]
            if row.empty:
                context_string += f"Product ID {pid}: Details not found.\n"
                continue
            row = row.iloc[0]
            # Safely get values or provide a fallback
            effects = ", ".join(row.get('effects', [])) if isinstance(row.get('effects', []), list) else str(row.get('effects', 'N/A'))
            ingredients = ", ".join(row.get('ingredients', [])) if isinstance(row.get('ingredients', []), list) else str(row.get('ingredients', 'N/A'))
            sales_data = row.get('sales_data', {})
            units_sold = sales_data.get('units_sold', 'N/A') if isinstance(sales_data, dict) else 'N/A'
            last_month_revenue = sales_data.get('last_month_revenue', 'N/A') if isinstance(sales_data, dict) else 'N/A'

            context_string += (
                f"--- Product {i}: {row['name']} ---\n"
                f"Type: {row.get('type', 'N/A')}\n"
                f"Description: {row.get('description', 'N/A')}\n"
                f"Effects: {effects}\n"
                f"Ingredients: {ingredients}\n"
                f"Price: ${row.get('price', 0):.2f}\n"
                f"Units Sold: {units_sold}\n"
                f"Last Month Revenue: ${last_month_revenue}\n"
                "\n"
            )
        return context_string.strip()

    # def generate_similar_products_with_llm(self, context_string: str, user_query: str, max_new_tokens: int = 32) -> str:
    #     """
    #     Generates product recommendations using the loaded LLM.
    #     """
    #     if self.llm_model is None or self.tokenizer is None:
    #         return "LLM model or tokenizer not loaded. Cannot generate recommendations."

    #     # IMPORTANT FIX: Use the imported PROMPT_TEMPLATE and then format it
    #     formatted_prompt = PROMPT_TEMPLATE.format(context=context_string, query=user_query)

    #     inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.llm_model.device)
    #     outputs = self.llm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    #     decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    #     # Safely remove the prompt part. `startswith` is robust.
    #     if decoded.startswith(formatted_prompt):
    #         return decoded[len(formatted_prompt):].strip()
    #     else:
    #         # Fallback if the model doesn't perfectly reproduce the prompt (e.g., if max_new_tokens is too low)
    #         print("Warning: LLM output did not start with the prompt. Returning full decoded output.")
    #         return decoded.strip()
    def generate_similar_products_with_llm(self, context_string, user_query, max_new_tokens=256):
        prompt = PROMPT_TEMPLATE.format(context=context_string, query=user_query)
        if self.gemini_model is None:
            return "Gemini model not configured. Please provide your Gemini API key."
        response = self.gemini_model.generate_content(prompt)
        return response.text.strip()

    def query_topk_products(self, query: str, k: int = 3) -> list:
        """
        Queries the product database for top-k similar products based on embedding similarity.
        Returns a list of product IDs.
        """
        if self.product_database is None or self.product_embeddings is None:
            print("Error: Product database or embeddings not set. Cannot perform query.")
            return []

        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        
        # Ensure product_embeddings is not empty
        if self.product_embeddings.numel() == 0:
            print("WARNING: Product embeddings are empty. No products to search.")
            return []

        cos_scores = util.pytorch_cos_sim(query_embedding, self.product_embeddings)[0]
        num_candidates = cos_scores.shape[0]

        if num_candidates == 0:
            print("WARNING: No candidates found for query:", query)
            return []
            
        # Adjust k if it's greater than available candidates
        actual_k = min(k, num_candidates)
        
        # If actual_k is 0 (e.g., if num_candidates was 0), return empty
        if actual_k == 0:
             return []

        top_results = torch.topk(cos_scores, k=actual_k)
        topk_indices = top_results.indices

        # Get product 'id's based on indices. Ensure 'id' column exists.
        # Use .iloc directly on the DataFrame for row access
        product_ids = [self.product_database.iloc[int(idx)]['id'] for idx in topk_indices]
        return product_ids

    def full_rag_enhanced_recommendation(self, user_query: str, user_preferences: dict = None, top_k: int = 3) -> dict:
        """
        Executes the full RAG-enhanced recommendation pipeline.
        """
        if self.product_database is None or self.product_embeddings is None:
            return {
                "context_summary": "Error: Product database not loaded.",
                "llm_generated_response": "Error: Cannot generate recommendations without product data.",
                "retrieved_product_ids": []
            }

        print(f"\n--- Initiating recommendation for query: '{user_query}' ---")
        
        # Step 1: Retrieve top-k similar products
        topk_product_ids = self.query_topk_products(user_query, k=top_k)
        print(f"Retrieved top-{top_k} product IDs: {topk_product_ids}")

        # Step 2: Augment context with detailed product information
        context = self.get_augmented_product_context(topk_product_ids, user_preferences)
        print("\n--- Generated Context Summary for LLM ---")
        print(context)

        # Step 3: Generate LLM response using the augmented context
        llm_response = self.generate_similar_products_with_llm(context, user_query)
        print("\n--- LLM Generated Recommendation ---")
        print(llm_response)

        return {
            "context_summary": context,
            "llm_generated_response": llm_response,
            "retrieved_product_ids": topk_product_ids
        }

# --- Main execution block for testing ---
if __name__ == "__main__":
    import json
    with open("products.json", "r") as f:
        products = json.load(f)
    gemini_api_key = "AIzaSyB7NlSYpIG5SxaX8fkYCdw8EFpC4_w7Ojg"
    recommender = RAGProductRecommender(
        product_database=products,
        gemini_api_key=gemini_api_key
    )
    query = "healthy energy drink with ginseng"
    result = recommender.full_rag_enhanced_recommendation(query)
    print("======== Retrieved Products ========")
    print(result["context_summary"])
    print("======== Gemini LLM Generated Recommendation ========")
    print(result["llm_generated_response"])