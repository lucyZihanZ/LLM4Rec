from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import pandas as pd
from prompt import prompt
class RAGProductRecommender:
    def __init__(
        self,
        product_database=None,          # 支持DataFrame或list[dict]，可为空
        embed_model_name='all-MiniLM-L6-v2',
        llm_model_name='mistralai/Mistral-7B-Instruct-v0.1',
        hf_token=api-key
    ):
        if hf_token:
            login(token=hf_token)
        self.embedder = SentenceTransformer(embed_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.product_database = None
        self.product_embeddings = None
        if product_database is not None:
            self.set_product_database(product_database)

    # def set_product_database(self, product_database):
    #     self.product_database = product_database
    #     self.product_embeddings = self.embedder.encode(product_database['description'], convert_to_tensor=True) 
    def set_product_database(self, product_database):
        if isinstance(product_database, list):
            product_database = pd.DataFrame(product_database)
        self.product_database = product_database.reset_index(drop=True)
        self.product_embeddings = self.embedder.encode(
            self.product_database['description'].tolist(), convert_to_tensor=True
        )

    def get_augmented_product_context(self, product_ids, user_preferences=None):
        context_string = "Here are some product recommendations with their details:\n\n"
        for i, pid in enumerate(product_ids, 1):
            row = self.product_database.loc[self.product_database['id'] == pid]
            if row.empty:
                context_string += f"Product ID {pid}: Details not found.\n"
                continue
            row = row.iloc[0]
            context_string += (
            "--- Product {i}: {name} ---\n"
            "Category: {category}\n"
            "Brand: {brand}\n"
            "Price: ${price:.2f}\n"
            "Rating: {rating} stars ({reviews_count} reviews)\n"
            "{on_sale_status}"
            "Description: {desc}\n"
            "Image URL: {image_url}\n\n").format(
            i=i,
            name=row['name'],
            category=row.get('category', 'N/A'),
            brand=row.get('brand', 'N/A'),
            price=row.get('price', 0),
            rating=row.get('rating', 'N/A'),
            reviews_count=row.get('reviews_count', 'N/A'),
            on_sale_status="Status: ON SALE!\n" if row.get('on_sale', False) else "",
            desc=row.get('personalized_description', row['description']),
            image_url=row.get('image_url', '')
        )
        return context_string.strip()

    def generate_similar_products_with_llm(self, context_string, user_query, max_new_tokens=256):
        prompt = prompt.format(context = context_string, query = user_query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        outputs = self.llm_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded

    def query_topk_products(self, query, k=1):
        assert self.product_database is not None, "You must set product_database first!"
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.product_embeddings)[0]
        top_results = torch.topk(cos_scores, k=k)
        topk_indices = top_results.indices
        product_ids = [int(self.product_database.iloc[int(i)]['id']) for i in topk_indices]
        return product_ids

    def full_rag_enhanced_recommendation(self, user_query, user_preferences=None, top_k=3):
        assert self.product_database is not None, "You must set product_database first!"
        topk_product_ids = self.query_topk_products(user_query, k=top_k)
        context = self.get_augmented_product_context(topk_product_ids, user_preferences)
        llm_response = self.generate_similar_products_with_llm(context, user_query)
        return {
            "context_summary": context,
            "llm_generated_response": llm_response,
            "retrieved_product_ids": topk_product_ids
        }

# ============ 用法示例 =============
if __name__ == "__main__":
    # 假如你有一个 list[dict] 或 DataFrame products
    products = [
        {
            "id": 1,
            "name": "Energy Booster Coffee",
            "description": "A rich, dark roast coffee blend with natural ginseng.",
            "category": "Beverage",
            "brand": "CoffeeBrand",
            "price": 9.99,
            "rating": 4.5,
            "reviews_count": 120,
            "on_sale": True,
            "personalized_description": "Rich in ginseng, perfect for energy.",
            "image_url": "http://example.com/1.jpg"
        },
        {
            "id": 2,
            "name": "Protein Power Bar",
            "description": "A high-protein snack bar made with nuts and whey.",
            "category": "Snack",
            "brand": "BarBrand",
            "price": 2.49,
            "rating": 4.8,
            "reviews_count": 98,
            "on_sale": False,
            "personalized_description": "Great for muscle recovery.",
            "image_url": "http://example.com/2.jpg"
        },
    ]

    recommender = RAGProductRecommender(
        embed_model_name='all-MiniLM-L6-v2',
        llm_model_name='mistralai/Mistral-7B-Instruct-v0.1',
        hf_token="api-key"
    )
    # 传入你的数据
    recommender.set_product_database(products)

    query = "energy and protein snack"
    result = recommender.full_rag_enhanced_recommendation(query)
    print("======== Retrieved Products ========")
    print(result["context_summary"])
    print("======== LLM Generated Recommendation ========")
    print(result["llm_generated_response"])
