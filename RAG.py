from langchain_core.prompts import PromptTemplate
import torch # Required for HuggingFacePipeline model loading
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from ml_rec import ProductRecommender
from prepare import load_product_database, add_combined_features
from prompt import prompt
from huggingface_hub import login
import sys

llm_api_key = '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'
login(token = llm_api_key)
# embedding process
embedder = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', torch_dtype=torch.float16, device_map = 'auto')

product_db = load_product_database('products.json')
product_database = add_combined_features(product_db)
product_descriptions = product_database['description'].tolist()
product_embeddings = embedder.encode(product_descriptions, convert_to_tensor=True)


def generate_similar_products_with_llm(context_string: str, user_query: str, max_new_tokens: int = 256) -> str:
    """
    Generates product recommendations using a free open-source LLM (Mistral).
    """
    prompt = prompt.format(context = context_string, query = user_query)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()
def get_augmented_product_context(product_ids: list, user_preferences: dict = None) -> str:
    """
    Retrieves and augments product details for a list of product IDs,
    then formats them into a single string suitable as context for an LLM.
    """
    augmented_products = ProductRecommender.enhance_recommendations(product_ids, user_preferences)
    
    context_string = "Here are some product recommendations with their details:\n\n"
    for i, product in enumerate(augmented_products):
        if product.get("status") == "details_not_found":
            context_string += f"Product ID {product['product_id']}: Details not found.\n"
            continue

        context_string += f"--- Product {i+1}: {product['name']} ---\n"
        context_string += f"Category: {product['category']}\n"
        context_string += f"Brand: {product['brand']}\n"
        context_string += f"Price: ${product['price']:.2f}\n"
        context_string += f"Rating: {product['rating']} stars ({product['reviews_count']} reviews)\n"
        if product['on_sale']:
            context_string += "Status: ON SALE!\n"
        context_string += f"Description: {product['personalized_description']}\n"
        context_string += f"Image URL: {product['image_url']}\n"
        context_string += "\n"
    return context_string.strip()

def full_rag_enhanced_recommendation_pipeline(recommended_ids: list, user_query: str, user_preferences: dict = None):
    context = get_augmented_product_context(recommended_ids, user_preferences)
    llm_response = generate_similar_products_with_llm(context, user_query)

    return {
        "context_summary": context,
        "llm_generated_response": llm_response
    }


query = "a healthy energy drink with ginseng"
query_embedding = embedder.encode(query, convert_to_tensor=True)
cos_scores = util.pytorch_cos_sim(query_embedding, product_embeddings)[0]
top_results = torch.topk(cos_scores, k=3)
print(top_results)
topk_indices = top_results.indices  # 这是个 tensor([0, 1, 2], device=...)
product_names = [product_database.iloc[int(i)]['name'] for i in topk_indices]
