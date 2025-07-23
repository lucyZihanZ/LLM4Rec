from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# ----- Mock Product Database -----
MOCK_PRODUCTS = [
    {
        "id": 1,
        "name": "Energy Booster Coffee",
        "description": "A rich, dark roast coffee blend with natural ginseng.",
        "augmented": "Contains ginseng which may help sustain energy levels throughout your day.",
        "tags": ["energy", "focus", "coffee"]
    },
    {
        "id": 2,
        "name": "Immunity Support Smoothie",
        "description": "Packed with vitamin C and antioxidants.",
        "augmented": "Formulated with orange and turmeric to support your immune health.",
        "tags": ["immune", "vitamin c", "smoothie"]
    },
    {
        "id": 3,
        "name": "Protein Power Bar",
        "description": "A high-protein snack bar made with nuts and whey.",
        "augmented": "Ideal for post-workout muscle recovery and on-the-go energy.",
        "tags": ["protein", "snack", "energy"]
    }
]

# ----- Pydantic Schemas -----
class Product(BaseModel):
    id: int
    name: str
    description: str
    augmented: str
    tags: List[str]

class RecommendationResponse(BaseModel):
    recommendations: List[Product]

class ProductResponse(BaseModel):
    product: Product

# ----- Basic Mock Recommendation -----
def recommend_products(query: str, top_k: int = 3):
    """ Very basic recommendation: return products whose tags or name match the query. """
    query_lower = query.lower()
    results = [
        prod for prod in MOCK_PRODUCTS
        if query_lower in prod['name'].lower() or any(query_lower in tag for tag in prod['tags'])
    ]
    # fallback: if no match, return top k
    if not results:
        results = MOCK_PRODUCTS[:top_k]
    return results[:top_k]

def get_product_by_id(pid: int):
    for prod in MOCK_PRODUCTS:
        if prod['id'] == pid:
            return prod
    return None

# ----- FastAPI Endpoints -----
@app.get("/recommend", response_model=RecommendationResponse)
def recommend_endpoint(query: str = Query(..., description="User query for recommendations"), top_k: int = 3):
    recs = recommend_products(query, top_k)
    return {"recommendations": recs}

@app.get("/product/{product_id}", response_model=ProductResponse)
def product_info_endpoint(product_id: int):
    prod = get_product_by_id(product_id)
    if not prod:
        return {"error": "Product not found"}
    return {"product": prod}

# Optionally add a root endpoint for demo
@app.get("/")
def root():
    return {"message": "Product RAG+ML Recommendation API is running."}
