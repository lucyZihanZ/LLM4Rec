# Combination of all functions.
from prepare import load_product_database, add_combined_features
from ml_rec import ProductRecommender
from rag import RAGProductRecommender
def main():
# load the data
    product = 'products.json'
    product_db = load_product_database(product)
    product_database = add_combined_features(product_db)
# ml for recommendations
    mlrec = ProductRecommender(product_database)
    product_name = "Energy Booster Coffee"
    product_row = mlrec.products_df[mlrec.products_df['name'].str.lower() == product_name.lower()]
    if product_row.empty:
        print(f"Product with name '{product_name}' not found.")
        return
    product_id = product_row.index[0]

    top_k = 3
    recommended_ids = mlrec.get_top_k_recommendations(product_id, k=top_k)
    print("Top-K recommended product ids:", recommended_ids)

    # 4. 增强产品
    enhanced_products = mlrec.enhance_recommendations(recommended_ids)
    # 5. 转换为产品名字
    product_names = [prod['name'] for prod in enhanced_products if 'name' in prod]
    print("Recommended product names:", product_names)


