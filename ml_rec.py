import pandas as pd
from prepare import load_product_database, add_combined_features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class ProductRecommender:
    """
    A class to handle product data loading, similarity calculation,
    and provide content-based recommendations with RAG augmentation capabilities.
    """
    def __init__(self, filename: str):
        """
        Args:
            filename (str): The path to the JSON file containing product data.
        """
        self.products = self._load_product_database(filename)
        self.products_df = pd.DataFrame(self.products)
        self.products_df.set_index('id', inplace=True)

        # Create a dictionary for quick lookup of product details by string ID
        self.product_db_indexed = {str(row_id): row.to_dict() for row_id, row in self.products_df.iterrows()}

        # Combine relevant text features for TF-IDF vectorization
        self.products_df['combined_features'] = self.products_df.apply(
            lambda row: f"{row['name']} {row['description']} {' '.join(row['effects'])} {' '.join(row['ingredients'])} {row['type']}",
            axis=1
        )

        # Initialize and fit TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.product_tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['combined_features'])
        self.cosine_sim_matrix = cosine_similarity(self.product_tfidf_matrix)
        self.cosine_sim_df = pd.DataFrame(
            self.cosine_sim_matrix,
            index=self.products_df.index,
            columns=self.products_df.index
        )

    def _load_product_database(self, filename: str) -> list:
        """
        Loads product data from a JSON file. If the file does not exist,
        it creates a dummy database and saves it to the file.
        This is an internal helper method.
        """
        if os.path.exists(filename):
            print(f"Loading product database from {filename}...")
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return f"File {filename} not found. Creating a dummy product database and saving it."
            

    def get_top_k_recommendations(self, product_id: int, k: int = 3) -> list:
        """
        Gets the top K most similar product recommendations for a given product ID
        based on content similarity.

        Args:
            product_id (int): The ID of the product for which to find recommendations.
            k (int): The number of top recommendations to return.

        Returns:
            list: A list of recommended product IDs, sorted by similarity in descending order.
                  Returns an empty list if the product ID is not found.
        """
        if product_id not in self.cosine_sim_df.index:
            print(f"Product ID {product_id} not found in the database.")
            return []
        # Get similarity scores for the given product from the pre-calculated matrix
        similarity_scores = self.cosine_sim_df[product_id]

        # Sort the scores in descending order and exclude the product itself
        sorted_similarities = similarity_scores.sort_values(ascending=False)
        
        # Filter out the queried product itself (similarity to self is 1.0)
        recommended_products_with_scores = sorted_similarities[sorted_similarities.index != product_id]
        
        # Get the top K recommendations
        top_k_recommendations = recommended_products_with_scores.head(k)
        return top_k_recommendations.index.tolist()

    def retrieve_product_details(self, product_id: str) -> dict | None:
        """
        Retrieves detailed information for a given product ID from the internal database.

        Args:
            product_id (str): The unique identifier of the product (as a string).

        Returns:
            dict or None: A dictionary containing the product's details if found,
                          otherwise None.
        """
        return self.product_db_indexed.get(str(product_id))

    def enhance_recommendations(self, recommended_ids: list, user_preferences: dict = None) -> list:
        """
        Retrieves detailed product information for recommended IDs and augments
        the recommendations with richer details and potentially personalized descriptions.

        Args:
            recommended_ids (list): A list of product IDs recommended by the algorithm.
            user_preferences (dict, optional): A dictionary of user preferences
                                              (e.g., {"prefers_vegan": True, "budget_conscious": True}).
                                              Defaults to None.

        Returns:
            list: A list of augmented product recommendation dictionaries.
        """
        augmented_recommendations = []
        for prod_id in recommended_ids:
            # Ensure prod_id is a string for lookup in product_db_indexed
            details = self.retrieve_product_details(str(prod_id))
            if details:
                # Create a copy to avoid modifying the original database entry
                augmented_item = details.copy()

                # Example of Enriching Recommendation Display
                display_text = f"**{augmented_item['name']}**"
                if augmented_item.get('on_sale', False): # Use .get() with default for safety
                    display_text += " (ON SALE!)"
                display_text += f" - ${augmented_item['price']:.2f}"
                display_text += f" (⭐{augmented_item.get('rating', 'N/A')})" # Use .get()
                augmented_item['display_text'] = display_text

                # Example of Generating Dynamic/Personalized Product Descriptions
                personalized_description = augmented_item['description']
                if user_preferences:
                    if user_preferences.get("prefers_vegan") and "vegan" in augmented_item['description'].lower():
                        personalized_description = f"**Great for you!** {personalized_description} This product aligns with your vegan preference."
                    if user_preferences.get("budget_conscious") and augmented_item.get('on_sale', False):
                        personalized_description = f"**Budget-friendly choice!** {personalized_description} Currently available at a special price."
                augmented_item['personalized_description'] = personalized_description

                augmented_recommendations.append(augmented_item)
            else:
                augmented_recommendations.append({"product_id": prod_id, "status": "details_not_found"})

        return augmented_recommendations

    def get_augmented_product_context(self, product_ids: list, user_preferences: dict = None) -> str:
        """
        Retrieves and augments product details for a list of product IDs,
        then formats them into a single string suitable as context for an LLM.

        Args:
            product_ids (list): A list of product IDs.
            user_preferences (dict, optional): User preferences for personalization.

        Returns:
            str: A formatted string containing augmented product information.
        """
        augmented_products = self.enhance_recommendations(product_ids, user_preferences)
        
        context_string = "Here are some product recommendations with their details:\n\n"
        for i, product in enumerate(augmented_products):
            if product.get("status") == "details_not_found":
                context_string += f"Product ID {product['product_id']}: Details not found.\n"
                continue

            context_string += f"--- Product {i+1}: {product['name']} ---\n"
            context_string += f"Category: {product.get('type', 'N/A')}\n" # Use .get() for safety
            context_string += f"Brand: {product.get('brand', 'N/A')}\n"
            context_string += f"Price: ${product['price']:.2f}\n"
            context_string += f"Rating: {product.get('rating', 'N/A')} stars ({product.get('reviews_count', 'N/A')} reviews)\n"
            if product.get('on_sale', False):
                context_string += "Status: ON SALE!\n"
            context_string += f"Description: {product['personalized_description']}\n"
            context_string += f"Image URL: {product.get('image_url', 'N/A')}\n"
            context_string += "\n"
        return context_string.strip()

if __name__ == "__main__":
    PRODUCT_DB_FILENAME = "products.json" # Make sure this matches your actual file name

    # Instantiate the recommender class
    # This creates an object 'recommender' which holds all the product data,
    # TF-IDF matrix, and cosine similarity DataFrame internally.
    recommender = ProductRecommender(PRODUCT_DB_FILENAME)

    print("--- Available Products ---")
    # Access the DataFrame through the 'recommender' object
    for p_id, product_info in recommender.products_df.iterrows():
        print(f"ID: {p_id}, Name: {product_info['name']}")

    print("\n--- Generating Top K Recommendations ---")

    # Example 1: Get 2 recommendations for "Digestive Wellness Shot" (ID: 4)
    # This was the product in your traceback that led to the error
    product_id_to_query = 4
    num_recs = 2
    recommended_ids_1 = recommender.get_top_k_recommendations(product_id_to_query, num_recs)
    print(f"\nTop {num_recs} recommendations for '{recommender.products_df.loc[product_id_to_query]['name']}' (ID: {product_id_to_query}):")
    if recommended_ids_1:
        for rec_id in recommended_ids_1:
            # Use 'recommender.products_df.loc[rec_id]' to get the product details
            # from the DataFrame that is correctly indexed by product IDs.
            print(f"  - {recommender.products_df.loc[rec_id]['name']} (ID: {rec_id})")
    else:
        print("  No recommendations found.")
    product_id_to_query = 5
    num_recs = 3
    recommended_ids_2 = recommender.get_top_k_recommendations(product_id_to_query, num_recs)
    print(f"\nTop {num_recs} recommendations for '{recommender.products_df.loc[product_id_to_query]['name']}' (ID: {product_id_to_query}):")
    if recommended_ids_2:
        for rec_id in recommended_ids_2:
            print(f"  - {recommender.products_df.loc[rec_id]['name']} (ID: {rec_id})")
    else:
        print("  No recommendations found.")

