import pandas as pd

class MostPopularProductRecommender:
    def __init__(self, purchase_data: pd.DataFrame):
        self.purchase_data = purchase_data

    def recommend(self, num_recommendations: int = 5) -> pd.Series:
        """Recommend the most bought product(s).

        Args:
            num_recommendations (int, optional): Number of recommendations to provide. Defaults to 1.

        Returns:
            list: List of item_id(s) recommended based on most bought.
        """
        # Group the purchase data by item_id and count the number of purchases for each product
        product_counts = self.purchase_data['item_id'].value_counts().reset_index()
        product_counts.columns = ['item_id', 'purchase_count']

        # Sort the products by purchase count in descending order
        sorted_products = product_counts.sort_values(by='purchase_count', ascending=False)

        # Get the most bought product(s) based on the specified number of recommendations
        top_products = sorted_products.head(num_recommendations)

        return top_products['item_id'].tolist()
