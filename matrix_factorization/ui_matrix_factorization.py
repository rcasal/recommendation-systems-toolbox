import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

class SVDRecommender:
    def __init__(self, user_item_ratings: pd.DataFrame):
        self.user_item_ratings = user_item_ratings
        
        # Check if user_item_ratings data contains required columns
        if set(['user_id', 'item_id', 'rating']).issubset(user_item_ratings.columns):
            # Convert user_item_ratings to user-item matrix
            self.user_item_matrix = user_item_ratings.pivot_table(index='user_id', columns='item_id', values='rating', aggfunc='mean').fillna(0)
            self.user_id_col = self.user_item_matrix.reset_index()['user_id']

        else:
            raise ValueError("Input DataFrame must contain columns 'user_id', 'item_id', and 'rating'")

        self.df_columns = self.user_item_matrix.columns
        self.predicted_ratings = None
    
    def get_predicted_ratings_mf(self, num_singular_values: int) -> pd.DataFrame:
        """ Factorizes a user-item matrix using SVD to make recommendations

        Args:
            num_singular_values (int): Number of singular values to use for SVD

        Returns:
            pd.DataFrame: Predicted ratings for each user and item
        """

        # Check if input data is a pandas DataFrame, and convert to numpy array if needed
        if isinstance(self.user_item_matrix, pd.DataFrame):
            user_item_matrix_np = self.user_item_matrix.to_numpy()
        else:
            raise TypeError("Input data must be a pandas DataFrame with user_id in rows and item_id in columns")

        # Demean the user-item matrix and store the rating means for reconstruction
        rating_means = np.mean(user_item_matrix_np, axis=1, keepdims=True)
        user_item_matrix_demeaned = user_item_matrix_np - rating_means

        # Perform SVD on the demeaned user-item matrix
        U, sigma, Vt = svds(user_item_matrix_demeaned, k=num_singular_values)

        # Convert sigma from diagonal matrix to vector
        sigma = np.diag(sigma)

        # Make predictions by multiplying U, sigma, and Vt and adding the rating means back in
        self.predicted_ratings = np.dot(np.dot(U, sigma), Vt) + rating_means.reshape(-1, 1)
        
        # check if both DataFrames have the same number of rows
        if len(self.user_id_col) != len(self.predicted_ratings):
            raise ValueError(f"The two DataFrames (user_id_col and predicted_ratings) do not have the same number of rows. "
                         f"df_left has {len(user_id_col)} rows and df_right has {len(predicted_ratings)} rows.")

        # concatenate horizontally while keeping only the first column of the left DataFrame
        self.predicted_ratings = pd.concat([self.user_id_col, pd.DataFrame(self.predicted_ratings, columns=self.df_columns)], axis=1)
        
        return self.predicted_ratings

    def recommend(self, user_id: str, num_recommendations: int = 5) -> pd.Series:
        """ Recommend top movies for a given user based on predicted ratings

        Args:
            user_id (str): ID of the user to recommend movies for
            num_recommendations (int, optional): Number of movies to recommend. Defaults to 5.

        Returns:
            pd.Series: Series of recommended movies and their predicted ratings
        """

        if self.predicted_ratings is None:
            raise ValueError("Must first call get_predicted_ratings_mf method to obtain predicted ratings")

        if not self.predicted_ratings.user_id.isin([user_id]).any(): #or not self.user_item_matrix.user_id.isin([user_id]).any():
            raise ValueError(f"The user_id: {user_id} is not present in the dataset.")

        # extract predicted ratings for the given user_id
        predicted_ratings_for_user = self.predicted_ratings[self.predicted_ratings.user_id==user_id].reset_index().transpose()[2::].rename(columns={0: 'ratings'}).sort_values(by='ratings', ascending=False).reset_index().rename(columns={'index': 'item_id'})

        # extract user_item_matrix row for the given user_id
        user_item_matrix_for_user = self.user_item_matrix.reset_index()[self.user_item_matrix.reset_index().user_id==user_id].reset_index().transpose()[2::].rename(columns={0: 'ratings'}).sort_values(by='ratings', ascending=False).reset_index().rename(columns={'index':'item_id'})
        user_item_matrix_for_user = user_item_matrix_for_user[user_item_matrix_for_user.ratings != 0]

        # filter out the 
        # sort the predicted ratings in descending order and return the top n movie recommendations
        merged_df = predicted_ratings_for_user.merge(user_item_matrix_for_user.item_id, on='item_id', how='left', indicator=True)
        top_recommendations = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)[:num_recommendations]
        
        return top_recommendations, user_item_matrix_for_user

