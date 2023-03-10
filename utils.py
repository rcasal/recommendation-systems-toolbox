import pandas as pd


def load_demo_df(name: str = 'MovieLens', n_samples: int = 10000) -> pd.DataFrame:
    """
    Load a demo DataFrame based on the name parameter.

    Args:
        name (str): The name of the dataset to load (currently 'MovieLens' or 'Vea').
        n_samples (int): The number of rows to sample from the dataset (default: 10,000).

    Returns:
        pd.DataFrame: A DataFrame with demo data.
    """

    # Convert name to lowercase for case-insensitivity
    name = name.lower()

    # Load the data based on the name parameter
    if name == 'movielens':
        df = pd.read_csv('data/movieLens/ratings.csv').sample(n=n_samples)
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
    elif name == 'vea':
        df = pd.read_csv('data/cm_vea_full_raw.csv').sample(n=n_samples)
        df = df.rename(columns={'IdCliente': 'user_id', 'CodMaterial': 'item_id', 'VolumenUnidades': 'rating'})
        df = df[df['item_id'] != '(not set)']
    else:
        raise ValueError(f"Invalid dataset name: {name}. Must be either 'MovieLens' or 'Vea'.")

    return df