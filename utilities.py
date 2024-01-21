import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pandas as pd
from scipy.sparse import csr_matrix

from visualizer import Visualizer


# pip install scikit-learn
def prepare_and_save(df, train_path, test_path):
    # Split dataset into train and test sets
    df = shuffle(df)
    # random_state: Controls the shuffling applied to the data before applying the split.
    train, test = train_test_split(df, test_size=0.2, random_state=1)
    # Save train and test sets to /data directory
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)


def load_data(file_path):
    # Load CSV file into a pandas dataframe
    df = pd.read_csv(file_path)

    # Extract the sentiment label column and convert it to a NumPy array
    labels = df['sentiment'].to_numpy()

    # Extract the feature columns (excluding sentiment, hasEmoji, and hasEmoticon)
    feature_cols = [col for col in df.columns if col not in ['sentiment', 'hasEmoji', 'hasEmoticon']]
    features = df[feature_cols].values

    # Convert features to a sparse matrix
    features_sparse = csr_matrix(features)

    return labels, features_sparse
