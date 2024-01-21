# pip install scikit-learn
import pandas as pd

from models import Models
from tweet_preprocessor import TweetPreprocessor
from utilities import prepare_and_save, load_data
from visualizer import Visualizer


def preprocessing():
    df = pd.read_csv('data/Tweets.csv', delimiter=',')
    df = df[df['airline_sentiment_confidence'] > 0.65]  # accuracy

    preprocessor = TweetPreprocessor(df)
    preprocessed_df = preprocessor.get_preprocessed_df()
    print(preprocessed_df.head())

    # prepare_and_save(preprocessed_df,
    #                  train_path="data/clean_data/train_set.csv",
    #                  test_path="data/clean_data/test_set.csv")


def models_training_testing():

    train_labels, train_features_sparse = load_data('data/clean_data/train_set.csv')
    test_labels, test_features_sparse = load_data('data/clean_data/test_set.csv')

    print('NB')
    Models.train_nb(train_features_sparse, train_labels, test_features_sparse, test_labels)

    print('SVM')
    Models.train_svm(train_features_sparse, train_labels, test_features_sparse, test_labels)

    print('DT')
    Models.train_dt(train_features_sparse, train_labels, test_features_sparse, test_labels)

    print('LR')
    Models.train_lr(train_features_sparse, train_labels, test_features_sparse, test_labels)

    print('xgb with considering emojis and emoticons')
    Models.train_xgb(train_features_sparse, train_labels, test_features_sparse, test_labels)


preprocessing()
# models_training_testing()
