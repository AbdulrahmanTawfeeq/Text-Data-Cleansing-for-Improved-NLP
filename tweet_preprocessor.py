import re

import contractions
import emoji
from emot import emoticons
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from visualizer import Visualizer


# nltk.download('stopwords')
# nltk.download('wordnet')'
# nltk.download('punkt')


class TweetPreprocessor:
    emoji_re = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u'\U00010000-\U0010ffff'
                          u"\u200d"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\u3030"
                          u"\ufe0f"
                          "]+", flags=re.UNICODE)
    emoticon_re = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
    url_re = re.compile("(https?://)?([\w\-])+\.{1}([a-zA-Z]{2,63})([/\w-]*)*/?\??([^#\n\r]*)?#?([^\n\r]*)")
    mention_re = re.compile("(@[\w_-]+)")
    hashtag_re = re.compile("(#[\w_-]+)")

    def __init__(self, df):
        selected_features = ["text", "airline_sentiment"]
        #
        print("-----Selected Features & Renaming Start-----")
        print(df.keys().tolist())
        # Preprocess the DataFrame
        df = TweetPreprocessor.feature_selection(df, selected_features)
        # Renaming the columns
        df.rename(columns={'airline_sentiment': 'sentiment'}, inplace=True)
        print(df.keys().tolist())
        print("-----Selected Features & Renaming  End-----\n")

        df = TweetPreprocessor.remove_duplicates(df)

        print(f"-----Check for null values Start-----")
        print(df.isnull().sum())
        print("-----Check for null values End-----\n")

        df = TweetPreprocessor.normalize(df)

        Visualizer.plot_comparison(df, 'sentiment', 'All Tweets Labels', 'Labels', 'Tweets')

        df = TweetPreprocessor.encode_sentiment(df)
        df = TweetPreprocessor.feature_selection(df, ['text', 'sentiment'])
        df = TweetPreprocessor.lowercase_tweet(df)
        df = TweetPreprocessor.replace_emoticons(df, emoticon_replacement='word')  # :) -> happy
        df = TweetPreprocessor.remove_hashtags_urls_mentions_text(df)
        df = TweetPreprocessor.expand_contractions(df)
        df = TweetPreprocessor.convert_emojis(df, emoji_replacement='word')
        df = TweetPreprocessor.remove_punctuations(df)  # will not remove _
        df = TweetPreprocessor.remove_stopwords(df)
        df = TweetPreprocessor.tokenize(df)
        df = TweetPreprocessor.lemmatize(df)
        df.to_csv("data/preprocessed_tweets/preprocessed_df.csv", index=False)
        df = TweetPreprocessor.vectorize(df)

        self.preprocessed_df = df

    def get_preprocessed_df(self):
        return self.preprocessed_df

    @staticmethod
    def feature_selection(df, selected_features):
        new_df = df[selected_features].copy()
        return new_df

    @staticmethod
    def remove_duplicates(df):
        df_copy = df.copy()
        print("-----Check Duplicated Start-----")
        print(f"Number of duplicated rows: {df_copy.duplicated().sum()}")
        df_copy = df_copy.drop_duplicates()
        print(f"Number of duplicated rows: {df_copy.duplicated().sum()}")
        print("-----Check Duplicated End-----\n")
        return df_copy

    @staticmethod
    def encode_sentiment(df):
        """
        Encodes the sentiment column of a dataframe with the following mapping:
        Positive: 1
        Neutral: 0
        Negative: -1
        """
        df_copy = df.copy()
        sentiment_map = {
            'positive': 1,
            'neutral': 0,
            'negative': -1,
        }
        df_copy['sentiment'] = df_copy['sentiment'].apply(lambda s: sentiment_map.get(s.lower()))
        print("-----Encoding Sentiment Start-----")
        print(df['sentiment'].iloc[1])
        print(df_copy['sentiment'].iloc[1])
        print("-----Encoding Sentiment End-----\n")
        return df_copy

    @staticmethod
    def normalize(df):
        normalized_df = df.copy()

        positive_df = normalized_df[normalized_df['sentiment'] == 'positive']
        negative_df = normalized_df[normalized_df['sentiment'] == 'negative']
        neutral_df = normalized_df[normalized_df['sentiment'] == 'neutral']

        min_count = min(positive_df.shape[0], negative_df.shape[0], neutral_df.shape[0])

        # Step 3: Sample a subset of rows for each label type to match the minimum count
        positive_sampled = positive_df.sample(min_count, replace=False)
        negative_sampled = negative_df.sample(min_count, replace=False)
        neutral_sampled = neutral_df.sample(min_count, replace=False)

        # Step 4: Concatenate the sampled subsets to create a balanced DataFrame
        normalized_df = pd.concat([positive_sampled, negative_sampled, neutral_sampled])

        # Shuffle the rows in the balanced DataFrame
        normalized_df = normalized_df.sample(frac=1).reset_index(drop=True)

        return normalized_df

    @staticmethod
    def lowercase_tweet(df):
        """
        helps to reduce the complexity of the data and make it more consistent.
        """
        # make a copy of the original DataFrame
        df_copy = df.copy()
        # convert text to lowercase
        df_copy['text'] = df_copy['text'].str.lower()
        # Output the modified DataFrame
        print("----- Lowercase Start -----")
        print(df['text'].iloc[1])
        print(df_copy['text'].iloc[1])
        print("----- Lowercase End -----\n")
        return df_copy

    @staticmethod
    def replace_emoticons(df, emoticon_replacement='word'):
        def replace_emoticon_with_word(tweet):
            """
            In this implementation, we define a nested function called replace_emoticon_with_word that takes a tweet
            text as its input and returns the tweet text with emoticons replaced with their corresponding words.

            The replace function is defined within replace_emoticon_with_word and operates on a single emoticon match.
            It returns the replacement word for the emoticon if it is found in the emoticons dictionary, or the original
            emoticon string otherwise.

            The apply method is called on the 'text' column of the input DataFrame df, passing
            replace_emoticon_with_word as the function to apply to each tweet text. The resulting DataFrame copy is
            returned as the output of the method.
            """

            def replace(match):
                """
                group(0) is a method of a regular expression match object (Match class in re) that returns the entire
                substring of the tweet that matched the pattern in the regular expression.

                The regular expression pattern (?:\:|\;|\=)(?:\-\)?)(?:\)|D|P) matches emoticons
                that start with either :, ;, or = followed by an optional -, and end with either ), D, or P.

                When a match is found in the tweet text, first param of get method which is match.group(0) returns
                the entire matched substring, which is the emoticon string. This value is then used as the key for
                the emoticons dictionary lookup to obtain the corresponding replacement word. If the key is not found
                in the dictionary, the second param of get method which is match.group(0) is returned instead to keep
                the original emoticon string in the tweet text. Two spaces added before and after
                """
                return ' ' + emoticons.get(match.group(0), match.group(0)) + ' '

            if emoticon_replacement == 'word':
                return TweetPreprocessor.emoticon_re.sub(replace, tweet)
            elif emoticon_replacement == 'space':
                # Means that you want to remove emoticons
                return TweetPreprocessor.emoticon_re.sub(' ', tweet)

        df_copy = df.copy()
        df_copy['text'] = df_copy['text'].apply(replace_emoticon_with_word)

        # Output the modified DataFrame
        print("----- Replace Emoticons Start -----")
        print(df['text'].iloc[1])
        print(df_copy['text'].iloc[1])
        print("----- Replace Emoticons End -----\n")
        return df_copy

    @staticmethod
    def convert_emojis(df, emoji_replacement='word'):
        # Create a copy of the original DataFrame
        df_copy = df.copy()
        # Apply the function to the text column of the copied DataFrame
        if emoji_replacement == 'word':
            df_copy['text'] = df_copy['text'].apply(
                lambda tweet: emoji.demojize(tweet, delimiters=(" ", " ")))
        elif emoji_replacement == 'space':
            # Means that you want to remove emojis
            df_copy['text'] = df_copy['text'].apply(lambda tweet: TweetPreprocessor.emoji_re.sub(" ", tweet))
            # remove the variation selector VS16, some emoji characters are represented by Unicode code ✈️: U+2708
            # will then be replaced by " " and "️"
            df_copy['text'] = df_copy['text'].str.replace(" ️", " ")

        # Output the modified DataFrame
        print("----- Convert emojis Start -----")
        print(df['text'].iloc[1])
        print(df_copy['text'].iloc[1])
        print("----- Convert emojis End -----\n")
        # Return the updated DataFrame
        return df_copy

    @staticmethod
    def remove_hashtags_urls_mentions_text(df):
        df_copy = df.copy()
        df_copy['text'] = df_copy['text'].apply(lambda tweet: re.sub(TweetPreprocessor.hashtag_re, ' ', tweet))
        df_copy['text'] = df_copy['text'].apply(lambda tweet: re.sub(TweetPreprocessor.url_re, ' ', tweet))
        df_copy['text'] = df_copy['text'].apply(lambda tweet: re.sub(TweetPreprocessor.mention_re, ' ', tweet))

        print("----- Remove hashtags, urls, and mentions Start -----")
        print(df['text'].iloc[1])
        print(df_copy['text'].iloc[1])
        print("----- Remove hashtags, urls, and mentions End -----\n")
        return df_copy

    @staticmethod
    def remove_punctuations(df):
        """
        A function to remove punctuations from a single string !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

        In sentiment analysis, question marks ("?") and exclamation points ("!") can convey important information about
        the sentiment or emotion being expressed in the text. So, we are not going to remove them

        In this pattern, each punctuation mark is included inside a character set []. Note that some punctuation marks
        have special meaning in regular expressions (such as . and |), so they need to be escaped with a backslash to be
        treated as literal characters. Also, the backslash character itself needs to be escaped with another backslash
        to be treated as a literal character in the regular expression pattern.
        """
        # Create a copy of the original DataFrame
        df_copy = df.copy()
        # Define the regular expression pattern to remove punctuation marks ”“"#$%&'()*+,-./:;<=>@[\]^_`{|}~
        # _ has been removed to not affect the text of converted emo like happy_face
        pattern = r'[\"\“\”#\$%&\'\(\)\*\+,\-\./:;<=>@\[\\\]\^`\{\|\}\~]'

        # Apply the regular expression pattern to the specified column and replace with spaces
        # so that hello...world not to be helloworld
        df_copy['text'] = df_copy['text'].apply(lambda x: re.sub(pattern, ' ', x))

        # the regular expression pattern ([?!]) matches any occurrence of either "?" or "!" and captures it as a group.
        # The replacement string r" \1 " replaces the matched text with itself, but with a space added before and after
        # it. to have like!!! be like ! ! !
        df_copy['text'] = df_copy['text'].apply(lambda x: re.sub(r"([?!])", r" \1 ", x))

        # Output the modified DataFrame
        print("----- Remove punctuations Start -----")
        print(df['text'].iloc[1])
        print(df_copy['text'].iloc[1])
        print("----- Remove punctuations End -----")
        print("")
        # Return the updated DataFrame
        return df_copy

    @staticmethod
    def remove_stopwords(df):
        """
        The benefit of removing stopwords is that they do not carry much meaning by themselves and are very common,
        so they can be safely removed without losing much information. This can improve the accuracy and efficiency
        of downstream natural language processing tasks, such as sentiment analysis.

        There are some potential drawbacks to removing stop words, including:

        Loss of Context: Stop words often provide context to the text, and removing them can result in the loss of
        important information. For example, removing "not" from a sentence can change its meaning entirely.

        Negation Handling: In sentiment analysis, negation handling is important to correctly identify the sentiment
        of the text. However, removing stop words like "not" and "no" can make it difficult to identify negation.
        """
        # I am ahmed ->
        # for word in ['I', 'am', 'ahmed']:
        #   if word.lower() not in stop_words:
        #       return word

        stop_words = set(stopwords.words('english'))
        stop_words.discard('not')
        stop_words.discard('no')
        stop_words.discard('never')
        stop_words.discard('nor')

        df_copy = df.copy()
        df_copy['text'] = df_copy['text'].apply(
            lambda tweet: ' '.join([word for word in tweet.split() if word.lower() not in stop_words]))

        # Output the modified DataFrame
        print("----- Remove stopwords Start -----")
        print(df['text'].iloc[1])
        print(df_copy['text'].iloc[1])
        print("----- Remove stopwords End -----")
        print("")
        return df_copy

    @staticmethod
    def expand_contractions(df):
        df_copy = df.copy()

        # Expanding contractions is an important step in text preprocessing as it helps to standardize text and reduce
        # variation in language. Ex: I'm will be im after lowering and removing punctuations and it is not a strop word,
        # so it will stay there.
        df_copy['text'] = df_copy['text'].apply(lambda tweet: contractions.fix(tweet))

        # Output the modified DataFrame
        print("----- Expand Contractions Start -----")
        print(df['text'].iloc[1])
        print(df_copy['text'].iloc[1])
        print("----- Expand Contractions End -----\n")
        return df_copy

    @staticmethod
    def tokenize(df):
        df_copy = df.copy()
        df_copy['text'] = df_copy['text'].apply(lambda x: ' '.join(word_tokenize(x)))

        # Output the modified DataFrame
        print("----- Tokenization Start -----")
        print(df['text'].iloc[1])
        print(df_copy['text'].iloc[1])
        print("----- Tokenization End -----\n")
        return df_copy
        return df_copy

    @staticmethod
    def lemmatize(df):
        """
        Stemming and lemmatization are two common techniques used in natural language processing to reduce words to
        their base or root forms. The main difference between stemming and lemmatization is that stemming involves
        simply removing the suffix from a word to reduce it to its base form, while lemmatization involves using
        knowledge of the language to transform the word to its base form by considering its context and morphology.

        stemming is faster but less accurate, lemmatization is slower but more accurate

        For example, stemming might convert the words "running" and "runner" to "run", while lemmatization would convert
        "running" to "run" and "runner" to "runner", recognizing that the latter is a noun form of the verb "run".
        as here we care more about the context because we'll analyze the sentiment, so we will use lemmatization
        """
        # 'new nice houses'
        # new_list = []
        # for word in tweet.split(" ")
        #   lem_word = lem.lemmatize(word) houses -> house
        #   new_list.append(lem_word)
        # new_list = ['new','nice','house']
        # ' '.join(new_list) -> 'new nice house'
        lem = WordNetLemmatizer()
        df_copy = df.copy()
        df_copy['text'] = df_copy['text'].apply(
            lambda tweet: " ".join([lem.lemmatize(word, pos="v") for word in tweet.split(" ")]))

        # Output the modified DataFrame
        print("----- Lemmatize Start -----")
        print(df['text'].iloc[1])
        print(df_copy['text'].iloc[1])
        print("----- Lemmatize End -----\n")
        return df_copy

    @staticmethod
    def vectorize(df):
        # count vectorizer
        # hello how are you doing. -> 111110000
        # hello I am fine, how about you. how about -> 120101112

        #     hello how are you doing I am fine about
        # 1   1      1   1   1   1    0  0  0     0
        # 2   1      2   0   1   0    1  1  1     2

        # TF-IDF -> n grams (1, 1) (1, 2) (1, 3) , (2, 2)
        # hello how, how are, never like
        # Example for TF-IDF: Imagine we have a document “Tweet” containing 10 words, and the word “like” appears 3
        # times. Assume that we have 1000 documents “Tweets” in the dataset and the word “like” appears in 10 tweets.
        # Calculate the weight for the word “like”.
        #
        # TF(like) = number of times “like” appears in the tweet / total words in the tweet
        # TF(like) = 3/10 = 0.3
        #
        # IDF(like) = log (tweets/ tweets have the  like word)
        # IDF(like) = log (1000/10) = 2
        #
        # TF-IDF weight = 0.3 * 2 = 0.6

        #     hello how are you doing I am fine about
        # 1   0.6    0.3   1   1   1    0  0  0     0
        # 2   1      2   0   1   0    1  1  1     2

        # create a copy of the DataFrame
        df_copy = df.copy()

        # drop rows where the "text" column is null
        df_copy.dropna(subset=['text'], inplace=True)

        # define the TF-IDF vectorizer with custom parameters.
        # analyzer='char'
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))

        # fitting the vectorizer on the "text" column involves analyzing the text data in the "text" column of the
        # DataFrame to create a vocabulary of unique words and bigrams (in this case), and assigning a weight to each
        # word/bigram based on its frequency in the corpus of text data. This process is used to create a sparse matrix
        # representation of the text data, where each row corresponds to a document (in this case, a tweet) and each
        # column corresponds to a unique word/bigram in the vocabulary. The values in the matrix represent the TF-IDF
        # weight of each word/bigram in each document.

        # transform the "text" column using the fitted vectorizer. To actually use this vocabulary to transform the text
        # data into a numerical representation that can be used as input to machine learning algorithms, we need to
        # "transform" the "text" column using the fitted vectorizer.
        vectors_features = tfidf.fit_transform(df_copy['text'])
        print(vectors_features)

        # create a traditional DataFrame from the sparse matrix using to array method and the original labels
        vectors_features = pd.DataFrame(vectors_features.toarray(), columns=tfidf.get_feature_names_out())

        # add the columns you want to include in the output dataframe
        other_cols = df_copy[['sentiment']].reset_index(drop=True)

        # concatenate the two dataframes horizontally
        df_copy1 = pd.concat([other_cols, vectors_features], axis=1)

        # Output the modified DataFrame
        print("----- Vectorize Start -----")
        print(vectors_features.shape)
        print(df_copy1.shape)
        print("----- Vectorize End -----\n")
        return df_copy1
