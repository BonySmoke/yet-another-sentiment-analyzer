import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import joblib
from wordcloud import WordCloud
from sentiment.util import update_stopwords
from sentiment.preprocessing import TextPreprocessing, UKTextPreprocessing
from pymorphy2 import MorphAnalyzer
import spacy


class Analizer:

    ML_ALGORITHM = {
        "SGD": SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001, max_iter=10, tol=None, shuffle=True, verbose=0, learning_rate='adaptive', eta0=0.01, early_stopping=False),
        "LogisticRegression": LogisticRegression(random_state=0, solver='lbfgs'),
        "SVM": svm.SVC(kernel='linear'),
        "RandomForest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2),
        "NaiveBayes": MultinomialNB()
    }

    def __init__(self):
        pass

    def load_model(self, filename):
        self.clf = joblib.load(filename)
        return self.clf

    @staticmethod
    def save_model(filename: str, model):
        joblib.dump(model, filename)
        return True

    def save_vectorizer(self, filename: str):
        if not self.tfidf:
            raise Exception("No Vectorizer object to save")
        joblib.dump(self.tfidf, filename)

    @staticmethod
    def save_dataframe(dataframe, filename: str):
        dataframe.to_csv(filename, index=False)

    def load_vectorizer(self, filename: str):
        self.tfidf = joblib.load(filename)
        return self.tfidf

    def get_sentiment_type_sum(self, s_type: str) -> int:
        return (self.data.sentiment.values == s_type).sum()

    def read_data_dev(self,
                      size=200,
                      balance=False,
                      maximum_tokens=0,
                      skiprows=0,
                      is_json=False,
                      dataset_path='data/Video_Games_5.json'):
        if is_json:
            self.data = pd.read_json(dataset_path, nrows=size, lines=True)
        else:
            self.data = pd.read_csv(dataset_path, nrows=size)
            # self.data.dropna(subset = ["processedText"], inplace=True)
        if skiprows:
            self.data = self.data[skiprows:]

        if balance:
            self._balance_data(maximum_tokens)
        self._set_review_weight()

    def prepare_data_dev(self, nlp=None, lang: str = "en"):
        if lang == "en":
            nlp = nlp if nlp else spacy.load("en_core_web_sm")
            self.data['processedText'] = self.data.apply(
                lambda text: TextPreprocessing(
                    text=str(text['reviewText']),
                    nlp=nlp,
                    expected_sentiment=text["overall"],
                ).normalize(), axis=1)
        if lang == "uk":
            nlp = nlp if nlp else MorphAnalyzer(lang="uk")
            self.data['processedText'] = self.data.apply(
                lambda text: UKTextPreprocessing(
                    text=str(text['reviewText']),
                    morph_analyzer=nlp
                ).normalize(), axis=1)

    def new_model(self, dataset_path: str = 'data/video_game_reviews.csv',
                  balance: bool = False):
        self._load_data(dataset_path=dataset_path)
        # self._set_review_weight()
        # self._prepare_data(language=language)
        if balance:
            self._balance_data()
        self._set_training_data()
        self._set_test_data()

    def _load_data(cls, dataset_path: str, size=25000, skip=0):
        """
        Convert json data into the dataframe and prepocess data

        Args:
            size (int, optional): The size of the dataframe. Defaults to 9000.
            skip (int, optional): The number of rows to skip in a dataset.
        """
        cls.data = pd.read_csv(dataset_path, nrows=size)
        cls.data = cls.data[skip:]
        cls.data.dropna(subset=["processedText"], inplace=True)
        print(len(cls.data))

    def _prepare_data(cls, language=None):
        nlp = language if language else spacy.load("en_core_web_sm")
        for i in range(cls.data.shape[0]):
            cls.data["reviewText"].iloc[i] = TextPreprocessing(
                text=cls.data["reviewText"].iloc[i],
                nlp=nlp
            ).normalize()

    def generate_wordcloud(self, reviews: list, new_stopwords=[]):
        """Generate a word cloud of the most popular words

        Args:
            reviews (list): [description]
            new_stopwords (list, optional): [description]. Defaults to [].

        Returns:
            Wordcloud: [description]
        """
        sw = update_stopwords(new_stopwords)
        text = " ".join(review for review in reviews.reviewText)
        wordcloud = WordCloud(stopwords=sw).generate(text)
        return wordcloud

    @staticmethod
    def _define_sentiment_type(rate):
        # if rate > 3:
        #     return "Positive"
        # if rate == 3:
        #     return "Neutral"
        # if rate < 3:
        #     return "Negative"
        if rate == 5: return "Very positive"
        if rate == 4: return "Positive"
        if rate == 3: return "Somewhat Neutral"
        if rate == 2: return "Negative"
        if rate == 1: return "Very negative"

    def _set_review_weight(cls):
        '''
        Add sentiment column to the dataframe
        '''
        # remove neutral reviews
        # cls.data = cls.data[cls.data['overall'] != 3]
        cls.data['sentiment'] = cls.data['overall'].apply(
            lambda rating: cls._define_sentiment_type(rating))
        # cls._balance_data() # adjust weights

    def _balance_data(cls, maximum_tokens: int = 0):
        """Balance data to get optimal distibution between types
        Method: undersampling. Remove samples from a bigger class
        to match the smaller one
        """
        if maximum_tokens:
            cls.data = cls.data[cls.data['reviewText'].str.split(
            ).str.len().lt(maximum_tokens)]
        sentiment_types = list()
        for sentiment in cls.data.sentiment.unique():
            sentiment_types.append({"type": sentiment, "samples": 0})

        for s_type in sentiment_types:
            s_type['samples'] = (cls.data.sentiment.values == s_type['type']
                                 ).sum()

        minor_class = sorted(sentiment_types, key=lambda x: x['samples'])[0]
        cls.data = cls.data.groupby('sentiment').apply(
            lambda x: x.sample(n=minor_class['samples'], replace=False)
        ).reset_index(drop=True)
        print(
            f'Finished balancing: {(cls.data.sentiment.values == "Very positive").sum()}')
        print(f'Total dataset size: {len(cls.data)}')
        return cls.data

    def _set_training_data(cls):
        cls.training_data = cls.data.sample(frac=0.8, random_state=25)

    def _set_test_data(cls):
        cls.testing_data = cls.data.drop(cls.training_data.index)

    def _split_dataset(self):
        """Split a dataset to training and testing
        """
        if not hasattr(self, "tfidf"):
            self.fit_corpus()

        self.train_features = self.transform_data(
            self.training_data)  # transforming
        self.test_features = self.transform_data(
            self.testing_data)  # Train and Test

        self.train_labels = self.training_data["sentiment"]
        self.test_labels = self.testing_data["sentiment"]

    def train(self, algorithm: str, partial: bool = False):
        """Train a model with a specific name

        Args:
            algorithm (str): The name of the classification algorithm
            partial (bool): If the data should be appended to the existing model
                            Does not work with all the models
        Returns:
            a trained model
        """
        if algorithm not in self.ML_ALGORITHM:
            raise KeyError(("There is no such algorithm. ") +
                           (f"Available algorithms are:\n{' '.join(self.ML_ALGORITHM.keys())}"))

        self._split_dataset()
        self.clf = self.ML_ALGORITHM[algorithm]

        if partial:
            print('partial is used')
            self.clf.partial_fit(self.train_features,
                                 self.train_labels,
                                 classes=np.unique(self.train_labels))
            return self.clf

        self.clf.fit(self.train_features, self.train_labels)
        return self.clf

    def fit_corpus(self):
        """Create TF-IDF representation of the dataframe

        Returns:
            TfidfVectorizer: [description]
        """
        corpus = pd.DataFrame(
            {"processedText": self.training_data["processedText"]})
        corpus.processedText.append(
            self.testing_data["processedText"], ignore_index=True)
        self.tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
        self.tfidf.fit(corpus["processedText"])
        return self.tfidf

    def transform_data(self, dataset) -> DataFrame:
        features = self.tfidf.transform(dataset["processedText"])
        return pd.DataFrame(features.todense(), columns=self.tfidf.get_feature_names_out())

    def precision_report(self, predictions):
        report = dict()
        report['confusion_matrix'] = confusion_matrix(
            self.test_labels, predictions)
        report['classification_report'] = classification_report(
            self.test_labels, predictions)
        report['accuracy_score'] = accuracy_score(
            self.test_labels, predictions)
        report['test_score'] = self.clf.score(
            self.train_features, self.train_labels) * 100
        return report
