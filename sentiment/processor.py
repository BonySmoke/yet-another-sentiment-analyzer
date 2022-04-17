from sentiment import en_nlp, uk_nlp
from sentiment.preprocessing import TextPreprocessing, UKTextPreprocessing
from sentiment.model import Analizer
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import time


class Process:
    '''
    Process user input and make predictions
    '''
    CLF_TYPES = ['binary', 'three', 'fine']
    ALGORITHMS = {
        "nb": "NaiveBayes",
        "forest": "RandomForest",
        "linear": "LogisticRegression"
    }

    def __init__(self,
                 lang: str = "en",
                 clf_type: str = "binary",
                 algorithm: str = "nb",
                 mode="execute") -> None:
        '''
        :param lang: either english (en) or ukrainian (uk)
        :param clf_type: classification type
        :param algorithm: algorithm used to training
        :param mode: how the process should behave. In case of
                    "execute", the algorithm uses an existing model
                    in case of "train", a new model can be trained
        '''
        self.lang = lang
        self.clf_type = clf_type
        self.algorithm = algorithm

        self.model = Analizer()
        self.mode = mode

        if self.mode == "execute":
            self._prepare_model()

    def _prepare_model(cls) -> None:
        """
        Initialize a classifier and vectorizer
        """
        if cls.clf_type not in cls.CLF_TYPES:
            raise Exception("No such classifier type")

        clf_name = f"./models/{cls.lang}_model_{cls.algorithm}_{cls.clf_type}.sav"
        vectorizer_name = f"./models/{cls.lang}_tfidfvector_{cls.algorithm}_{cls.clf_type}.sav"

        cls.clf = cls.model.load_model(clf_name)
        cls.vectorizer = cls.model.load_vectorizer(vectorizer_name)

        if cls.lang == "uk":
            cls.nlp = uk_nlp
        elif cls.lang == "en":
            cls.nlp = en_nlp
        else:
            raise ValueError("This language is not supported")

    def _process_user_input(self, raw_text: str):
        """Normalize user input

        Args:
            raw_text (str): user input

        Returns:
            DataFrame: processed user input
        """
        if self.lang == "uk":
            self.normal_text = UKTextPreprocessing(
                raw_text, morph_analyzer=self.nlp).normalize()
        elif self.lang == "en":
            self.normal_text = TextPreprocessing(
                raw_text, nlp=self.nlp).normalize()
        self.user_input = self.model.transform_data(
            pd.DataFrame({"processedText": [self.normal_text]})
        )
        return self.user_input

    def make_prediction(self, user_text: str) -> dict:
        self._process_user_input(user_text)
        probas = self.clf.predict_proba(self.user_input).tolist()[0]
        classes = self.clf.classes_.tolist()
        class_proba_mapping = dict(zip(classes, probas))
        response = {
            "tag": self.clf.predict(self.user_input)[0],
            "probability_scores": class_proba_mapping
        }
        return response

    def explain(self):
        '''
        Explain the choice of a model
        '''
        if not hasattr(self, "user_input"):
            raise Exception("Cannot build Lime model without predicted output")
        c = make_pipeline(self.vectorizer, self.clf)
        explainer = LimeTextExplainer(class_names=self.model.clf.classes_)
        exp = explainer.explain_instance(
            self.normal_text, c.predict_proba, num_features=600)
        exp.save_to_file('sentiment/static/analysis.html')
        return exp

    def train(self, dataset_path="data/three_hope.csv",
              model_output_path="./models/model.sav",
              vectorizer_output_path="./models/vectorizer.sav"):
        '''TRAIN MODEL AND SAVE OUTPUT TO FILES'''
        start = time.perf_counter()
        self.model.new_model(dataset_path=dataset_path, balance=True)
        self.new_model = self.model.train(self.ALGORITHMS[self.algorithm], partial=False)
        self.model.save_model(model_output_path, self.new_model)
        self.model.save_vectorizer(vectorizer_output_path)
        end = time.perf_counter()

        print('Duration: {}'.format(end - start))

        return self.new_model

    def get_model_stats(self):
        '''
        Construct a confusion matrix for a new model.
        If the model was not trained, it will be
        '''
        if not hasattr(self, 'new_model'):
            self.train()
        predictions = self.new_model.predict(self.model.test_features)
        report = self.model.precision_report(predictions)
        for k, v in report.items():
            print(k, v)

        plot_confusion_matrix(
            self.new_model, self.model.test_features, self.model.test_labels)
        plt.show()
