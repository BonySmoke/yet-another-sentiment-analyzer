import spacy
from spacy.language import Language
from textblob import TextBlob
from pymorphy2 import MorphAnalyzer
import re

class TextPreprocessing:

    GARBAGE_TOKENS = "~`!@#$%^&*()_-+={[}]|\:;'<,>.?/"
    CONJ_TYPES = ['CONJ', 'SCONJ', 'CCONJ']
    SENTIMENT_POS = ['ADJ', 'VERB']
    NEGATIVE_PREFIX = 'not_'

    def __init__(self, text: str,
                 nlp: Language = None,
                 expected_sentiment: int = None,
                 token_limit: int = None) -> None:
        self.original_text = text.lower()
        self.expected_sentiment = expected_sentiment if expected_sentiment else None
        self.token_limit = token_limit if token_limit else 0

        self.nlp = nlp if nlp else spacy.load("en_core_web_sm")

        self._doc = self.nlp(self.original_text)
        self.tokens = self.__construct_text_dict()

    def __construct_text_dict(self) -> dict:
        """Create an object containing text and id referencing the doc"""
        self.text_dict = dict()
        for token in self._doc:
            # remove punctuation and extra spaces
            if token.is_punct or token.pos_ == "SPACE":
                continue
            self.text_dict[token.i] = token.text

        # decrease the number of tokens to token_limit
        # TODO: TO BE RECONSIDERED AS IT IS TOO NAIVE
        if self.token_limit:
            self.text_dict = {k:v for i, (k,v) in enumerate(self.text_dict.items())
                                if i <= self.token_limit}
        return self.text_dict

    def _dict_to_text(self):
        self.text = re.sub(' +', ' ', " ".join([t for t in self.tokens.values()]))
        return self.text

    def negate_pos(self, token: str) -> str:
        """Negate word by adding the not_ prefix
        Returns:
            token (str): the modified token
        """
        return f'{self.NEGATIVE_PREFIX}{token}'

    def split_sents_to_clauses(self, tokens: list, ids: list) -> list:
        """Split a list of tokens to clauses

        Args:
            tokens (list): spaCy Token objects
            ids (list): The IDs of conjuction words

        Returns:
            list: a list of clauses
        """
        clauses = list()

        for index, c_id in enumerate(ids):
            # first element
            if (index - 1 < 0):
                # append from 0 to id
                clauses.append(tokens[:c_id])
            # last element
            elif (index+1 >= len(ids)):
                # append from prev id to the current one
                clauses.append(tokens[ids[index-1]:c_id])
                # append from current to -1
                clauses.append(tokens[c_id:])
            # between elements
            else:
                # append from prev id to the current one
                clauses.append(tokens[ids[index-1]:c_id])

        self.clauses = clauses
        return self.clauses

    def negate_sequences(self) -> list:
        """Convert sentences with negation to negative sentiment
        Algorithm:
        1. Split the text.
        2. Make a POS analysis.
        3. Find if the negation label is present.
        3.1 If not, return tokenized text.
        3.2 If yes, check if there are multiple clauses.
            4.1 If not, negate tokenized text
            4.2 If yes, negate each clause in the text.

        Args:
            sample ([str]): The text to be checked for negation

        Returns:
            sample_tokens ([list]): a list of tokens with negation
        """
        tokens = [str(t.text) for t in self._doc]

        has_negation = next((True for token in self._doc if token.dep_ == 'neg'), False)
        if has_negation:
            conj_ids = [token.i for token in self._doc if token.pos_ in self.CONJ_TYPES]
            # if there is only one clause
            if not conj_ids:
                for token in self.tokens:
                    if self._doc[token].pos_ in self.SENTIMENT_POS:
                        self.tokens[token] = self.negate_pos(self._doc[token].text)
                return self._dict_to_text()

            clauses = self.split_sents_to_clauses(self._doc, conj_ids)
            for clause in clauses:
                is_negative_clause = [True for tkn in clause if tkn.dep_ == 'neg']
                if not is_negative_clause:
                    continue

                for token in clause:
                    if token.pos_ in self.SENTIMENT_POS:
                        self.tokens[token.i] = self.negate_pos(token.text)
            return self._dict_to_text()

        return self._dict_to_text()

    def remove_garbage(self):
        self.text = self._dict_to_text()
        self.text = "".join([char for char in self.text if char not in self.GARBAGE_TOKENS])
        return self.text

    def remove_stop_words(self):
        all_stopwords = self.nlp.Defaults.stop_words
        for key, value in self.tokens.copy().items():
            if value in all_stopwords:
                del self.tokens[key]
        return self._dict_to_text()

    def _clean_review_noise(self):
        """Many reviews with a specific sentiment contain noise words
        that contradict the actual rating. For example, a neutral review
        may contain negative and positive adjectives. This complicates
        the process of estimating text sentiment analysis.
        """
        for token in self.tokens.copy():
            if (self._doc[token].pos_ in self.SENTIMENT_POS
                and not self.tokens[token].startswith(self.NEGATIVE_PREFIX)):

                word_polarity = TextBlob(self._doc[token].lemma_).polarity
                if (self.expected_sentiment > 3
                    and self.is_negative(word_polarity)):
                    del self.tokens[token]
                if (self.expected_sentiment == 3
                    and self.is_not_neutral(word_polarity)):
                    del self.tokens[token]
                if (self.expected_sentiment < 3
                    and self.is_positive(word_polarity)):
                    del self.tokens[token]

    @staticmethod
    def is_negative(word_polarity) -> bool:
        return word_polarity < -0.1

    @staticmethod
    def is_not_neutral(word_polarity) -> bool:
        return -0.3 > word_polarity > 0.5

    @staticmethod
    def is_positive(word_polarity):
        return word_polarity > 0.1

    def lemmatize(self):
        for token in self.tokens:
            if self.tokens[token].startswith(self.NEGATIVE_PREFIX):
                continue
            self.tokens[token] = self._doc[token].lemma_
        self.text = self._dict_to_text()
        return self.text

    def pos_tag(self):
        """Append a POS-tag to the word

        Returns:
            [type]: [description]
        """
        for token in self.tokens:
            self.tokens[token] += f'_{self._doc[token].pos_}'
        self.text = self._dict_to_text()
        return self.text

    def normalize(self):
        self.negate_sequences()
        self.remove_stop_words()
        self.lemmatize()
        self.remove_garbage()
        self.pos_tag()
        if self.expected_sentiment:
            self._clean_review_noise()
        return self._dict_to_text()


class UKTextPreprocessing:

    GARBAGE_TOKENS = "~`!@#$%^&*()_-+={[}]|\:;'<,>.?/"

    def __init__(self, text: str, morph_analyzer: MorphAnalyzer = None) -> None:
        self.text = text.lower()
        self.analyzer = morph_analyzer if morph_analyzer else MorphAnalyzer(lang="uk")

        self._construct_text_dict()

    def _construct_text_dict(cls):
        # remove garbage tokens
        cls.text = "".join([char for char in cls.text
                        if char not in cls.GARBAGE_TOKENS])

        cls.tokens = cls.text.split()

    def _to_text(cls):
        return " ".join(cls.tokens)

    def _load_ukrainian_stopwords(cls):
        with open("./data/ukrainian_stopwords.txt", "r", encoding="utf-8") as f:
            cls.stopwords = [word.strip() for word in f]

    def lemmatize(self):
        """Get the normal form of each word

        Returns:
            tokens: normalized tokens
        """
        for i in range(len(self.tokens)):
            parsed_token = self.analyzer.parse(self.tokens[i])
            if not parsed_token:
                continue
            self.tokens[i] = parsed_token[0].normal_form

        return self._to_text()

    def remove_stopwords(self):
        if not hasattr(self, 'stopwords'):
            self._load_ukrainian_stopwords()

        self.tokens = [t for t in self.tokens
                       if t not in self.stopwords]

        return self._to_text()

    def normalize(self):

        self.lemmatize()
        self.remove_stopwords()

        return self._to_text()
