from flask import Flask
from pymorphy2 import MorphAnalyzer
import spacy

uk_nlp = MorphAnalyzer(lang="uk")
en_nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)
app.config['SECRET_KEY'] = '4e9990c558c6b9fe12f2c6583b8cca63'

from sentiment import routes
