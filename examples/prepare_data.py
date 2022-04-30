'''
Prepare data for model training
'''
import sys
from pymorphy2 import MorphAnalyzer
sys.path.insert(1, '../')
from sentiment.model import Analizer
# nlp = spacy.load("en_core_web_sm") # English
nlp = MorphAnalyzer(lang="uk") # Ukrainian
a = Analizer()
a.read_data_dev(size=50000, skiprows=0, balance=False, maximum_tokens=170, is_json=False,
                dataset_path="../data/fine_uk.csv")
a.prepare_data_dev(nlp=nlp, lang="uk")
a.save_dataframe(a.data, "../data/fine_uk.csv")
