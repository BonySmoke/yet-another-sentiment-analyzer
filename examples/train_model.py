'''
Train a model and get a confusion matrix for it
'''
import sys
sys.path.insert(1, '../')
from sentiment.processor import Process

process = Process(algorithm="linear", mode="train")
process.train(dataset_path="../data/fine_uk.csv",
model_output_path="../models/uk_model_linear_fine.sav",
              vectorizer_output_path="../models/uk_tfidfvector_linear_fine.sav")
process.get_model_stats()
