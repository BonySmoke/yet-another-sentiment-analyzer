'''
Train a model and get a confusion matrix for it
'''
import sys
sys.path.insert(1, '../')
from sentiment.processor import Process

process = Process(algorithm="linear", mode="train")
process.train(dataset_path="../data/three_hope.csv",
model_output_path="../models/en_model_linear_three.sav",
              vectorizer_output_path="../models/en_tfidfvector_linear_three.sav")
process.get_model_stats()
