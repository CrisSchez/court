import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import spacy
from spacy.lang.de.examples import sentences 

nlp = spacy.load("de_core_news_sm",disable=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat'])

def predict(args):
  entrada=np.array(list(args.values()))
  
  pickle_in = open("le.pickle","rb")
  le = pickle.load(pickle_in)
  pickle_in = open("mlp.pickle","rb")
  mlp = pickle.load(pickle_in)


  
  entrada2=nlp(str(entrada[0])).vector
  return le.inverse_transform(mlp.predict(entrada2.reshape(1,-1)))[0]