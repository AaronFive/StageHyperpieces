import os
import pickle
from basic_utilities import PICKLE_DIR, pickle_file
from play_parsing import get_play_from_file, get_dracor_id, get_full_text
from utils import get_title


def get_title_dict():
    path = os.path.join(PICKLE_DIR, 'id_title_correspondance.pkl')
    file = open(path, 'rb')
    title_dict = pickle.load(file)
    return title_dict

def get_vectorized_corpus():
    path = os.path.join(PICKLE_DIR, 'vectorized_corpus.pkl')
    file = open(path, 'rb')
    vectorized_corpus = pickle.load(file)
    return vectorized_corpus

def get_tokenized_corpus():
    path = os.path.join(PICKLE_DIR, 'tokenized_plays_dracor.pkl')
    file = open(path, 'rb')
    tokenized_corpus = pickle.load(file)
    return tokenized_corpus

def get_full_text_corpus():
    path = os.path.join(PICKLE_DIR, 'full_plays_dracor.pkl')
    full_text_file = open(path, 'rb')
    full_text_corpus = pickle.load(full_text_file)
    return full_text_corpus

def make_full_text_corpus(corpusFolder):
    full_plays = dict()
    for file in os.listdir(corpusFolder):
        play = get_play_from_file(os.path.join(corpusFolder, file))
        title = get_title(play)
        print(title)
        identifier = get_dracor_id(play)
        full_text = get_full_text(play)
        full_plays[identifier] = full_text
    pickle_file(full_plays, "full_plays_dracor")
    return full_plays


