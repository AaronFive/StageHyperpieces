from sklearn.feature_extraction.text import CountVectorizer
import spacy
from xml.dom import minidom
import os
import play_parsing
import pickle
from scipy.spatial.distance import cosine
import csv
import numpy as np
from nltk.corpus import PlaintextCorpusReader

# nltk has an XML corpus reader accessible here :
# 'https://www.nltk.org/_modules/nltk/corpus/reader/xmldocs.html#XMLCorpusReader' but it seems too basic Hence I'll
# convert XML docs to txt and then use their regular corpus reader


nlp_fr = spacy.load("fr_core_news_lg")

corpus = 'corpusDracor'
corpusraw = 'Texte bruts Dracor'
stopwords_file = 'StoplistFrench.txt'
stopwords_doc = open(stopwords_file, 'r')
stopwords = list(stopwords_doc.read().splitlines())


def clean_title(s):
    """Removes characters from s that can't be used in a filename"""
    illegal_characters = ['<', '>', '.', ':', '"', '/', '\\', '|', '?', '*']
    new_title = ''
    for x in s:
        if x not in illegal_characters:
            new_title = new_title + x
    return new_title


def remove_point(s):
    if s and s[-1] == '.':
        return s[:-1]
    else:
        return s


def extract_texts_corpus(corpus):
    """Parse the xml files and produce a raw txt version of the plays, only keeping the spoken text"""
    failed = open('failed_encodings.txt', 'w')
    cp = os.path.join(play_parsing.folder, 'Corpus', corpus)
    for d in os.listdir(cp):
        d_file = os.path.join(cp, d)
        d_doc = minidom.parse(open(d_file, 'rb'))
        title = play_parsing.get_title(d_doc)
        print(title)
        d_text = play_parsing.get_full_text(d_doc)
        d_text = ' '.join([f'{remove_point(t[0])}:{t[1]}' for t in d_text])
        translation_dict = {'\x81': None, '\x84':None,'\x85': '...', '\x91': '\'', '\x93': '\"','\x95' :'•', '\x96': '-', '\uff0c': ',','\u0144':'ń', '\u0173':'ų',
                            '\u2015': '―'}
        d_text = d_text.translate(str.maketrans(translation_dict))
        raw_filename = d.replace('.xml', '')
        saved_d_file = open(os.path.join('Texte bruts Dracor', f'{raw_filename}_brut.txt'), 'w')
        try:
            saved_d_file.write(d_text)
        except UnicodeEncodeError as e:
            failed.write(f'{title} : {e} \n')
            print('Fail')
        saved_d_file.close()


# def compute_doc_value(cp):
#     """Convert txt plays into vector embeddings"""
#     vec_dict = {}
#     save = open('pickled_vectors', 'wb')
#     for doc in os.listdir(cp):
#         txt = open(os.path.join(cp, doc), 'r').read()
#         v1 = nlp_fr(txt).vector.reshape(-1)
#         vec_dict[doc] = v1
#         print(doc)
#     pickle.dump(vec_dict, save)
#     save.close()


def not_stop(tok, stopword_list=None):
    if stopword_list is None:
        is_stopword = tok.is_stop
    else:
        is_stopword = tok.text in stopword_list
    return not (is_stopword or tok.is_punct or tok.text.strip() == '')


def average_tokens(l):
    n = len(l)
    return sum([token.vector.reshape(-1) for token in l])/n


def vectorize_without_stopwords(corpus, stopwords):
    """Convert txt plays into vector embeddings, removing stopwords and punctuation, once with spacy built-in method,
    once with a custom list"""
    FILE_PATTERN = r'.*\.txt'
    c_reader = PlaintextCorpusReader(corpus, FILE_PATTERN, encoding='latin-1')
    vec_dict = {}
    for x in c_reader.fileids():
        print(x)
        words = nlp_fr(c_reader.raw(x))
        words_spacy = [token for token in words if not_stop(token)]
        words_manual = [token for token in words if not_stop(token, stopwords)]

        text_spacy = ' '.join([x.text for x in words_spacy])
        text_manual = ' '.join([x.text for x in words_manual])
        if words_manual and words_spacy:
            vect_spacy = nlp_fr(text_spacy).vector.reshape(-1)
            vect_manual = nlp_fr(text_manual).vector.reshape(-1)
            vec_dict[x] = {'Spacy': vect_spacy, 'Manual': vect_manual}
    save = open('vectorized_texts.pkl', 'wb')
    pickle.dump(vec_dict, save)
    save.close()


def generate_two_words_resumes(d):
    i = 0
    for x in d:
        i+=1
        vect_x = d[x]['Spacy']
        word = nlp_fr.vocab.vectors.most_similar(vect_x.reshape(1,-1), n=3)
        three_words = 
        word = nlp_fr.vocab[word[0][0]].text
        print(x, word)

def compute_distances(d):
    distance_dict = dict()
    seen = set()
    for play1 in d:
        print(play1)
        seen.add(play1)
        spacy_vector = d[play1]['Spacy']
        manual_vector = d[play1]['Manual']
        if len(spacy_vector) != 0:
            for play2 in d:
                if play2 not in seen and len(d[play2]['Spacy']) != 0:
                    spacy_vector2 = d[play2]['Spacy']
                    manual_vector2 = d[play2]['Manual']
                    spacy_dist = cosine(spacy_vector, spacy_vector2)
                    manual_dist = cosine(manual_vector, manual_vector2)
                    distance_dict[(play1, play2)] = {'Spacy': spacy_dist, 'Manual': manual_dist}
    savefile = open('distances_vectors.pkl', 'wb')
    pickle.dump(distance_dict, savefile)


def denoise_title(s):
    t1, t2 = s[0], s[1]
    t1 = t1.replace('_brut.txt', '')
    t2 = t2.replace('_brut.txt', '')
    return t1, t2


if __name__ == "__main__":
    d = pickle.load(open('vectorized_texts.pkl','rb'))
    generate_two_words_resumes(d)

    #extract_texts_corpus(corpus)
    # vectorize_without_stopwords(corpusraw, stopwords)
    # d = pickle.load(open('vectorized_texts.pkl', 'rb'))
    # compute_distances(d)
    # f = open('distances_vectors.pkl', 'rb')
    # d = pickle.load(f)
    # l_spacy = [(x, d[x]['Spacy']) for x in d]
    # l_manual = [(x, d[x]['Manual']) for x in d]
    # size1, size2 = len(l_spacy),len(l_manual)
    # l_spacy.sort(key=(lambda x: x[1]))
    # l_manual.sort(key=(lambda x: x[1]))
    # l_spacy = l_spacy[:int(size1/3)]
    # l_manual = l_manual[:int(size2/3)]
    # print('sort ok')
    # output = open('close_plays_by_embedding.csv', 'w')
    # fieldnames = ['Play 1', 'Play 2', 'Distance_spacy', 'Distance_manual']
    # gwriter = csv.DictWriter(output, fieldnames=fieldnames)
    # gwriter.writeheader()
    # overall_dict = dict()
    # for x in l_spacy:
    #     print(f'1 - {x}')
    #     couple_name = x[0]
    #     distance_spacy = x[1]
    #     p1, p2 = denoise_title(couple_name)
    #     overall_dict[(p1,p2)] = {'Distance_spacy':distance_spacy,'Distance_manual': None }
    # for x in l_manual:
    #     print(f'2 - {x}')
    #     couple_name = x[0]
    #     distance_manual = x[1]
    #     p1, p2 = denoise_title(couple_name)
    #     if (p1,p2) in overall_dict:
    #         overall_dict[(p1,p2)]['Distance_manual'] = distance_manual
    #     else:
    #         overall_dict[(p1, p2)] = {'Distance_spacy': None, 'Distance_manual': distance_manual}
    # print('dicts done')
    # for (p1, p2) in overall_dict:
    #     row = {'Play 1': p1, 'Play 2': p2, 'Distance_spacy': overall_dict[(p1,p2)]['Distance_spacy'], 'Distance_manual':overall_dict[(p1,p2)]['Distance_manual']}
    #     gwriter.writerow(row)
