"""Functions for the simple NLP comparisons.
Raw text extraction, vectorization of plays into a single word, and comparisons"""

import os
import pickle
import re
from xml.dom import minidom

# from sklearn.feature_extraction.text import CountVectorizer
import spacy
from nltk.corpus import PlaintextCorpusReader
from scipy.spatial.distance import cosine

import play_parsing

# nltk has an XML corpus reader accessible here :
# 'https://www.nltk.org/_modules/nltk/corpus/reader/xmldocs.html#XMLCorpusReader' but it seems too basic Hence I'll
# convert XML docs to txt and then use their regular corpus reader


nlp_fr = spacy.load("fr_core_news_lg")
corpus = os.path.join('Corpus', 'Corpus Dracor - new')
corpusraw = 'Data\\Textes bruts Corpus\\Corpus Dracor'
stopwords_file = 'Data\\StoplistFrench.txt'
stopwords_doc = open(stopwords_file, 'r')
stopwords = list(stopwords_doc.read().splitlines())
output_dir = 'Outputs\\NLP - Comparisons on simple indicators'
data_dir = 'Data\\Pickled saves'


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
    """Parse the xml files and produce a raw txt version of the plays, only keeping the spoken text.
    Resulting text file is of the form LOCUTOR1 : text_said_by_locutor1 LOCUTOR2 : text_said_by_locutor2
    LOCUTOR1 : text_said_by_locutor1, etc."""
    failed = open('failed_encodings.txt', 'w')
    cp = os.path.join(play_parsing.folder, 'Corpus', corpus)
    for d in os.listdir(cp):
        d_file = os.path.join(cp, d)
        d_doc = minidom.parse(open(d_file, 'rb'))
        title = play_parsing.get_title(d_doc)
        print(title)
        d_text = play_parsing.get_full_text(d_doc)
        d_text = ' '.join([f'{remove_point(t[0])}:{t[1]}' for t in d_text])
        translation_dict = {'\x81': None, '\x84': None, '\x85': '...', '\x91': '\'', '\x93': '\"', '\x95': '•',
                            '\x96': '-', '\uff0c': ',', '\u0144': 'ń', '\u0173': 'ų',
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


def extract_text_per_speaker(corpus):
    failed = open('failed_encodings.txt', 'w')
    translation_dict = {'\x81': None, '\x84': None, '\x85': '...', '\x91': '\'', '\x93': '\"', '\x95': '•', '\x96': '-',
                        '\uff0c': ',', '\u0144': 'ń', '\u0173': 'ų',
                        '\u2015': '―'}
    cp = os.path.join(play_parsing.folder, 'Corpus', corpus)
    for d in os.listdir(cp):
        d_file = os.path.join(cp, d)
        d_doc = minidom.parse(open(d_file, 'rb'))
        title = play_parsing.get_title(d_doc)
        print(title)
        d_text = play_parsing.get_full_text(d_doc)
        text_by_speaker = dict()
        for (speaker, speaker_text) in d_text:
            speaker_text = speaker_text.translate(str.maketrans(translation_dict))
            if speaker not in text_by_speaker:
                text_by_speaker[speaker] = [speaker_text]
            else:
                text_by_speaker[speaker].append(speaker_text)
        pickle_filename = d.replace('.xml', '.pkl')
        saved_d_file = open(os.path.join('Textes par locuteur Dracor', f'{pickle_filename}'), 'wb')
        try:
            pickle.dump(text_by_speaker, saved_d_file)
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
    """Checks if a token is a stopword
    If stopword_list is provided, it is used as a reference for stopwords.
    Otherwise, uses spacy predefined list"""
    if stopword_list is None:
        is_stopword = tok.is_stop
    else:
        is_stopword = tok.text in stopword_list
    return not (is_stopword or tok.is_punct or tok.text.strip() == '')


def average_tokens(l):
    """Averages all tokens in the list to compute the document embedding"""
    n = len(l)
    return sum([token.vector.reshape(-1) for token in l]) / n


def vectorize_without_stopwords(corpus, stopwords, save_every=100, existing_save=None):
    """Converts txt plays into document embeddings and saves them regularly.

    Args:
        corpus (str): path to the corpus.
        stopwords (list): custom list of stopwords.
        save_every (int): number of documents to process before saving intermediate results.
    Returns:
        dict: A dictionary with plays and their 3 vectorizations.
    """
    FILE_PATTERN = r'.*\.txt'
    c_reader = PlaintextCorpusReader(corpus, FILE_PATTERN, encoding='utf-8')
    if existing_save is None:
        vec_dict = {}
    else:
        vec_dict = pickle.load(open(existing_save, 'rb'))
    file_number = 0

    for x in c_reader.fileids():
        file_number += 1
        print(f'{file_number} : {x}')
        if x in vec_dict:
            continue
        words = nlp_fr(c_reader.raw(x))
        doc_embedding = words.vector  # Vectorization with stopwords

        words_spacy = [token for token in words if not_stop(token)]  # Vectorization without stopwords (spacy list)
        words_manual = [token for token in words if
                        not_stop(token, stopwords)]  # Vectorization without stopwords (custom list)

        text_spacy = ' '.join([token.text for token in words_spacy])
        text_manual = ' '.join([token.text for token in words_manual])

        if words_manual and words_spacy:
            vect_spacy = nlp_fr(text_spacy).vector.reshape(-1)
            vect_manual = nlp_fr(text_manual).vector.reshape(-1)
            vec_dict[x] = {
                'Spacy': vect_spacy,
                'Manual': vect_manual,
                'Doc_embedding': doc_embedding
            }

        # Periodic save
        if file_number % save_every == 0:
            savefile = os.path.join(data_dir, f'vectorized_checkpoint_{file_number}_utf.pkl')
            with open(savefile, 'wb') as f:
                pickle.dump(vec_dict, f)
            print(f">>> Intermediate save at {file_number} documents.")

    # Final save
    final_savefile = os.path.join(data_dir, 'vectorized_dracor_doc2vec_utf.pkl')
    with open(final_savefile, 'wb') as f:
        pickle.dump(vec_dict, f)
    print(">>> Final save complete.")

    return vec_dict

def generate_three_words_resumes(d, top_n=3,
                                 output_file='Outputs\\NLP - Comparisons on simple indicators\\words_for_document_embeddings.csv'):
    """Computes the n closest words for each embedding, and saves the result in a csv."""

    def cosine_similarity(a, b):
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(a_norm, b_norm.T)

    # Prepare vocab keys and vectors only once
    keys = list(nlp_fr.vocab.vectors.keys())
    vectors = nlp_fr.vocab.vectors.data
    string_lookup = nlp_fr.vocab.strings

    rows = []

    for play_name in d:
        print(play_name)
        row = {'play_name': play_name}
        embeddings = {
            'doc': d[play_name]['Doc_embedding'],
            'spacy': d[play_name]['Spacy'],
            'manual': d[play_name]['Manual']
        }

        for method, emb in embeddings.items():
            doc_vector = emb.reshape(1, -1)
            sims = cosine_similarity(doc_vector, vectors)
            top_indices = sims[0].argsort()[-top_n:][::-1]
            top_words = [string_lookup[keys[i]] for i in top_indices]
            for i, word in enumerate(top_words):
                row[f'{method}_word_{i + 1}'] = word

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f'Data saved to {output_file}')
    return df


def compute_distances(d):
    """Computes all pairwise distances of the embeddings, and saves them in a csv file.

    Warning : the resulting CSV is huge.

    Args:
        d(dict): The dictionnary with all the pre-computed embeddings
    Returns:
        None """
    distance_dict = dict()
    seen = set()
    for play1 in d:
        print(play1)
        seen.add(play1)
        spacy_vector = d[play1]['Spacy']
        manual_vector = d[play1]['Manual']
        doc_vector = d[play1]['Doc_embedding']
        if len(spacy_vector) != 0:
            for play2 in d:
                if play2 not in seen and len(d[play2]['Spacy']) != 0:
                    spacy_vector2 = d[play2]['Spacy']
                    manual_vector2 = d[play2]['Manual']
                    doc_vector2 = d[play2]['Doc_embedding']
                    spacy_dist = cosine(spacy_vector, spacy_vector2)
                    manual_dist = cosine(manual_vector, manual_vector2)
                    doc_dist = cosine(doc_vector, doc_vector2)
                    distance_dict[(play1, play2)] = {'Spacy': spacy_dist, 'Manual': manual_dist,
                                                     'Doc_embedding': doc_dist}
    savefile = os.path.join(data_dir, 'distance_vectors_doc2vec_dracor_new.pkl')
    with open(savefile, 'wb') as f:
        pickle.dump(distance_dict, f)


def find_most_similar_plays(d):
    """For each play in the corpus d, find the most similar, using cosine distance between doc embeddings.

    The result is saved in a csv. A closest play is computed for each vectorization. """
    d = {re.sub(r'\.xml_raw\.txt', '', play): d[play] for play in d}
    distance_dict = {play: {method: {'distance': float('inf'), 'closest play': None}
                            for method in ['Spacy', 'Manual', 'Doc_embedding']}
                     for play in d}
    seen = set()
    i = 0
    for play1 in d:
        print(f'{i} : {play1}')
        i += 1
        seen.add(play1)
        vectors_1 = {method: d[play1][method] for method in {'Spacy', 'Manual', 'Doc_embedding'}}
        for play2 in d:
            if play2 not in seen:
                vectors_2 = {method: d[play2][method] for method in {'Spacy', 'Manual', 'Doc_embedding'}}
                for method in vectors_1:
                    method_dist = cosine(vectors_1[method], vectors_2[method])
                    if method_dist < distance_dict[play1][method]['distance']:
                        distance_dict[play1][method]['distance'] = round(method_dist, 4)
                        distance_dict[play1][method]['closest play'] = play2
                    if method_dist < distance_dict[play2][method]['distance']:
                        distance_dict[play2][method]['distance'] = round(method_dist, 4)
                        distance_dict[play2][method]['closest play'] = play1
        #print(f'most similar is {distance_dict[play1]}')
    savefile = os.path.join(data_dir, 'closest_plays_doc2vec_dracor_new.pkl')
    with open(savefile, 'wb') as f:
        pickle.dump(distance_dict, f)


def denoise_title(s):
    t1, t2 = s[0], s[1]
    t1 = t1.replace('_brut.txt', '')
    t2 = t2.replace('_brut.txt', '')
    return t1, t2


if __name__ == "__main__":
    # # To compute the vectorization of all plays :
    # vectorize_without_stopwords(corpusraw, stopwords, existing_save='Data\\Pickled saves\\vectorized_checkpoint_200_utf.pkl')

    # To compute the closest words for each embedding
    # print('Loading ...')
    # d = pickle.load(open('Data\\Pickled saves\\vectorized_dracor_doc2vec_utf.pkl', 'rb'))
    # print('Loaded')

    # # To compute the closest words for each embedding
    # generate_three_words_resumes(d)

    # # To compute the closest play for each play
    # find_most_similar_plays(d)
    print('Loading ...')
    savefile = os.path.join(data_dir, 'closest_plays_doc2vec_dracor_new.pkl')
    d = pickle.load(open(savefile, 'rb'))
    print('Loaded')

    # Prepare the data for DataFrame
    rows = []
    for play, methods in d.items():
        row = {'Play': play}
        for method, values in methods.items():
            method_name = method
            row[f'Closest play {method_name}'] = values['closest play']
            row[f'Closest distance {method_name}'] = values['distance']
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Optional: Save to CSV
    df.to_csv('distance_results.csv', index=False)
    df.to_csv(f'{output_dir}\\closest_play_by_doc_embedding.csv', index=False, encoding='utf8')

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
    #     row_to_merge = {'Play 1': p1, 'Play 2': p2, 'Distance_spacy': overall_dict[(p1,p2)]['Distance_spacy'], 'Distance_manual':overall_dict[(p1,p2)]['Distance_manual']}
    #     gwriter.writerow(row_to_merge)
