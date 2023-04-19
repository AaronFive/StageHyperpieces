from sklearn.feature_extraction.text import CountVectorizer
import spacy
from xml.dom import minidom
import os
import play_parsing
import pickle
from scipy.spatial.distance import cosine
import csv


# nlp_fr_sm = spacy.load("fr_core_news_sm")


def clean_title(s):
    illegal_characters = ['<', '>', '.', ':', '"', '/', '\\', '|', '?', '*']
    new_title = ''
    for x in s:
        if x not in illegal_characters:
            new_title = new_title + x
    return new_title


def extract_texts_corpus(corpus):
    failed = open('failed_encodings.txt', 'w+')
    cp = os.path.join(play_parsing.folder, 'Corpus', corpus)
    for d in os.listdir(cp):
        d_file = os.path.join(cp, d)
        d_doc = minidom.parse(open(d_file, 'rb'))
        title = play_parsing.get_title(d_doc)
        print(title)
        d_text = play_parsing.get_full_text(d_doc)
        d_text = ' '.join([f'{t[0]}:{t[1]}' for t in d_text])
        raw_filename = d.replace('.xml', '')
        saved_d_file = open(os.path.join('Texte bruts Dracor', f'{raw_filename}_brut.txt'), 'w+')
        try:
            saved_d_file.write(d_text)
        except UnicodeEncodeError as e:
            failed.write(f'{title} : {e} \n')
            print('Fail')
        saved_d_file.close()


def compute_doc_value(cp):
    vec_dict = {}
    save = open('pickled_vectors', 'wb')
    for doc in os.listdir(cp):
        txt = open(os.path.join(cp, doc), 'r').readlines()
        txt = ''.join(txt)
        v1 = nlp_fr_sm(txt).vector.reshape(-1)
        vec_dict[doc] = v1
        print(doc)
    pickle.dump(vec_dict, save)
    save.close()


def compute_distances(d):
    savefile = open('distances_vectors', 'wb')
    distance_dict = dict()
    for x in d:
        print(x)
        if len(d[x]) != 0:
            for y in d:
                if x != y and (y, x) not in d and len(d[y]) != 0:
                    dist = cosine(d[x], d[y])
                    distance_dict[(x, y)] = dist
    pickle.dump(distance_dict, savefile)


def denoise_title(s):
    t1, t2 = s[0], s[1]
    t1 = t1.replace('_brut.txt', '')
    t2 = t2.replace('_brut.txt', '')
    return t1, t2


if __name__ == "__main__":
    f = open('distances_vectors', 'rb')
    d = pickle.load(f)
    new_d = dict()
    for (x, y) in d:
        if (y, x) not in new_d and (x,y) not in new_d:
            new_d[(x,y)] = d[(x,y)]
    f.close()
    d = new_d
    print('del ok')
    savefile = open('distances_vectors', 'wb')
    l = [(x, d[x]) for x in d]
    l.sort(key=(lambda x: x[1]))
    pickle.dump(d, savefile)
    file = open('sorted_vector_dict', 'wb')
    pickle.dump(l, file)
    print('dump sort ok')
    output = open('close_plays_by_embedding.csv', 'w')
    fieldnames = ['Play 1', 'Play 2', 'Distance']
    gwriter = csv.DictWriter(output, fieldnames=fieldnames)
    gwriter.writeheader()
    for x in l:
        couple_name = x[0]
        distance = x[1]
        p1, p2 = denoise_title(couple_name)
        row = {'Play 1': p1, 'Play 2': p2, 'Distance': distance}
        gwriter.writerow(row)

