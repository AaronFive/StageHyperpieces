import math
import random

import play_parsing
from play_parsing import corpus, get_all_scenes_dialogues,
import os, sys
from xml.dom import minidom
import matplotlib.pyplot as plt
import numpy as np
from Conversion_and_scraping import downloadDracor
import pickle
from queue import Queue


def normalize_scene(scene, return_dict = False):
    """Given a list of characters, transforms it in a parameterized word of the form ABABC"""
    character_normalizing = dict()
    order = 65
    normalized_scene = []
    for x in scene:
        if x not in character_normalizing:
            character_normalizing[x] = chr(order)
            order += 1
        normalized_scene.append(character_normalizing[x])
    if return_dict:
        return "".join(normalized_scene), character_normalizing
    else:
        return "".join(normalized_scene)


def get_normalized_scenes(doc):
    """Given a play doc, returns the list of its scenes as parameterized words"""
    scenes = get_all_scenes_dialogues(doc)
    return [normalize_scene(scene) for scene in scenes]


def count_scenes(c, min_date = 1600, max_date=1636, allowed_genres = None ):
    normalized_scenes_dict = dict()
    problematic_pieces = []
    for doc in os.listdir(c):
        play = minidom.parse(open(os.path.join(corpus, doc), 'rb'))
        date = play_parsing.get_date(play)
        title = play_parsing.get_title(play)
        genre = play_parsing.get_genre(play)
        print(play_parsing.get_title(play))
        if date != '' and min_date <= int(date) <= max_date and (allowed_genres is None or genre in allowed_genres):
            n_scenes = get_normalized_scenes(play)
            for (i, x) in enumerate(n_scenes):
                # if has_consecutive_letters(x):
                #     print(x, title, f"Scene {i + 1}")
                #     problematic_pieces.append(title)
                normalized_scenes_dict[x] = normalized_scenes_dict.get(x, 0) + 1
    return normalized_scenes_dict


def count_subscenes(c, subscene_length, min_date = 1550, max_date=1650, allowed_genres = None, renormalize = True):
    normalized_scenes_dict = dict()
    for doc in os.listdir(c):
        play = minidom.parse(open(os.path.join(corpus, doc), 'rb'))
        date = play_parsing.get_date(play)
        genre = play_parsing.get_genre(play)
        if date != '' and min_date <= int(date) <= max_date and (allowed_genres is None or genre in allowed_genres):
            print(play_parsing.get_title(play))
            n_scenes = get_normalized_scenes(play)
            for (i, scene) in enumerate(n_scenes):
                if len(scene) >= subscene_length:
                    start, stop = 0, subscene_length
                    for _ in range(len(scene) - subscene_length + 1):
                        subscene = scene[start:stop]
                        if renormalize:
                            subscene = normalize_scene(subscene)
                        normalized_scenes_dict[subscene] = normalized_scenes_dict.get(subscene, 0) + 1
                        start += 1
                        stop += 1
    return normalized_scenes_dict


def close_scenes(word):
    res = set()
    n = len(word)
    if n == 0:
        return ['A']

    max_ord = 65
    for x in word:
        max_ord = max(ord(x), max_ord)

    previous_letter, next_letter = None, word[0]
    seen_letters = set()
    current_ord = 64  # We have seen letters up to this ord

    for i in range(n + 1):

        # Inserting an already seen letter
        for l in seen_letters:
            if l != previous_letter and l != next_letter:
                new_word = "".join([word[:i], l, word[i:]])
                res.add(new_word)
                print(f"Position {i} seen letter {l} : {new_word}")

        # Inserting a brand new letter
        new_letter = chr(current_ord + 1)
        new_current_ord = current_ord + 1
        new_word = list(word[:i])
        new_word.append(new_letter)
        for x in word[i:]:
            if ord(x) >= new_current_ord:  # All following lettters are shifted by 1 if they were new letters
                new_word.append(chr(ord(x) + 1))
            else:
                new_word.append(x)
        new_word = "".join(new_word)
        res.add(new_word)
        print(f"Position {i} new letter : {new_word}")

        # Inserting a letter that'll appear later but hasn't occured already
        for future_ord in range(current_ord + 1,
                                max_ord + 1):  # future_ord is the letter that we are inserting that is coming later
            if future_ord != ord(next_letter):
                new_letter = chr(current_ord + 1)
                new_current_ord = current_ord + 1
                new_word = list(word[:i])
                new_word.append(new_letter)
                for x in word[i:]:
                    if ord(x) < new_current_ord:
                        new_word.append(x)
                    elif ord(x) == future_ord:
                        new_word.append(chr(new_current_ord))
                    else:  # TODO : Have to check if we are before the first occurence of the future letter or not, in which case, should do +1
                        new_word.append(chr(ord(x) + 1))
                new_word = "".join(new_word)
                res.add(new_word)
                print(f"Position {i} future letter {chr(future_ord)} : {new_word}")

        if i < n:
            seen_letters.add(word[i])
            current_ord = len(seen_letters) + 64
            previous_letter = word[i]
        if i + 1 < n:
            next_letter = word[i + 1]

    return res


def has_consecutive_letters(word):
    last_letter = None
    for x in word:
        if x == last_letter:
            return True
        else:
            last_letter = x
    return False


def random_word(length, nb_letters):
    if nb_letters == 1:
        return 'A'
    letters = {chr(65 + i) for i in range(nb_letters)}
    previous_letter = None
    word = []
    x = None
    for k in range(length):
        while x == previous_letter:
            x = random.sample(letters, 1)[0]
        word.append(x)
        previous_letter = x
    return "".join(word)


def check_well_formed(alphabet_size, word):
    last_letter = None
    alphabet = set()
    for x in word:
        if x == last_letter:
            return False
        else:
            alphabet.add(x)
            last_letter = x
    return alphabet_size == len(alphabet)


def random_words_repartition(nb_letters, length, nb_words, word_condition):
    total_words = 0
    res = dict()
    while total_words < nb_words:
        w = random_word(length, nb_letters)
        if word_condition(nb_letters, w):
            total_words += 1
            res[w] = res.get(w, 0) + 1
    return res


def make_graph_no_borns(dic, graph_name):
    labels, occurences = [], []
    for x in dic:
        labels.append(x)
        occurences.append(dic[x])
    plt.xticks(range(len(dic)), labels, rotation=45)
    plt.bar(labels, occurences)
    plt.savefig(graph_name)
    plt.show()


def make_graph(dic, expected_value, borns, graph_name):
    labels, occurences = [], []
    for x in dic:
        labels.append(x)
        occurences.append(dic[x])
    plt.xticks(range(len(dic)), labels, rotation=45)
    plt.bar(labels, occurences)
    plt.plot(labels, [expected_value for _ in range(len(labels))], 'red')
    plt.plot(labels, [expected_value + borns for _ in range(len(labels))], 'r--')
    plt.plot(labels, [expected_value - borns for _ in range(len(labels))], 'r--')
    plt.savefig(graph_name)
    plt.show()


def make_fixed_size_graph(d, word_length, min_letter_number, simple = False, max_letter_number=None):
    if max_letter_number is None:
        max_letter_number = min_letter_number
    min_letter = chr(64 + min_letter_number)
    excluded_letter = chr(64 + max_letter_number + 1)
    d = {k: v for k, v in sorted(d.items(), key=lambda item: - item[1])}
    longueur_k = dict()
    total = 0
    for x in d:
        if min_letter in x and excluded_letter not in x and len(x) == word_length and not has_consecutive_letters(
                x):
            longueur_k[x] = d[x]
            total += d[x]
    nb_of_diff_words = len(longueur_k)
    if not simple:
        expected_value = total / nb_of_diff_words
        borns = 10 * math.sqrt(total * (nb_of_diff_words - 1)) / (
                    nb_of_diff_words * math.sqrt(5))  # Ecart de confiance à 95%, calculs à vérifier (Tchebytchev)
        borns = math.sqrt(20*total*(nb_of_diff_words-1))/nb_of_diff_words
        make_graph(longueur_k, expected_value, borns, f"Size {word_length} with {min_letter_number} letters")
    else:
        make_graph_no_borns(longueur_k,f"Size {word_length} with {min_letter_number} letters no borns")


def get_deviation_from_main_value(d):
    """Given a dictionnary d, returns the sum of all values except the biggest one"""
    m = 0
    dev = 0
    for x in d:
        if d[x] < m:
            dev += d[x]
        else:
            dev += m
            m = d[x]
    return dev


def number_of_factors(w):
    factors = {w[i:i+2] for i in range(len(w)-1)}
    return factors


def regularity(word):
    """TODO"""
    succesions = dict()
    i = 0
    while i <= len(word) - 2:
        letter, next_letter = word[i], word[i+1]
        if letter in succesions:
            succesions[letter][next_letter] = succesions[letter].get(next_letter, 0) + 1
        else:
            succesions[letter] = {next_letter : 1}
        i += 1
    score = 0
    for letter in succesions:
        score += get_deviation_from_main_value(succesions[letter])
    return score


if __name__ == "__main__":
    # marianne_pwords = get_normalized_scenes(marianne_doc)
    # print(marianne_pwords)
    # marianne_tristan_pwords = get_normalized_scenes(marianne_tristan_doc)
    # print(marianne_tristan_pwords)
    d = count_scenes(corpus,1550, 1750, ["Comédie"])
    make_fixed_size_graph(d,5,3,True)

    # d = count_subscenes(corpus,6, 1800)
    # d = {k: v for k, v in sorted(d.items(), key=lambda item: - item[1])}
    # pickle.dump(d, open("renormalized_corpus_subscenes_all.pkl", "wb"))
    # # d = pickle.load(open("normalized_corpus_1636.pkl", "rb"))
    # d = pickle.load(open("renormalized_corpus_subscenes_all.pkl", "rb"))
    # print(d)
    # d_all = pickle.load(open("renormalized_corpus_subscenes_all.pkl", "rb"))
    # make_fixed_size_graph(d_all, 6, 3)
    # for x in d:
    #     if has_consecutive_letters(x):
    #         print(x, d[x])
    # print(f"Nombre de scenes: {tirage_total}")
    # print(longueur_5)
    # print(random_words_repartition(3, 6, 800, check_well_formed))
