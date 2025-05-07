# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:19:25 2021

@author: aaron

Compute various indicators, based on Marcus work.
"""

import csv
import os
import re
import sys
from xml.dom import minidom

from play_parsing import get_genre

# Get the current folder
folder = os.path.abspath(os.path.dirname(sys.argv[0]))
corpusFolder = os.path.join(folder, "corpus11paires")
corpusDracor = os.path.join("Corpus", "CorpusDracor - new")


# playname = "GILBERT_RODOGUNE"
# play = open(playname + ".xml", 'rb')
# mydoc = minidom.parse(play)
# tags = mydoc.getElementsByTagName('castItem')


def get_characters_by_cast(doc):
    """Returns the list of identifiers of characters, when the list is declared at the start"""
    id_list = []
    char_list = doc.getElementsByTagName('role')
    for c in char_list:
        name_id = c.getAttribute("id")
        if name_id == "":
            name_id = c.getAttribute("xml:id")
        if name_id == "":
            print("Warning, role has no id nor xml:id attribute")
        id_list.append(c.getAttribute("id"))
    return id_list


def get_characters_by_bruteforce(doc):
    """Returns the list of identifiers of characters, by looking at each stance"""
    id_list = []
    repliques = doc.getElementsByTagName('sp')
    for r in repliques:
        speaker_id = r.getAttribute("who")
        if speaker_id not in id_list:
            id_list.append(speaker_id)
    return id_list


# def get_characters(doc):
#     """Returns the list of identifiers of characters"""
#     char_list = doc.getElementsByTagName('role')
#     if not char_list:
#         return get_characters_by_bruteforce(doc)
#     else:
#         return get_characters_by_cast(doc)


# characters=get_characters(mydoc)
# characters = get_characters_by_bruteforce(mydoc)


# def get_genre(doc):
#     """Returns the genre of the play : Tragedie, Comédie, Tragi-Comédie.
#     TODO : Make sure it works """
#     genre = ""
#     genre_entry = doc.getElementsByTagName('genre')
#     if len(genre_entry) >= 1:
#         genre = genre_entry[0].firstChild.nodeValue
#     if genre == "":
#         title = doc.getElementsByTagName('title')
#         for c in title:
#             typ = c.getAttribute("type")
#             if typ == "sub":
#                 genre = c.firstChild.nodeValue
#                 genre = re.split(", ", genre)[1]
#     if genre == "":
#         term = doc.getElementsByTagName("term")
#         for c in term:
#             typ = c.getAttribute("type")
#             if typ == "genre":
#                 genre = c.firstChild.nodeValue
#     return genre


### HERE I COMPUTE THNGS BASED ONLY ON PRESENCE/ABSENCE FROM A SCENE ###

def get_matrix(doc, characters):
    """ Returns a matrix A such that A[i][j]=1 iff character j SPEAKS in scene i
    Note that this might yield a different result than considering the scenes where characts APPEAR"""
    scene_list = doc.getElementsByTagName('div2')
    scene_list = scene_list + doc.getElementsByTagName('div')
    A = []
    scene_number = -1
    for s in scene_list:
        if s.getAttribute("type") == "scene":
            scene_number += 1
            A.append([0 for i in range(len(characters))])
            speakers = s.getElementsByTagName("sp")
            for sp in speakers:
                speaker_name = sp.getAttribute("who")
                speaker_number = characters.index(speaker_name)
                if A[scene_number][speaker_number] == 0:
                    A[scene_number][speaker_number] = 1
    return A


# A = get_matrix(mydoc, characters)


def character_density(A, k, characters):
    """Returns character density up to scene k"""
    chi = 0
    m = len(characters)
    for i in range(k):
        for j in range(m):
            chi += A[i][j]
    return chi / (m * (k + 1))


def get_scenes_sets(characters, A):
    """ Returns the list of sets of scenes in which characters appear """
    return [set([i for i in range(len(A)) if A[i][char] == 1]) for char in range(len(characters))]


# setlist = get_scenes_sets(characters, A)


def character_occurences(characters, A):
    """Returns the list of the number of times a character occurs in the play"""
    return [sum(A[i][j] for i in range(len(A))) for j in range(len(characters))]


# occurences = character_occurences(characters, A)


def character_frequencies(characters, A):
    """Returns the list of the number of times a character occurs in the play, normalized by the number of scenes"""
    return [sum(A[i][j] for i in range(len(A))) / (len(A)) for j in range(len(characters))]


# frequencies = character_frequencies(characters, A)


def get_co_occurences(setlist, characters):
    """returns the list of number of co-occurences"""
    k = len(characters)
    return [[len(setlist[i].intersection(setlist[j])) for i in range(k)] for j in range(k)]


# co_occurences = get_co_occurences(setlist, characters)


def occurences_deviation(A, characters, co_occurences, occurences):
    k = len(characters)
    n = len(A)
    deviations = [[0 for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            qi, qj = occurences[i], occurences[j]
            nij = co_occurences[i][j]
            deviations[i][j] = nij - n * qi * qj
            if (n * qi * qj) != 0:
                deviations[i][j] = (deviations[i][j]) ** 2 / (n * qi * qj)  # normalisation step
    return deviations


# deviations = occurences_deviation(A, characters, co_occurences, frequencies)


def get_confrontations(A, characters):
    """Returns the list of confrontation factors for each character
    The confrontation factor of x is the sum over y!=x of (number of scenes where x and y appear)"""
    setlist = get_scenes_sets(characters, A)
    co_occurences = get_co_occurences(setlist, characters)
    confrontations = [0 for i in range(len(characters))]
    for i in range(len(co_occurences)):
        l = co_occurences[i]
        conf = sum(l) - l[i]
        confrontations[i] = conf
    tot = sum(confrontations)
    return [x / (tot if tot != 0 else 1) for x in confrontations]


# confrontations = get_confrontations(A, characters)


###HERE I COMPUTE THINGS BASED ON THE NUMBER OF LINES SPOKEN ###

def get_lines(doc, characters):
    total_lines = 0
    lines = [0 for i in range(len(characters))]
    repliques = doc.getElementsByTagName('sp')
    for r in repliques:
        speaker_name = r.getAttribute("who")
        speaker_number = characters.index(speaker_name)
        for phrases in r.childNodes:
            if phrases.nodeName == "l":
                total_lines += 1
                lines[speaker_number] += 1
    return (total_lines, lines)


# total_lines, lines = get_lines(mydoc, characters)
# speech_frequency = [x / total_lines for x in lines]


# Returns the list of succesions
# succesions[i][j] is the number of time character i has spoken after character j
def get_successions(doc, characters):
    n = len(characters)
    successions = [[0 for _ in range(n)] for _ in range(n)]
    repliques = doc.getElementsByTagName('sp')
    previous_speaker_number = -1
    for r in repliques:
        speaker_name = r.getAttribute("who")
        speaker_number = characters.index(speaker_name)
        if not (previous_speaker_number == -1):
            successions[speaker_number][previous_speaker_number] += 1
        previous_speaker_number = speaker_number
    return successions


# successions = get_successions(mydoc, characters)
# normalized_succesions = [[x / sum(l) for x in l] for l in successions]


### OUTPUT TO CSV : CONVERTING TO DICTIONNARIES AND WRITING
def list_to_dict(l, characters, name):
    d = {name: l[characters.index(name)] for name in characters}
    d['VALEUR'] = name
    return (d)


def table_to_dict(l, characters, value):
    d = {name: name for name in characters}
    d['VALEUR'] = value
    dicts = [d]
    for c in l:
        dicts.append(list_to_dict(c, characters, characters[l.index(c)]))
    return (dicts)


# Writing all data corresponding to a play in a separate file

def create_outputs(corpusFolder):
    output_file = os.path.join("Outputs", "Stats Markov Dracor")
    global_csv_file = open(os.path.join(output_file, 'global_stats_Dracor.csv'), mode='w', encoding='utf8', newline='')
    fieldnames = ['Pièce', 'Perso principal', 'genre', 'Première apparition', 'Fréquence(scène)',
                  'Fréquence(répliques)', 'Confrontation']
    gwriter = csv.DictWriter(global_csv_file, fieldnames=fieldnames)
    gwriter.writeheader()
    for c in os.listdir(corpusFolder):
        play = open(corpusFolder + '\\' + c, 'rb')
        playname = re.split(".xml", c)[0]
        print(playname)
        mydoc = minidom.parse(play)
        genre = get_genre(mydoc)
        print(genre)
        characters = get_characters_by_bruteforce(mydoc)
        if len(characters) <= 1:
            continue
        A = get_matrix(mydoc, characters)
        if not A:
            continue
        setlist = get_scenes_sets(characters, A)
        frequencies = character_frequencies(characters, A)
        confrontations = get_confrontations(A, characters)
        total_lines, lines = get_lines(mydoc, characters)
        speech_frequency = [x / (total_lines if total_lines else 1) for x in lines]
        co_occurences = get_co_occurences(setlist, characters)
        deviations = occurences_deviation(A, characters, co_occurences, frequencies)
        successions = get_successions(mydoc, characters)
        normalized_succesions = [[(x / sum(l)) if sum(l) != 0 else None for x in l] for l in successions]

        # Handling global csv
        mc_index = frequencies.index(max(frequencies))
        main_char = characters[mc_index]
        d = {'Pièce': playname, 'Perso principal': main_char, 'genre': genre}
        d['Fréquence(scène)'] = frequencies[mc_index]
        d['Fréquence(répliques)'] = speech_frequency[mc_index]
        d['Confrontation'] = confrontations[mc_index]
        d['Première apparition'] = 'TODO'
        gwriter.writerow(d)

        # Handling individual csv
        individual_csv_file = open(os.path.join(output_file, f'stats {playname}.csv'), mode='w', encoding='utf8',
                                   newline='')
        fieldnames = ['VALEUR'] + characters
        iwriter = csv.DictWriter(individual_csv_file, fieldnames=fieldnames)
        iwriter.writeheader()

        freqdict = list_to_dict(frequencies, characters, "Fréquences (scènes)")
        linesdict = list_to_dict(speech_frequency, characters, "Fréquences(répliques)")
        confrdict = list_to_dict(confrontations, characters, "Confrontation")

        iwriter.writerow(freqdict)
        iwriter.writerow(linesdict)
        iwriter.writerow(confrdict)

        deviationsdicts = table_to_dict(deviations, characters, "Déviations")
        for d in deviationsdicts:
            iwriter.writerow(d)

        succdicts = table_to_dict(normalized_succesions, characters, "Successions")
        for d in succdicts:
            iwriter.writerow(d)


if __name__ == "__main__":
    create_outputs(corpusDracor)

# folder = os.getcwd()
# corpusFolder = os.path.join(folder,"corpusTC")
# for c in os.listdir(corpusFolder):
#     print(c)
#     m,maxplay,charlist=0,"",[]
#     play = open(corpusFolder +'\\'+ c, 'rb')
#     mydoc = minidom.parse(play)
#     char=get_characters(mydoc)
#     if len(char)> m:
#         m=len(char)
#         maxplay=c
#         charlist=char
