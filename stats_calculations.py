# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 10:15:46 2021

@author: aaron
"""

import glob, os, re, sys, requests, math, csv
from xml.dom import minidom

# Get the current folder
folder = os.path.abspath(os.path.dirname(sys.argv[0]))
corpus = 'Corpus pieces tres proches'
corpusFolder = os.path.join(folder, corpus)

playname1 = os.path.join(corpusFolder, "Couple 1 - la-dama-duende.xml")
playname2 = os.path.join(corpusFolder, "Couple 1 - ouville_espritfolet.xml")
play1 = open(playname1, 'rb')
play2 = open(playname2, 'rb')
mydoc = minidom.parse(play1)
mydoc2 = minidom.parse(play2)


###PARSING FUNCTIONS###
def get_genre(doc):
    title = doc.getElementsByTagName('title')
    for c in title:
        typ = c.getAttribute("type")
        if typ == "sub":
            genre = c.getValue()
    return genre


def get_title(doc):
    title_node = doc.getElementsByTagName('title')[0]
    return title_node.firstChild.nodeValue


def get_characters_in_scene(s):
    """Given a scene s, returns a set of its characters"""
    characters = set()
    repliques = s.getElementsByTagName('sp')
    for r in repliques:
        speaker_id = r.getAttribute("who")
        characters.add(speaker_id)
    return characters


def get_scene(doc):
    """"Given a play, returns the list of the successions of characters"""
    scene_list = doc.getElementsByTagName('div2')
    scene_list = scene_list + doc.getElementsByTagName('div')
    scene_list = [s for s in scene_list if s.getAttribute("type") == "scene"]
    return [get_characters_in_scene(s) for s in scene_list]


def get_acts(doc):
    act_list = doc.getElementsByTagName('div1')
    act_list = act_list + doc.getElementsByTagName('div')
    act_list = [a for a in act_list if a.getAttribute("type") in ["act", "acte"]]
    return [get_scene(a) for a in act_list]


##### Computations on plays####
# Une pièce (ou un acte) est une liste de set

def cast(play):
    """Returns the set of characters appearing in the play"""
    c = set()
    for sc in play:
        for char in sc:
            c.add(char)
    return c


def approximate_renaming(name1, name2, tolerance):
    """Tries to check for typos and variance in character names.
    Checks if candidate is a substring or is at distance <tolerance of real name"""
    if (name1 in name2) or (name1 in name2):
        return (True)
    # if enchant.utils.levenshtein(real_name,candidate)<=tolerance:
    #    return(True)
    return (False)


def fix_character_names(play):
    """Tries to fix typos in the name of characters in a play by
        the specification given by approximate_renaming """
    characters = dict()  # dictionnary counting number of occurences
    for s in play:
        for c in s:
            characters[c] = characters.get(c, 0) + 1
    renaming_dict = {c: [] for c in characters}
    for c in characters:
        for d in characters:
            if characters[c] < characters[d] and approximate_renaming(d, c, 1):
                print(c, " renommé en ", d)
                renaming_dict[c].append(d)
    for c in renaming_dict:
        if len(renaming_dict[c]) > 0:
            renaming_dict[c] = max(renaming_dict[c], key=(lambda x: characters[x]))
    for s in play:
        for c in s.copy():
            if type(renaming_dict[c]) == str:
                s.remove(c)
                s.add(renaming_dict[c])
    return (play)


def pathwidth(play):
    """Returns the pathwidth of a play,
    i.e. its bag representation, 
    and also the max number of active character at all times"""
    c = cast(play)
    # Getting min and max occurence of each character
    bounds = {char: [None, None] for char in c}
    for sc in play:
        i = play.index(sc)
        for char in sc:
            if bounds[char][0] == None:
                bounds[char][0] = i
            bounds[char][1] = i
    bags = []
    for i in range(len(play)):
        bags.append({char for char in c if bounds[char][0] <= i and bounds[char][1] >= i})
    return (bags, max([len(b) for b in bags]))


def create_outputs_structure(corpus):
    """Generates two outputs, one for acts, and one for complete plays"""
    output_scene = open("Outputscenes" + corpus + ".txt", 'w+')
    output_acts = open("Outputactes" + corpus + ".txt", 'w+')
    all_acts = []
    for c in os.listdir(corpus):
        play = minidom.parse(open(os.path.join(corpus, c), 'rb'))
        play_name = get_title(play)
        print(play_name)
        acts = get_acts(play)
        all_acts = all_acts + acts
        pl = []
        for a in acts:
            output_acts.write(play_name + " Acte " + str(acts.index(a) + 1) + " :" + str(a) + '\n')
            pl = pl + a
        fixed_play = fix_character_names(pl)
        output_scene.write(play_name + " : " + str(fixed_play) + '\n')
    return all_acts


def create_output_pathwidth(corpus):
    output_pathwidth = open("Outputpathwidth" + corpus + ".txt", 'w+')
    for c in os.listdir(corpus):
        play = minidom.parse(open(os.path.join(corpus, c), 'rb'))
        play_name = get_title(play)
        print(play_name)
        scenes = get_scene(play)
        nb_cast = len(cast(scenes))
        bags, pw = pathwidth(scenes)
        output_pathwidth.write(play_name + " : " + "Cast size " + str(nb_cast) + ", Pathwidth : " + str(pw) + '\n')


# create_output_pathwidth(corpus)

def spm(play1, play2):
    """Checks if two plays/acts are in spm"""
    if len(play1) == len(play2):  # Can only match if both inputs have same structure
        alphabet = set()  # set of characters
        for scene in play1:
            if len(scene) != len(play2[play1.index(scene)]):  # Can only match if both inputs have same structure
                return False
            for char in scene:
                alphabet.add(char)
        # Construct the potential image of each character
        potential_images = {char: alphabet for char in alphabet}
        for scene in play1:
            for char in scene:
                potential_images[char] = play2[play1.index(scene)].intersection(potential_images[char])

    else:
        return (False)


# def normalize_play(p):
#     characters=dict()
#     normalized_play=[]
#     nb_characters=0
#     for s in p:
#         normalized_scene=set()
#         for c in s:
#             if c not in characters:
#                 characters[c]=nb_characters
#                 nb_characters+=1
#             normalized_scene.add(characters[c])
#         normalized_play.append(normalized_scene)

if __name__ == "__main__":
    create_outputs_structure(corpus)
