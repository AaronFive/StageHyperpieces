# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 10:15:46 2021

@author: aaron
"""

import glob, os, re, sys, requests, math, csv
from xml.dom import minidom
import networkx as nx

# from play_parsing import get_title, get_genre, get_acts, get_characters

# Get the current folder
folder = os.path.abspath(os.path.dirname(sys.argv[0]))
corpus = 'Corpus pieces tres proches'
outputDir = 'Output'
corpusFolder = os.path.join(folder, corpus)
outputFolder = os.path.join(folder, outputDir)

playname1 = os.path.join(corpusFolder, "Couple 1 - la-dama-duende.xml")
playname2 = os.path.join(corpusFolder, "Couple 1 - ouville_espritfolet.xml")
play1 = open(playname1, 'rb')
play2 = open(playname2, 'rb')
mydoc = minidom.parse(play1)
mydoc2 = minidom.parse(play2)


##### Computations on plays####

# Une pièce (ou un acte) est une liste de set
def annotate_characters(play, prefix):
    """Prefixes all character names in play by prefix"""
    for scene in play:
        for char in scene.copy():
            scene.add("_".join([prefix, char]))
            scene.remove(char)
    return play


def spm(play1, play2):
    """Checks if two plays/acts are in spm"""
    if len(play1) == len(play2):  # Can only match if both inputs have same structure
        alphabet1 = set()
        alphabet2 = set()  # set of characters
        for (i, scene) in enumerate(play1):
            if len(scene) != len(play2[i]):  # Can only match if both inputs have same structure
                return False, 1
            for char in scene:
                alphabet1.add(char)
        for scene in play2:
            for char in scene:
                alphabet2.add(char)
        # Construct the potential image of each character
        potential_images = {char: alphabet2 for char in alphabet1}
        for (i, scene) in enumerate(play1):
            for char in scene:
                potential_images[char] = potential_images[char].intersection(play2[i])
        # Graph construction
        G = nx.Graph()
        nodes_1, nodes_2 = list(alphabet1), list(alphabet2)
        G.add_nodes_from(nodes_1, bipartite=0)
        G.add_nodes_from(nodes_2, bipartite=1)
        G.add_edges_from([(char1, char2) for char1 in potential_images for char2 in potential_images[char1]])
        try:
            matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G, nodes_1)
        except KeyError:
            print('Erreur pour le graphe suivant')
            print(G.nodes)
            print(G.edges)
            return (False, 0.001)
        if len(matching) == len(alphabet1) + len(alphabet2):
            return True, matching
        else:
            return False, 1
    else:
        return False, 1


def spm_hamming(play1, play2, cutoff=3):
    # Matchings do not seem to be deterministic : fix an order
    # Number of mismatches : check stage report
    """Checks if two plays/acts are in spm"""
    if len(play1) != len(play2):  # Can only match if both inputs have same structure
        return False, 1
    alphabet1 = set()
    alphabet2 = set()  # set of characters
    weights = {}
    # Checking structure compatibility and getting number of correspondances
    for (scene_nb, scene) in enumerate(play1):
        if len(scene) != len(play2[play1.index(scene)]):  # Can only match if both inputs have same structure
            return False, 1
        for char in scene:
            alphabet1.add(char)
            weights[char] = weights.get(char, {})
            for char2 in play2[scene_nb]:
                weights[char][char2] = weights[char].get(char2, 0) + 1
                alphabet2.add(char2)

    # Graph construction
    G = nx.Graph()
    nodes_1, nodes_2 = list(alphabet1), list(alphabet2)
    G.add_nodes_from(nodes_1, bipartite=0)
    G.add_nodes_from(nodes_2, bipartite=1)
    for char1 in weights:
        for char2 in weights[char1]:
            G.add_edge(char1, char2, weight=weights[char1][char2])

    # Getting maximal matching
    try:
        matching = nx.algorithms.matching.max_weight_matching(G, nodes_1)
    # I got a Key Error at some point and didn't know why, but that seems to be very rare
    except KeyError:
        print('Erreur pour le graphe suivant')
        print(G.nodes)
        print(G.edges)
        return False, 0.001

    # NetworkX does not return matchings in a standardized way, it's not even deterministic
    # We put it in form {c1:c2} where c1 is in play1 and c2 in play2
    normalized_matching = dict()
    for (c1, c2) in matching:
        if c1 in alphabet1:
            normalized_matching[c1] = c2
        else:
            normalized_matching[c2] = c1
    # Computing number of mismatches
    distance = 0
    for (i, scene) in enumerate(play1):
        for char in scene:
            if normalized_matching.get(char, "char_not_found") not in play2[i]:
                distance += 1
    if distance <= cutoff:
        return True, normalized_matching
    else:
        return False, 1


def get_intervals(play):
    """Returns a dictionnary containing the first and last apperance of a character within an act.
    It is supposed that no character enters twice in an act, otherwse raises an Error."""
    intervals = dict()
    for (scene_nb, scene) in enumerate(play):
        for char in scene:
            if char not in intervals:
                intervals[char] = [scene_nb, scene_nb]
            else:
                if intervals[char][1] != scene_nb - 1:
                    raise ValueError(" ".join(
                        ['Character', str(char), 'outs at scene', str(intervals[char][1]+1), 'and enters again at scene',
                         str(scene_nb+1)]))
                intervals[char][1] = scene_nb
    return intervals


def check_character_apperance_rules(play):
    try:
        get_intervals(play)
    except ValueError as s:
        return True, s
    return False, ''


def alignement_cost(int1, int2):
    (m1, M1) = int1
    (m2, M2) = int2
    if M1 < m2 or M2 < m1:
        return M1 - m1 + M2 - m2
    else:
        return abs(m1 - m2) + abs(M1 - M2)


def spm_hamming_2(play1, play2, cutoff=2):
    int1, int2 = get_intervals(play1), get_intervals(play2)
    nodes1, nodes2 = [], []
    edges = []
    first_loop = True
    for char1 in int1:
        nodes1.append(char1)
        for char2 in int2:
            if first_loop:
                nodes2.append(char2)
            edges.append((char1, char2, alignement_cost(int1[char1], int2[char2])))
        first_loop = False
    G = nx.Graph()
    G.add_nodes_from(nodes1, bipartite=0)
    G.add_nodes_from(nodes2, bipartite=1)
    G.add_weighted_edges_from(edges)
    m = nx.bipartite.minimum_weight_full_matching(G, top_nodes=nodes1)
    print(m)
    return sum(alignement_cost(x, m[x]) for x in m) <= cutoff


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
    return bags, max([len(b) for b in bags])


if __name__ == "__main__":
    p1 = [{'a', 'b', 'c'}, {'a', 'c'}, {'a', 'c'}]
    p2 = [{'1', '2', '3'}, {'1', '2','3'}, {'1', '3'}]
    m = spm_hamming_2(p1, p2)
