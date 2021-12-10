# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 10:15:46 2021

@author: aaron
"""

import glob, os, re, sys, requests, math, csv
from xml.dom import minidom
import networkx as nx
#from play_parsing import get_title, get_genre, get_acts, get_characters

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

def spm(play1, play2):
    """Checks if two plays/acts are in spm"""
    if len(play1) == len(play2):  # Can only match if both inputs have same structure
        alphabet1 = set()
        alphabet2 = set()  # set of characters
        for scene in play1:
            if len(scene) != len(play2[play1.index(scene)]):  # Can only match if both inputs have same structure
                return (False, 1)
            for char in scene:
                alphabet1.add(char)
        for scene in play2:
            if len(scene) != len(play1[play2.index(scene)]):  # Can only match if both inputs have same structure
                return (False, 1)
            for char in scene:
                alphabet2.add(char)
        # Construct the potential image of each character
        potential_images = {char: alphabet2 for char in alphabet1}
        for scene in play1:
            for char in scene:
                potential_images[char] = play2[play1.index(scene)].intersection(potential_images[char])

        # Graph construction
        G = nx.Graph()
        nodes_1, nodes_2 = list(alphabet1), list(alphabet2)
        G.add_nodes_from(nodes_1, bipartite=0)
        G.add_nodes_from(nodes_2, bipartite=1)
        G.add_edges_from([(char1, char2) for char1 in potential_images for char2 in potential_images[char1]])
        try:
            matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G, nodes_1)
        except KeyError:
            print(G.nodes)
            print(G.edges)
            return (False, 0.001)
        return (True, matching)
    else:
        return (False, 1)


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
    p1 = [{'a', 'b', 'c'}, {'b', 'c'}, {'a', 'c'}]
    p2 = [{'1', '2', '3'}, {'2', '3'}, {'1', '3'}]
    m = spm(p1, p2)
    print(m)
