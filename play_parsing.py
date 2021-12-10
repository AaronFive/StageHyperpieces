# -*- coding: utf-8 -*-
"""
@author: aaron
Collection of functions used to parse XML-TEI plays
"""
import glob, os, re, sys, requests, math, csv, enchant, warnings
from xml.dom import minidom
from parameterized_matching import spm

# Get the current folder
folder = os.path.abspath(os.path.dirname(sys.argv[0]))
corpus = 'Corpus Bibdramatique'
outputDir = 'Output'
corpusFolder = os.path.join(folder, corpus)
outputFolder = os.path.join(folder, outputDir)


def get_genre(doc):
    """Returns the genre of the play : Tragedie, Comédie, Tragi-Comédie.
    TODO : Make sure it works """
    genre = ""
    genre_entry = doc.getElementsByTagName('genre')
    if len(genre_entry) >= 1:
        genre = genre_entry[0].firstChild.nodeValue
    if genre == "":
        title = doc.getElementsByTagName('title')
        for c in title:
            typ = c.getAttribute("type")
            if typ == "sub":
                genre = c.firstChild.nodeValue
                genre = re.split(", ", genre)[1]
    if genre == "":
        term = doc.getElementsByTagName("term")
        for c in term:
            typ = c.getAttribute("type")
            if typ == "genre":
                genre = c.firstChild.nodeValue
    return genre


def get_title(doc):
    """Returns the title of a play"""
    title_nodes = doc.getElementsByTagName('title')
    if len(title_nodes) > 0:
        return title_nodes[0].firstChild.nodeValue
    else:
        warnings.warn("No title found")


def get_characters_by_cast(doc):
    """Returns the list of identifiers of characters, when the list is declared at the start"""
    id_list = []
    char_list = doc.getElementsByTagName('role')
    for c in char_list:
        name_id = c.getAttribute("id")
        if name_id == "":
            name_id = c.getAttribute("xml:id")
        if name_id == "":
            warnings.warn("Role has no id nor xml:id attribute")
        id_list.append(name_id)
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


def get_characters(doc):
    """Returns the list of identifiers of characters"""
    char_list = doc.getElementsByTagName('role')
    if not char_list:
        return get_characters_by_bruteforce(doc)
    else:
        return get_characters_by_cast(doc)


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
    """"Given a play, returns the list of the acts"""
    act_list = doc.getElementsByTagName('div1')
    act_list = act_list + doc.getElementsByTagName('div')
    act_list = [a for a in act_list if a.getAttribute("type") in ["act", "acte"]]
    return [get_scene(a) for a in act_list]


def approximate_renaming(name1, name2, tolerance):
    """Tries to check for typos and variance in character names.
    Checks if candidate is a substring or is at distance <tolerance of real name
    Problem : 'soldat1' gets renamed to 'soldat2' """
    edit = False
    if enchant.utils.levenshtein(name1, name2) <= tolerance:
        if not (name1[-1].isnumeric() or name2[-1].isnumeric()):
            edit = True
    return edit or (name1 in name2) or (name1 in name2)


def is_list_of_characters(name, characters):
    chars_in_name = name.split(" ")
    return len(chars_in_name) > 1 and all([c in characters for c in chars_in_name])


def fix_character_names(play):
    """Tries to fix typos in the name of characters in a play by
        the specification given by approximate_renaming """
    characters = dict()  # dictionnary counting number of occurences
    for s in play:
        for c in s:
            characters[c] = characters.get(c, 0) + 1
    # Either a character is renamed because it's acually a list of characters name
    # Or because there's a typo
    # We indicate the list of character case by True and the typo case by False
    renaming_dict = {c: [False] for c in characters}

    # Splitting character names that are a list of characters
    for c in characters:
        if is_list_of_characters(c, characters):
            renaming_dict[c][0] = True
            renaming_dict[c] = renaming_dict[c] + c.split(" ")
            # print(c, "is the list of characters", renaming_dict[c][1:])
    for c in characters:
        for d in characters:
            if not renaming_dict[d][0] and characters[c] < characters[d] and approximate_renaming(d, c, 1):
                renaming_dict[c].append(d)
    # If multiple candidates are possible for the renaming, we pick the most frequent one
    for c in renaming_dict:
        if not (renaming_dict[c][0]) and len(renaming_dict[c]) > 1:
            most_frequent = max(renaming_dict[c][1:], key=(lambda x: characters[x]))
            renaming_dict[c] = [False, most_frequent]
            # print(c, " renommé en ", renaming_dict[c][1])
    for s in play:
        for c in s.copy():
            if len(renaming_dict[c]) > 1:  # If c is renamed
                s.remove(c)
                for new_name in renaming_dict[c][1:]:
                    s.add(new_name)
    return play

def get_parameterized_play(play):
    sc = get_scene(play)
    return fix_character_names(sc)


def cast(play):
    """Returns the set of characters appearing in the play"""
    c = set()
    play = fix_character_names(play)
    for sc in play:
        for char in sc:
            c.add(char)
    return c


def differences_cast_declaration(doc):
    characters_list = get_characters_by_cast(doc)
    if characters_list:
        characters_list.sort()
        title = get_title(doc)
        play = get_scene(doc)
        cast_list = list(cast(play))
        cast_list.sort()
        if characters_list != cast_list:
            print(title, "\n", "characters_list : ", characters_list, "\n", "cast_list :", cast_list)
            return False
        return True

def get_corpus_parameterized_plays(corpus):
    res = dict()
    for c in os.listdir(corpus):
        play = minidom.parse(open(os.path.join(corpus, c), 'rb'))
        play_name = get_title(play)
        res[play_name] = get_parameterized_play(play)
        print ('Parsing '+ play_name)
    return res

def get_corpus_parameterized_acts(corpus):
    res = dict()
    for c in os.listdir(corpus):
        play = minidom.parse(open(os.path.join(corpus, c), 'rb'))
        play_name = get_title(play)
        print('Parsing ' + play_name)
        acts = get_acts(play)
        for a in acts:
            res[play_name + ' Acte ' + str(acts.index(a) + 1)] = a
    return res

def generic_corpus_traversal_1(corpus, f_list, output_name):
    """Iterates functions in f_list over given corpus and saves the output as a text file"""
    output = open(os.path.join(outputFolder, " ".join(["Output ", output_name, corpus, ".txt"]), 'w+'))
    pp_corpus = get_corpus_parameterized_plays(corpus)
    for play_name, p_p in pp_corpus.item():
        print(play_name)
        for f in f_list:
            res_f = f(p_p)
            output.write(" ".join([play_name, ':', f.__name__, str(res_f), '\n']))
        print()



def generic_corpus_traversal_2(corpus, f_list, output_name):
    """Iterates functions in f_list over pairs in the given corpus and saves the output as a text file
        Function should return a boolean and a result (eventually None).
        Only calls returning True as their first value will be taken into account. """
    output = open(os.path.join(outputFolder, " ".join(["Output ", output_name, corpus, ".txt"])), 'w+')
    seen = set()
    incompatibles, total = 0, 0
    pp_corpus = get_corpus_parameterized_acts(corpus).items()
    for play_name1, pp1 in pp_corpus:
        seen.add(play_name1)
        print(play_name1)
        for play_name2, pp2 in pp_corpus:
            if play_name2 not in seen:
                total += 1
                for f in f_list:
                    res_f = f(pp1, pp2)
                    if res_f[0]:
                        output.write(" ".join([play_name1, play_name2, ":", f.__name__, str(res_f[1]), '\n']))
                    else :
                        incompatibles += res_f[1]
    print( 'incompatibles : ', incompatibles)
    print('total :', total)


def check_corpus(corpus):
    mismatches = 0
    size = 0
    for c in os.listdir(corpus):
        size += 1
        doc = minidom.parse(open(os.path.join(corpus, c), 'rb'))
        if differences_cast_declaration(doc):
            mismatches += 1
    print("size : ", size, "mismatches : ", mismatches)


def create_outputs_structure(corpus):
    """Generates two outputs, one for acts, and one for complete plays"""
    output_scene = open(os.path.join(outputFolder, "Outputscenes") + corpus + ".txt", 'w+')
    output_acts = open(os.path.join(outputFolder, "Outputactes") + corpus + ".txt", 'w+')
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
    output_pathwidth = open(os.path.join(outputFolder, "Outputpathwidth") + corpus + ".txt", 'w+')
    for c in os.listdir(corpus):
        play = minidom.parse(open(os.path.join(corpus, c), 'rb'))
        play_name = get_title(play)
        print(play_name)
        scenes = get_scene(play)
        nb_cast = len(cast(scenes))
        bags, pw = pathwidth(scenes)
        output_pathwidth.write(play_name + " : " + "Cast size " + str(nb_cast) + ", Pathwidth : " + str(pw) + '\n')


if __name__ == "__main__":
    #pp_corpus = get_corpus_parameterized_plays(corpus)
    generic_corpus_traversal_2(corpus, [spm], 'SPM')