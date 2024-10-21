# -*- coding: utf-8 -*-
"""
@author: aaron
Collection of functions used to parse XML-TEI plays
"""
import glob, os, re, sys, requests, math, csv, warnings
import ast
import enchant
from xml.dom import minidom
import networkx as nx
import pickle
import parameterized_matching
from rules_checker import *
from parameterized_matching import spm, spm_hamming, annotate_characters
from Levenshtein import distance

# Get the current folder
folder = os.path.abspath(os.path.dirname(sys.argv[0]))
corpus = 'CorpusDracor'
outputDir = 'Output'
corpusFolder = os.path.join(folder, 'Corpus', corpus)
outputFolder = os.path.join(folder, outputDir)

corpuscsv = 'Dracor_parameterized_plays.csv'
corpus_plays = 'Pickled Dracor/full_plays_dracor.pkl'
corpus_acts_merged = 'Pickled Dracor/merged_acts_dracor.pkl'
corpus_acts_separed = 'Pickled Dracor/separed_acts_dracor.pkl'


def get_play_from_file(file):
    """Returns a document object from a file"""
    f = open(file, 'rb')
    play = minidom.parse(f)
    return play


# Fetching data from a play. Inputs are XML-TEI files parsed by minidom
def get_genre(doc):
    """Returns the genre of the play : Tragedie, Comédie, Tragi-Comédie."""
    genre_list = ["Tragédie", "Tragedie", "Comedie", "Comédie", "Tragicomédie", "Tragi-comédie"]
    genre = ""
    genre_entry = doc.getElementsByTagName('genre')
    if len(genre_entry) >= 1:
        genre = genre_entry[0].firstChild.nodeValue

    firstgenre = ""
    if genre == "":
        term = doc.getElementsByTagName("term")
        for c in term:
            typ = c.getAttribute("type")
            possible_genre = c.firstChild.nodeValue
            if typ == "genre":
                genre = possible_genre
            elif possible_genre in ["Tragédie", "Tragedie", "Comedie", "Comédie", "Tragicomédie", "Tragi-comédie",
                                    "Pastorale"]:
                genre = c.firstChild.nodeValue
            if firstgenre == "" and possible_genre not in ["vers", "prose", "mixte"]:
                firstgenre = possible_genre

    if genre == "":
        genre = firstgenre

    if genre == "":
        title = doc.getElementsByTagName('title')
        firstgenre = ""
        for c in title:
            typ = c.getAttribute("type")
            if genre not in genre_list and typ == "sub" and c.firstChild is not None:
                if firstgenre == "":
                    firstgenre = c.firstChild.nodeValue
                genre = c.firstChild.nodeValue
                if genre not in genre_list:
                    for x in genre_list:
                        if x in genre:
                            genre = x

        genre = firstgenre
    return genre


# Many pays have multiple <titles> tags. This functions returns the content of the first one, because it usually contains the actual title
# Additional title nodes usually contain information such as subtitle, genre, versification, if the play is in multiple parts, etc
def get_title(doc):
    """Returns the title of a play"""
    title_nodes = doc.getElementsByTagName('title')
    if len(title_nodes) > 0:
        return title_nodes[0].firstChild.nodeValue
    else:
        warnings.warn("No title found")


def get_dracor_id(doc):
    id_nodes = doc.getElementsByTagName('idno')
    id = None
    for n in id_nodes:
        if n.getAttribute('type') == "dracor":
            if id is None:
                id = n.firstChild.nodeValue
            else:
                title = get_title(doc)
                warnings.warn(f'Play {title} has multiple ids')
    if id is None:
        title = get_title(doc)
        warnings.warn(f'Play {title} has no id')
    return id


def get_date(doc):
    """Returns date of printing of a play"""
    date_nodes = doc.getElementsByTagName('date')
    print_date = None
    if date_nodes:
        print_date = date_nodes[0].getAttribute("when")
    return print_date


# Characters can be declared in two ways in an XML-TEI files : either in a <ListPerson> section or in a <castList> section
# The <castList> corresponds to the declaration of characters as it is printed in a paper edition of the book, presenting the characters
# The <ListPerson> is an internal XML list
# Both these sections contain identifiers for the characters, but they do not necesarily match
# All Dracor files contain both a <ListPerson> and a <castList>.
# The <ListPerson> has been generated from the file, and uses the <castList> to infer character full names in French
# In general, the ListPerson section is more coherent with the XML document and should be chosen first
def get_characters_by_cast(doc):
    """Returns the list of identifiers of characters, when the list is declared at the start"""
    id_list = []
    listperson = doc.getElementsByTagName('listPerson')
    if listperson:
        char_list = listperson[0].getElementsByTagName('person')
        return ["".join(["#", c.getAttribute("xml:id")]) for c in char_list]  # prefixing every name by #
    else:
        # if there is no listPerson we use the the castList
        char_list = doc.getElementsByTagName('role')
        for c in char_list:
            name_id = c.getAttribute("corresp")
            if name_id == "":
                name_id = c.getAttribute("id")
            if name_id == "":
                name_id = c.getAttribute("xml:id")
            if name_id == "":
                title = get_title(doc)
                warnings.warn(f" Play {title} :Role has no id nor xml:id nor corresp attribute")
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
    char_list = get_characters_by_cast(doc)
    if not char_list:
        char_list = get_characters_by_bruteforce(doc)
    return char_list


def get_scenes(doc):
    """"Given a play, returns the list of the successions of characters"""
    scene_list = doc.getElementsByTagName('div2')
    scene_list = scene_list + doc.getElementsByTagName('div')
    scene_list = [s for s in scene_list if s.getAttribute("type") == "scene"]
    return [get_characters_in_scene(s) for s in scene_list]


def get_scene_text(scene):
    """"Given a scene, returns a list of the form [(locutor, text said by locutor)]"""
    speaker_list = scene.getElementsByTagName('sp')
    full_text = []
    for speaker_nodes in speaker_list:
        sentences = speaker_nodes.getElementsByTagName('l')
        sentences = sentences + speaker_nodes.getElementsByTagName('s')
        sentences = sentences + speaker_nodes.getElementsByTagName('stage')
        text = ' '.join([s.firstChild.nodeValue for s in sentences if s.childNodes])
        locutor_node = speaker_nodes.getElementsByTagName('speaker')
        if locutor_node and locutor_node[0].childNodes:
            locutor_name = locutor_node[0].firstChild.nodeValue
        else:
            locutor_name = None
        full_text.append((locutor_name, text))
    return full_text


def get_full_text(doc):
    """ Given a play, returns the list of the whole text in the form [(locutor, text said by locutor)].
    This removes the separation by acts and scenes.
    #TODO: Adapt to make it work on plays with no scenes. How many are there ?"""
    scene_list = doc.getElementsByTagName('div2')
    scene_list = scene_list + doc.getElementsByTagName('div')
    scene_list = [s for s in scene_list if s.getAttribute("type") == "scene"]
    full_text_list = [get_scene_text(s) for s in scene_list]
    flattened_list = [t for f in full_text_list for t in f]
    return flattened_list


def get_all_scenes_dialogues(doc):
    """Returns the succession of characters talking, in all scenes"""
    scene_list = doc.getElementsByTagName('div2')
    scene_list = scene_list + doc.getElementsByTagName('div')
    scene_list = [s for s in scene_list if s.getAttribute("type") == "scene"]
    return [get_stances_succession(s) for s in scene_list]


def get_all_acts_dialogues(doc, split_by_act=False):
    """Returns the succession of characters talking, in all acts"""
    act_list = doc.getElementsByTagName('div') + doc.getElementsByTagName('div1') + doc.getElementsByTagName('div2')
    act_list = [s for s in act_list if s.getAttribute("type") in ["act", "acte"]]
    if not split_by_act:
        return [get_stances_succession(s) for s in act_list]  # The scenes in an act are merged
    else:
        return [get_all_scenes_dialogues(s) for s in act_list]  # The scenes in an act are not merged


def get_fixed_parameterized_play(play):
    """Returns a paramterized play after correction of character names """
    sc = get_scenes(play)
    return fix_character_names(sc)


# Functions that try to fix typos in character names, and split list of characters
def approximate_renaming(name1, name2, tolerance):
    """Tries to check for typos and variance in character names.
    Checks if candidate is a substring or is at distance <tolerance of real name
    Might rename characters with names too close"""
    edit = False
    if enchant.utils.levenshtein(name1, name2) <= tolerance:
        if not (name1[-1].isnumeric() or name2[-1].isnumeric()):
            edit = True
    return edit or (name1 in name2) or (name1 in name2)


def is_list_of_characters(name, characters):
    """ Checks if a string is a list of characters i.e. "Character 1 Character 2 Character 3"
    Useful for stances said by multiple characters at once"""
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
    for c in characters:
        for d in characters:
            if not renaming_dict[d][0] and characters[c] < characters[d] and approximate_renaming(d, c, 0):
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


# Fetching data from scene nodes directly
def get_characters_in_scene(s):
    """Given a scene node s, returns a set of its characters"""
    characters = set()
    repliques = s.getElementsByTagName('sp')
    for r in repliques:
        speaker_id = r.getAttribute("who")
        characters.add(speaker_id)
    return characters


def get_stances_succession(s):
    """Given a scene s, returns the list of name of characters talking"""
    repliques = s.getElementsByTagName('sp')
    scene = [r.getAttribute("who") for r in repliques]
    return scene


def get_characters_in_scene_from_header(s):
    """Reads the declaration of the form 1 'SCENE II. Perso1, Perso2, Perso3,...'
        And returns [Perso1, Perso2, Perso3,...]"""
    characters = set()
    head = s.getElementsByTagName('head')
    if len(head) == 0:
        warnings.warn("Scene has no header")
    else:
        header = head[0].firstChild.nodeValue
        characters = header.split(".")[1]
        characters = characters.split(",")
    return characters


def get_acts(doc):
    """"Given a play, returns the list of the acts"""
    act_list = doc.getElementsByTagName('div1')
    act_list = act_list + doc.getElementsByTagName('div')
    act_list = [a for a in act_list if a.getAttribute("type") in ["act", "acte", "ACTE"]]
    return [get_scenes(a) for a in act_list]


def fixed_cast(play):
    """Returns the set of characters appearing in the play after correction"""
    c = set()
    play = fix_character_names(play)
    for sc in play:
        for char in sc:
            c.add(char)
    return c


def differences_cast_declaration(doc):
    """Checks for the difference in cast between the initial list of characters and the character names appearing.
    Useful to check for typos in TEI"""

    declared_characters = get_characters_by_cast(doc)
    if declared_characters:
        declared_characters.sort()
        title = get_title(doc)
        play = get_scenes(doc)
        cast_list = list(fixed_cast(play))
        cast_list.sort()
        if not all([x == '' for x in declared_characters]) and declared_characters != cast_list:
            print(title, "\n", "declared_characters : ", declared_characters, "\n", "cast_list :", cast_list)
            return False
        return True
    return True


def get_corpus_parameterized_plays(corpus):
    """Returns a dictionnary whose keys are play names and values are paramaterized plays"""
    res = dict()
    for c in os.listdir(corpus):
        play = get_play_from_file(os.path.join(corpus, c))
        play_name = get_title(play)
        res[play_name] = get_parameterized_play(play)
        print('Parsing ' + play_name)
    return res


def get_corpus_parameterized_acts(corpus, act_types="separate"):
    """Returns a dictionnary whose keys are play names and act number and values are parameterized plays (default)
        If act_types is "merged", keys are play_name and values are list of acts """
    res = dict()
    for c in os.listdir(corpus):
        play = get_play_from_file(os.path.join(corpus, c))
        play_name = get_title(play)
        print('Parsing ' + play_name)
        acts = get_acts(play)
        if act_types == "separate":
            for (i, a) in enumerate(acts):
                res[play_name + str(i + 1)] = a
        elif act_types == "merged":
            res[play_name] = acts
        else:
            ValueError(f"Unkwon argument for act_types : {act_types} (must be separate or merged)")
    return res


def get_rich_dictionnary_play(play):
    d = dict()
    play_name = get_title(play)
    print('Parsing ' + play_name)
    d['Nom'] = play_name
    d['Genre'] = get_genre(play)
    d['Date'] = get_date(play)
    acts = get_acts(play)
    nb_scenes = 0
    full_play = []
    for (i, a) in enumerate(acts):
        d["Acte " + str(i + 1)] = a
        nb_scenes += len(a)
        full_play.extend(a)
    d['Nombre actes'] = len(acts)
    d['Nombre de scenes'] = nb_scenes
    d['Piece'] = full_play
    d['Personnages'] = fixed_cast(full_play)
    return d


def same_play(play_name1, play_name2):
    """Checks if play_name1 and play_name2 are acts from the same play"""
    return play_name1[:-1] == play_name2[:-1]


def generic_corpus_traversal_1(corpus, f_list, output_name, acts=False):
    """Iterates functions in f_list over given corpus and saves the output as a csv file"""
    output = open(os.path.join(outputFolder, " ".join(["Output ", output_name, corpus, ".csv"])), 'w+')
    fieldnames = ['Nom'] + [f.__name__ for f in f_list]
    gwriter = csv.DictWriter(output, fieldnames=fieldnames)
    gwriter.writeheader()
    if acts:
        pp_corpus = get_corpus_parameterized_acts(corpus, "merged")
    else:
        pp_corpus = get_corpus_parameterized_plays(corpus)

    for play_name in pp_corpus:
        d = dict()
        d['Nom'] = play_name
        for f in f_list:
            res_f = f(pp_corpus[play_name])
            if res_f[0]:
                d[f.__name__] = str(res_f[1])
                # output.write(" ".join([play_name, ':', f.__name__, str(res_f[1]), '\n']))
        gwriter.writerow(d)


def create_csv_output(corpus, output_name):
    output = open(output_name + '.csv', mode='w')
    fieldnames = ['Nom', 'Genre', 'Date', 'Nombre actes', 'Nombre de scenes', 'Acte 1', 'Acte 2', 'Acte 3', 'Acte 4',
                  'Acte 5', 'Piece', 'Personnages']
    gwriter = csv.DictWriter(output, fieldnames=fieldnames)
    gwriter.writeheader()
    for c in os.listdir(corpus):
        play = get_play_from_file(os.path.join(corpus, c))
        d = get_rich_dictionnary_play(play)
        for f in fieldnames:
            if f not in d:
                d[f] = []
        if d['Nombre actes'] > 5:
            print(f"{d['Nom']} a  {d['Nombre actes']} actes")
        else:
            gwriter.writerow(d)
        if all([x == '' for x in d.values()]):
            print('empty_dict')


def generic_corpus_traversal_2(corpus, f_list, output_name, acts=False, rename=False):
    """Iterates functions in f_list over pairs in the given corpus and saves the output as a text file
        Function should return a boolean and a result (eventually None).
        Only calls returning True as their first value will be taken into account. """
    # Initializing
    output = open(os.path.join(outputFolder, " ".join(["Output ", output_name, corpus, ".txt"])), 'w+')
    seen = set()
    incompatibles, total = 0, 0

    # Getting parameterized plays or acts
    if acts:
        pp_corpus = get_corpus_parameterized_acts(corpus)
    else:
        pp_corpus = get_corpus_parameterized_plays(corpus)

    # Renaming characters by prefixing them with the play they come from
    if rename:
        for play_name in pp_corpus:
            pp_corpus[play_name] = annotate_characters(pp_corpus[play_name], play_name)

    # Iterating over all pairs of plays in the corpus
    for play_name1 in pp_corpus:
        seen.add(play_name1)
        print(play_name1)
        play1 = pp_corpus[play_name1]
        for play_name2 in pp_corpus:
            # Checking if the play hasn't already been seen, and if we're working with acts, that both acts are not
            # from the same play
            if play_name2 not in seen and (not acts or not same_play(play_name1, play_name2)):
                total += 1
                play2 = pp_corpus[play_name2]
                for f in f_list:
                    res_f = f(play1, play2)
                    if res_f[0]:
                        to_output = [play_name1, "et", play_name2, ":", f.__name__, str(res_f[1]), '\n']
                        output.write(" ".join(to_output))
                    else:
                        incompatibles += res_f[1]
    print('incompatibles : ', incompatibles)
    print('total :', total)


def check_corpus(corpus):
    mismatches = 0
    size = 0
    for c in os.listdir(corpus):
        size += 1
        doc = get_play_from_file(os.path.join(corpus, c))
        if differences_cast_declaration(doc):
            mismatches += 1
    print("size : ", size, "mismatches : ", mismatches)


def create_outputs_structure(corpus):
    """Generates two outputs, one for acts, and one for complete plays"""
    output_scene = open(os.path.join(outputFolder, "Outputscenes") + corpus + ".txt", 'w+')
    output_acts = open(os.path.join(outputFolder, "Outputactes") + corpus + ".txt", 'w+')
    all_acts = []
    for c in os.listdir(corpus):
        play = get_play_from_file(os.path.join(corpus, c))
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
        play = get_play_from_file(os.path.join(corpus, c))
        play_name = get_title(play)
        print(play_name)
        scenes = get_scene(play)
        nb_cast = len(fixed_cast(scenes))
        bags, pw = pathwidth(scenes)
        output_pathwidth.write(play_name + " : " + "Cast size " + str(nb_cast) + ", Pathwidth : " + str(pw) + '\n')


def list_to_dict(l, characters, name):
    d = {name: l[characters.index(name)] for name in characters}
    d['VALEUR'] = name
    return d


def table_to_dict(l, characters, value):
    d = {name: name for name in characters}
    d['VALEUR'] = value
    dicts = [d]
    for c in l:
        dicts.append(list_to_dict(c, characters, characters[l.index(c)]))
    return dicts


def check_character_rule(c, genre):
    output_char_rule = open(os.path.join(outputFolder, "Output apparition persos " + genre) + corpus + ".txt", 'w+')
    nb_genre = 0
    nb_wrong = 0
    for d in c:
        if d['Genre'] == genre and d['Date'] != '' and 1700 >= int(d['Date']) >= 1600 and int(d['Nombre actes']) > 1:
            nb_genre += 1
            wrong = False
            for i in range(1, 5):
                check, s = parameterized_matching.check_character_apperance_rules(ast.literal_eval(d['Acte ' + str(i)]))
                wrong = wrong or check
                if check:
                    output_char_rule.write(" ".join([d['Nom'], "Acte ", str(i), str(s), '\n']))
            if wrong:
                nb_wrong += 1
    output_char_rule.write(f"{nb_wrong} {genre} brisent les règles sur {nb_genre}\n")


def normalized_levenshtein(char1, char2):
    d = distance(char1[:-1], char2[:-1])
    s1, s2 = len(char1), len(char2)
    alignment = (s1 + s2 - d) / 2
    res = alignment / max(s1, s2)
    if res < 0.9:
        res = 0
    return res


def annotate_cast(c, number):
    if c[0] == '#':
        new_c = c[1:]
    return f'{new_c}{number}'


def cast_distance(cast1, cast2, distance_function):
    # Graph construction
    G = nx.Graph()
    cast1, cast2 = [annotate_cast(c, 1) for c in cast1], [annotate_cast(c, 2) for c in cast2]
    nodes_1, nodes_2 = cast1, cast2
    G.add_nodes_from(nodes_1, bipartite=0)
    G.add_nodes_from(nodes_2, bipartite=1)
    weights = {char1: {char2: distance_function(char1, char2) for char2 in cast2} for char1 in cast1}
    for char1 in cast1:
        for char2 in cast2:
            G.add_edge(char1, char2, weight=weights[char1][char2])
    # Getting maximal matching
    matching = nx.algorithms.matching.max_weight_matching(G, nodes_1)
    normalized_matching = dict()
    total_weight = 0
    for (c1, c2) in matching:
        if c1 not in cast1:
            c1, c2 = c2, c1
        wc1c2 = weights[c1][c2]
        normalized_matching[c1] = (c2, wc1c2)
        total_weight += wc1c2
    return total_weight, normalized_matching


def get_close_plays_by_cast(corpus, cast_file="dracor_casts.pkl", distances_file=None):
    if cast_file is None:
        casts = dict()
        for f in os.listdir(corpus):
            play = get_play_from_file(os.path.join(corpus, f))
            cast = get_characters(
                play)  # Getting characters. We don't fix the character names for efficiency purpose, but maybe that would be better ?
            title = get_title(play)
            print(title)
            casts[title] = cast
            pickle.dump(casts, open("casts.pkl", 'wb'))
    else:
        casts = pickle.load(open(cast_file, 'rb'))
    if distances_file is None:
        distances = dict()
    else:
        distances = pickle.load(open(distances_file, 'rb'))
    nb_play_added = 0
    nb_total_plays = len(casts)
    # tmp_play_timeout = 50
    # tmp_play_done = 0
    for play1 in casts:
        # if tmp_play_done > tmp_play_timeout:
        #     break
        print(play1)
        if len(distances.get(play1, [])) < len(casts) - 1:
            if play1 not in distances:
                distances[play1] = dict()
            # Checking to see if all couples with play1 have been already computed
            for play2 in casts:
                if play2 not in distances.get(play1, dict()) and play1 not in distances.get(play1,
                                                                                            dict()) and play1 != play2:
                    weight, matching = cast_distance(casts[play1], casts[play2], normalized_levenshtein)
                    distances[play1][play2] = weight, matching
                    if play2 not in distances:
                        distances[play2] = dict()
                    distances[play2][play1] = weight, matching
                nb_play_added += 1
                percent_done = round(nb_play_added / (nb_total_plays ** 2) * 100, 2)
                if nb_play_added % 25000 == 0:
                    pickle.dump(distances, open('distances_cast.pkl', 'wb'))
                    print(f'saved at {percent_done}%')
        # tmp_play_done += 1
        # print(f'done:{tmp_play_done}')
    pickle.dump(distances, open('distances_cast.pkl', 'wb'))
    return distances


# First we generate a list of cast of plays

# zoro_pp = get_all_acts_dialogues(zoroastre_doc)
# nostra_pp = get_all_acts_dialogues(nostradamus_doc)

def sort_dict(d):
    l = [(x, d[x]) for x in d]
    return sorted(l, key=lambda x: -x[1][1])


# make a dict
# keys are speaker of the plays
# values are list of repliques said by each speaker
# Then get a vector representation of every replique, average them

# For every pair of plays :
# Get the vector of each character
# Score character similiraty with cosine distance between chars
# Do a max matching
def get_speech_of_characters(play):
    pass


if __name__ == "__main__":
    distances = pickle.load(open('distances_cast.pkl', 'rb'))
    output = open('close_plays_by_cast.csv', 'w', newline='', encoding='utf8')
    fieldnames = ["Play Name", "First closest", "Average distance", "First distance", "Cast Matching 1",
                  "Second closest", "Cast Matching 2"]
    gwriter = csv.DictWriter(output, fieldnames=fieldnames)
    gwriter.writeheader()
    for play1 in distances:
        for play2 in distances[play1]:
            normalized_distance = distances[play1][play2][0]
            distances[play1][play2] = (normalized_distance, sort_dict(distances[play1][play2][1]))
    for play1 in distances:
        print(play1)
        play1_list = sorted([(x, distances[play1][x]) for x in distances[play1]], key=lambda x: -x[1][0])
        average_dist1 = sum([distances[play1][x][0] for x in distances[play1]]) / len(distances[play1])
        top_3_closest = play1_list[:2]
        csv_row = dict()
        csv_row["Play Name"] = play1
        csv_row["Average distance"] = average_dist1
        csv_row["First distance"] = top_3_closest[0][1][0]
        csv_row["First closest"] = top_3_closest[0][0]
        csv_row["Cast Matching 1"] = top_3_closest[0][1][1]
        csv_row["Second closest"] = top_3_closest[1][0]
        csv_row["Cast Matching 2"] = top_3_closest[1][1]
        gwriter.writerow(csv_row)

    # get_close_plays_by_cast(corpusFolder)
    # print("Loading")
    # docs = pickle.load(open(corpus_docs, 'rb'))
    # print('Done')
    # print (get_genre(docs['Mariamne']))
    # marianne_text = get_full_text(marianne_doc)
    # for (speaker,text) in marianne_text:
    #     print(f'{speaker} : {text}')

    # c_a = open(corpus_acts_merged,'rb')
    # plays = pickle.load(c_a)
    # for x in plays:
    #     if "Mariane" in x:
    #         print(x, plays[x])
    # with open(corpuscsv, newline='') as csvfile:
    #     d = csv.DictReader(csvfile, dialect='unix')
    #     check_character_rule(d,"Tragi-comédie")
    # generic_corpus_traversal_1(corpus, [parameterized_matching.check_character_apperance_rules], 'censor', True)
    # pp_corpus = get_corpus_parameterized_plays(corpus)
    # generic_corpus_traversal_2(corpus, [spm_hamming], 'SPM_Hamming_1', True)
    # create_csv_output(corpus, 'Dracord_parameterized_plays')
    # r = get_corpus_parameterized_acts(corpus)
    # print(r)
    # create_csv_output(corpus, "Dracor_parameterized_plays")
    # generic_corpus_traversal_1(corpus, [check_rule_1_play, check_rule_2_play, check_rule_4_play, check_rule_5_play ],'rules_douguet', True)
