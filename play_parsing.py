# -*- coding: utf-8 -*-
"""
@author: aaron
Collection of functions used to parse XML-TEI plays
"""
import glob, os, re, sys, requests, math, csv, enchant, warnings
import ast
from xml.dom import minidom

import parameterized_matching
from parameterized_matching import spm, spm_hamming, annotate_characters

# Get the current folder
folder = os.path.abspath(os.path.dirname(sys.argv[0]))
corpus = 'CorpusDracor'
outputDir = 'Output'
corpusFolder = os.path.join(folder, corpus)
outputFolder = os.path.join(folder, outputDir)

corpuscsv = 'Dracor_parameterized_plays.csv'


# Fetching data from a play
def get_genre(doc):
    """Returns the genre of the play : Tragedie, Comédie, Tragi-Comédie.
    TODO : Make sure it works """
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
            elif possible_genre in ["Tragédie", "Tragedie", "Comedie", "Comédie", "Tragicomédie", "Tragi-comédie", "Pastorale"]:
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


def get_title(doc):
    """Returns the title of a play"""
    title_nodes = doc.getElementsByTagName('title')
    if len(title_nodes) > 0:
        return title_nodes[0].firstChild.nodeValue
    else:
        warnings.warn("No title found")


def get_date(doc):
    """Returns date of printing of a play"""
    date_nodes = doc.getElementsByTagName('date')
    print_date = None
    if date_nodes:
        print_date = date_nodes[0].getAttribute("when")
    return print_date


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


def get_scene(doc):
    """"Given a play, returns the list of the successions of characters"""
    scene_list = doc.getElementsByTagName('div2')
    scene_list = scene_list + doc.getElementsByTagName('div')
    scene_list = [s for s in scene_list if s.getAttribute("type") == "scene"]
    return [get_characters_in_scene(s) for s in scene_list]


# Fetching data from scenes
def get_characters_in_scene(s):
    """Given a scene s, returns a set of its characters"""
    characters = set()
    repliques = s.getElementsByTagName('sp')
    for r in repliques:
        speaker_id = r.getAttribute("who")
        characters.add(speaker_id)
    return characters


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
    act_list = [a for a in act_list if a.getAttribute("type") in ["act", "acte"]]
    return [get_scene(a) for a in act_list]


# Functions that try to fix typos in character names, and split list of characters
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
    """Returns a dictionnary whose keys are play names and values are paramaterized plays"""
    res = dict()
    for c in os.listdir(corpus):
        play = minidom.parse(open(os.path.join(corpus, c), 'rb'))
        play_name = get_title(play)
        res[play_name] = get_parameterized_play(play)
        print('Parsing ' + play_name)
    return res


def get_corpus_parameterized_acts(corpus):
    """Returns a dictionnary whose keys are play names and act number and values are paramaterized plays"""
    res = dict()
    for c in os.listdir(corpus):
        play = minidom.parse(open(os.path.join(corpus, c), 'rb'))
        play_name = get_title(play)
        print('Parsing ' + play_name)
        acts = get_acts(play)
        for (i, a) in enumerate(acts):
            res[play_name + str(i + 1)] = a
    return res


def get_rich_dictionnary_play(play):
    d = dict()
    play_name = get_title(play)
    print ('Parsing ' + play_name)
    d['Nom'] = play_name
    d['Genre'] = get_genre(play)
    d['Date'] = get_date(play)
    acts = get_acts(play)
    nb_scenes = 0
    full_play = []
    for (i, a) in enumerate(acts):
        d["Acte " + str(i+1)] = a
        nb_scenes += len(a)
        full_play.extend(a)
    d['Nombre actes'] = len(acts)
    d['Nombre de scenes'] = nb_scenes
    d['Piece'] = full_play
    d['Personnages'] = cast(full_play)
    return d


def same_play(play_name1, play_name2):
    """Checks if play_name1 and play_name2 are acts from the same play"""
    return play_name1[:-1] == play_name2[:-1]


def generic_corpus_traversal_1(corpus, f_list, output_name, acts=False):
    """Iterates functions in f_list over given corpus and saves the output as a csv file"""
    output = open(os.path.join(outputFolder, " ".join(["Output ", output_name, corpus, ".txt"])), 'w+')
    if acts:
        pp_corpus = get_corpus_parameterized_acts(corpus)
    else:
        pp_corpus = get_corpus_parameterized_plays(corpus)

    for play_name in pp_corpus:
        print(play_name)
        for f in f_list:
            res_f = f(pp_corpus[play_name])
            if res_f[0]:
                output.write(" ".join([play_name, ':', f.__name__, str(res_f[1]), '\n']))
        print()


def create_csv_output(corpus, output_name):
    output = open(output_name + '.csv', mode='w')
    fieldnames = ['Nom', 'Genre', 'Date', 'Nombre actes', 'Nombre de scenes', 'Acte 1', 'Acte 2', 'Acte 3', 'Acte 4', 'Acte 5', 'Piece', 'Personnages']
    gwriter = csv.DictWriter(output, fieldnames=fieldnames)
    gwriter.writeheader()
    for c in os.listdir(corpus):
        play = minidom.parse(open(os.path.join(corpus, c), 'rb'))
        d = get_rich_dictionnary_play(play)
        for f in fieldnames:
            if f not in d:
                d[f] = []
        if d['Nombre actes']>5:
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
    output_char_rule = open(os.path.join(outputFolder, "Output apparition persos " +genre) + corpus + ".txt", 'w+')
    nb_genre = 0
    nb_wrong = 0
    for d in c:
        if d['Genre'] == genre and d['Date'] != '' and 1700 >= int(d['Date']) >= 1600 and int(d['Nombre actes']) > 1:
            nb_genre += 1
            wrong = False
            for i in range(1,5):
                check, s = parameterized_matching.check_character_apperance_rules(ast.literal_eval(d['Acte '+ str(i)]))
                wrong = wrong or check
                if check:
                    output_char_rule.write(" ".join([d['Nom'], "Acte ", str(i), str(s), '\n']))
            if wrong:
                nb_wrong += 1
    output_char_rule.write(f"{nb_wrong} {genre} brisent les règles sur {nb_genre}\n")


if __name__ == "__main__":
    with open(corpuscsv, newline='') as csvfile:
        d = csv.DictReader(csvfile, dialect='unix')
        check_character_rule(d,"Tragi-comédie")
    #generic_corpus_traversal_1(corpus, [parameterized_matching.check_character_apperance_rules], 'censor', True)
    # pp_corpus = get_corpus_parameterized_plays(corpus)
    # generic_corpus_traversal_2(corpus, [spm_hamming], 'SPM_Hamming_1', True)
    # create_csv_output(corpus, 'Dracord_parameterized_plays')
    # r = get_corpus_parameterized_acts(corpus)
    # print(r)
    # create_csv_output(corpus, "Dracor_parameterized_plays")
