import os
import random
import time

import play_parsing
import parameterized_matching
import re
from scene_speech_repartition import normalize_scene
import doctest
from xml.dom import minidom
from unidecode import unidecode

# This file implements the max-sat reduction

# Utilities

def invert_dic(d, v):
    """Given a dictionnary and a value v, returns the first key such that d[k] = v, or None if it doesn't exist
    >>> invert_dic({1:'a',2:'b'},'b')
    2
    >>> invert_dic({1:'a',2:'b'},2)

    >>> invert_dic({1:'a',2:'a', 3:'b'},'a')
    1
    """
    for x in d:
        if d[x] == v:
            return x
    return None


# Alphabetical parameterized word: word of the form ABABCBD... where letters appear consecutively by order of the
# alphabet. May use other symbols if there are more than 26 letters : the ord() of all characters must be consecutive
# and start at 65 (ord('A') = 65)


def get_alphabet_size(s_1, s_2):
    """Given two alphabetical parameterized words, returns the size of the alphabet needed to write both of them
     i.e. the rank in the alphabet of the largest letter appearing in both.
    >>> get_alphabet_size('ABABC', 'ABABCBAB')
    4"""
    return max(max(ord(x) for x in s_1), max(ord(x) for x in s_2)) - 64


def get_pi(alphabet_size):
    """ Returns an alphabet ['A', 'B', 'C',...] up until the alphabet_size-th letter of the alphabet
    >>> get_pi(0)
    []
    >>> get_pi(3)
    ['A','B','C']
    """
    pi = []
    for i in range(alphabet_size):
        pi.append(chr(65 + i))
    return pi


# We index variables x_i{i,j} and y_{a,b} and keep them in two dictionnaries
def make_corresp_dictionnaries(string_1, string_2):
    """Given two alphabetical parameterized words string_1, string_2,
    Returns two dictionnaries x_dict and y_dict whose keys are variables x_{i,j} and y_{a,b} respectively
    That fix an ordering on those variables
    The values in x_dict and y_dict are consecutive"""
    n, m = len(string_1), len(string_2)
    pi_size = get_alphabet_size(string_1, string_2)
    pi = get_pi(pi_size)
    x_dict, y_dict = dict(), dict()
    value = 0
    for i in range(n):
        for j in range(m):
            if (i,j) not in x_dict:
                value += 1
                x_dict[i, j] = value
    for a in pi:
        for b in pi:
            if (a,b) not in y_dict:
                value += 1
                y_dict[a, b] = value
    return x_dict, y_dict


def no_double_i_clause(x_dict, i, j1, j2):
    return " ".join([str(-x_dict[i, j1]), str(-x_dict[i, j2]), "0"])


def no_double_j_clause(x_dict, i1, i2, j):
    return " ".join([str(-x_dict[i1, j]), str(-x_dict[i2, j]), "0"])


def no_crossing_clause(x_dict, i1, i2, j1, j2):
    return " ".join([str(-x_dict[i1, j1]), str(-x_dict[i2, j2]), "0"])


def function_clause(y_dict, a, b1, b2):
    return " ".join([str(-y_dict[a, b1]), str(-y_dict[a, b2]), "0"])


def match_clause(x_dict, y_dict, i, j, u, v):
    return " ".join([str(-x_dict[i, j]), str(y_dict[u[i], v[j]]), "0"])


def make_sat_instance(comments, string_1, string_2):
    # Getting the alphabets of both strings
    n, m = len(string_1), len(string_2)
    pi_size = get_alphabet_size(string_1, string_2)
    pi = get_pi(pi_size)
    # Making the dictionnaries to enumerate the variables we will need
    x_dict, y_dict = make_corresp_dictionnaries(string_1, string_2)

    # Comments in the output have to be prefixed by c
    # comments is a list of strings
    # unidecode suppress weird characters that may trip up maxhs parsing (?)
    comments_list = [" ".join(["c", unidecode(x)]) for x in comments]
    comment_string = "\n".join(comments_list)

    # top is the weight we use to specify a clause is hard in max sat.
    # The sum of the weights of soft clauses is enough
    top = n * m

    # No_Double_i clauses
    clauses_i = []
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                if j1 != j2:
                    clauses_i.append((" ".join([f"{top}", no_double_i_clause(x_dict, i, j1, j2)])))
    clauses_i_string = "\n".join(clauses_i)

    # No_Double_j clauses
    clauses_j = []
    for j in range(m):
        for i1 in range(n):
            for i2 in range(n):
                if i1 != i2:
                    clauses_j.append((" ".join([f"{top}", no_double_j_clause(x_dict, i1, i2, j)])))
    clauses_j_string = "\n".join(clauses_j)

    # No_Crossing clauses
    clauses_crossings = []
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(m):
                for j2 in range(j1):
                    clauses_crossings.append(" ".join([f"{top}", no_crossing_clause(x_dict, i1, i2, j1, j2)]))
    clauses_crossings_string = "\n".join(clauses_crossings)

    # Function clauses
    clauses_function = []
    for a in pi:
        for b in pi:
            for c in pi:
                if b != c:
                    clauses_function.append(" ".join([f"{top}", function_clause(y_dict, a, b, c)]))
    clauses_function_string = "\n".join(clauses_function)

    # Match clauses
    clauses_match = []
    for i in range(n):
        for j in range(m):
            clauses_match.append((" ".join([f"{top}", match_clause(x_dict, y_dict, i, j, string_1, string_2)])))
    clauses_match_string = "\n".join(clauses_match)

    nbvar = len(string_1) * len(string_2) + pi_size ** 2
    nb_clauses = len(clauses_match) + len(clauses_function) + len(clauses_crossings) + len(clauses_j) + len(clauses_i)

    header = f"p wcnf {nbvar} {nb_clauses} {top}"

    # To maximize

    clauses_max = []
    for i in range(n):
        for j in range(m):
            clauses_max.append(" ".join(["1", str(x_dict[i, j]), "0"]))
    clauses_max_string = "\n".join(clauses_max)
    all_clauses = [comment_string, header, clauses_i_string, clauses_j_string, clauses_crossings_string,
                   clauses_function_string, clauses_match_string, clauses_max_string]
    all_clauses = [x for x in all_clauses if x != ""]  # Some clauses may not appear on very small inputs
    final_string = "\n".join(all_clauses)

    return final_string

# Given two scenes, create the maxhs input file

def encode_scenes(scene1, scene2, name = 'test'):
    u, d1 = normalize_scene(scene1, True)
    v, d2 = normalize_scene(scene2, True)
    output_for_maxhs = open(name + 'input_maxhs', 'w')
    s = make_sat_instance([str(d1), str(d2)], u, v)
    output_for_maxhs.write(s)
    output_for_maxhs.close()
    return name + 'input_maxhs', d1, d2, u,v


#Given a maxhs output file, translate it
def decode_max_hs_output(d1, d2, u, v, maxhs_answer, name='test'):
    answer = open(maxhs_answer, 'r')
    output_human = open(name + 'output_humain', 'w')
    positives = answer.readlines()
    # positives = [re.sub('[^0,1]', '', x) for x in positives]
    # positives = "".join(positives)
    good_line = [x for x in positives if x[0] == 'v'][0]
    positives = [int(x) for x in good_line if x in ['0', '1']]
    x_dict, y_dict = make_corresp_dictionnaries(u, v)
    output_human.write("Littéraux vrais :")
    for (i, truth_value) in enumerate(positives):
        if truth_value == 1:
            pos = invert_dic(x_dict, i + 1)
            if pos is not None:
                output_human.write(f"x_{pos} (match entre les positions {pos})\n")
            else:
                pos = invert_dic(y_dict, i + 1)
                if pos is not None:
                    a, b = invert_dic(y_dict, i + 1)
                    output_human.write(f"y_{a, b} ({invert_dic(d1,a)} renommé en {invert_dic(d2,b)})\n")
                else:
                    print('warning : too many variables')


def compare_pieces(f1, f2, final_output_name, final_output_dir = 'corpus11paires'):
    piece1 = minidom.parse(open(f1, 'rb'))
    piece2 = minidom.parse(open(f2, 'rb'))
    title1,title2 = unidecode(play_parsing.get_title(piece1)), unidecode(play_parsing.get_title(piece2))
    acts1, acts2 = play_parsing.get_all_acts_dialogues(piece1), play_parsing.get_all_acts_dialogues(piece2)
    if len(acts1) != len(acts2):
        raise ValueError(f"{title1} and {title2} do not have the same number of acts ({len(acts1)} and {len(acts2)})")
    for (act_number,(a1, a2)) in enumerate(zip(acts1, acts2)):
        input_name, d1, d2, normalized_a1, normalized_a2 = encode_scenes(a1, a2, f'{title1} et {title2}_acte{act_number}')
        output_name = f"{input_name}_acte{act_number}_output"
        os.system(f"/usr/local/MaxHS-2021_eval/build/release/bin/maxhs -printSoln '{input_name}' > '{output_name}'")
        time.sleep(7)
        decode_max_hs_output(d1, d2, normalized_a1, normalized_a2, output_name, os.path.join(f'{final_output_dir}', f'Comparaison {final_output_name}  actes {act_number}'))
        print(f'done for act {act_number}')


def compare_pieces_corpus(folder):
    """folder must contain folders with a pair of plays to compare """
    folders = os.listdir(folder)
    for f in folders:
        pair_name = f
        plays = os.listdir(f)
        play1, play2 = plays[0], plays[1]
        compare_pieces(play1, play2, pair_name)




scene_test = ['PersoA', 'PersoB']


def change_slightly(scene,k):
    for i in range(k):
        insert_or_pop = random.randint(0,1)
        indice = random.randint(0, len(scene)-1)
        if insert_or_pop == 0:
            scene.pop(indice)
        else:
            random_char = random.choice(scene)
            scene.insert(indice, random_char)
    return scene


Medee_v1 = 'ABABABABABBCDCDCDCDEDEFDFDFBFBFBFBF' # 'TMTMTMTMTMMAUAUAUAUOUOJUJUJMJMJMJMJ'
dico_medee1 = {'Theandre':'A', 'Medee':'B', 'Cleandre': 'C','Perso_U': 'D','Jason':'F', 'Cleone':'E', 'Choeur':'K'}
Medee_v2 = 'ABABCBCBCBCBCBDCDCECEDCCBCBCFCFCFFCFBFBFBFB'#'TMTMUMUMUMUMUMKUKUOUOKUUMUMUJUJUJJUJMJMJMJM'
dico_medee2 = {'Theandre':'A', 'Medee':'B', 'Cleandre': 'C','Perso_U': 'C','Jason':'F', 'Cleone':'E', 'Choeur':'D'}
# changed_marianne_1 = change_slightly(scene_2_marianne, 1)
# changed_marianne_3 = change_slightly(scene_2_marianne, 3)

if __name__ == "__main__":
    # u = "AB"
    # v = "AB"
    # s = make_sat_instance(["Test"], u,v)
    # print(s)
    # encode_scenes(scene_2_marianne, scene_2_marianne, 'marianne_entree_identique')
    # s = make_sat_instance(["Comparaisons acte V de Medee"], Medee_v1, Medee_v2)
    # f = open('medeeV_input_maxHS','w')
    # f.write(s)
    # f.close()
    #decode_max_hs_output(dico_medee1,dico_medee1,Medee_v1,Medee_v2, "MedeeV")
    # doctest.testmod()
    # print('ok')
    didon1 = os.path.join('corpustestdidon', '108_JODELLE_DIDON.xml')
    didon2 = os.path.join('corpustestdidon', '218_Théâtre complet. Tome I - Didon se sacrifiant.xml')
    compare_pieces(didon1, didon2)
