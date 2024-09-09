import io, itertools, time
import multiprocessing
import os
import csv
import queue
from typing import List
from itertools import count
from multiprocessing import Process
from xml.dom import minidom
from unidecode import unidecode
from Levenshtein import distance
from polyleven import levenshtein as bounded_distance
import sat_instance
from play_parsing import get_all_acts_dialogues
from utils import get_title, normalize_scene
from sat_instance import encode_scenes,invert_dic
from quick_comparison import fieldnames
from basic_utilities import characterList,stringToIntegerList
"""
    fpt_alphabet_size v1.0, 2022-12-05
    Solving the parameterized matching problem between two strings,
    with Levenshtein distance and injective variable renaming functions
    Copyright (C) 2022 - Philippe Gambette
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
"""



def allPermutationsAfterElement(lst, i):
    """Returns the list of all permutations of lst keeping the i first elements in place.

    Args:
        lst (list): list of elements to permute
        i (int): Empty set to fill with all sources.

    Returns:
        list: List of permutations
    """
    result = []
    if i == len(lst) - 1:
        result = [lst]
    else:
        for j in range(i, len(lst)):
            list2 = lst.copy()
            list2[i] = lst[j]
            list2[j] = lst[i]
            permutations = allPermutationsAfterElement(list2, i + 1)
            for permutation in permutations:
                result.append(permutation)
    return result


def allPermutations(lst):
    """Returns the list of all permutations of lst """
    return allPermutationsAfterElement(lst, 0)


def allSubsets(s, n):
    """Return all subsets of s of size n """
    return list(itertools.combinations(s, n))


def buildString(integerList, characterIntegerList):
    """Inverse of stringToIntegerList. Rebuilds the string given in integerList using the translation given by characterIntegerList.
    Args:
        integerList(list): a string to be rebuilt, written as a list of integers
        characterIntegerList(list): a list of integers appearing in integerList
    Returns:
        str: integerList with integers replaced by letters of the alphabet.
    """
    characterTransformation = {}
    characterNb = 0
    string = ""
    for c in characterIntegerList:
        characterTransformation[c] = chr(65 + characterNb)
        characterNb += 1

    for c in range(0, len(integerList)):
        if c not in characterTransformation:
            characterTransformation[c] = chr(65 + characterNb)
            characterNb += 1
    for i in integerList:
        string += characterTransformation[i]
    return string


# FPT algorithm in the size of the alphabets of the two input strings
# Complexity if s1 is the size of the smallest alphabet of the two input strings
# and s2 is the size of the alphabet of the other input strings: 
# O(s1! * A(s2, s1) * poly(size of input strings)) where A(n,k) is the number of arrangements of k elements among n
def parameterizedAlignment(a, b, queue, pair_name=None, heuristic=None ):
    """Returns the Levenshtein parameterized distance between `a` and `b` if the variables are linked by an injection.
    Sends details about the instance up queue to be logged later.
    Args:
        a (str): string to be compared
        b (str): string to be compared
        queue(multiprocessing.queues.Queue) : queue used to pass results to parent caller
                characterIntegerList(list): a list of integers appearing in integerList
        pair_name(str): Name describing the instance, for logging purposes.
    Returns:
        str: integerList with integers replaced by letters of the alphabet.
    """
    startTime = time.time()

    # Put the smallest string in a
    aCharacterList = characterList(a)
    bCharacterList = characterList(b)

    if len(aCharacterList) > len(bCharacterList):
        a, b = b, a
        aCharacterList = characterList(a)
        bCharacterList = characterList(b)
    allSubs = allSubsets(set(stringToIntegerList(b)), len(aCharacterList))
    aIntegerList = stringToIntegerList(a)
    bIntegerList = stringToIntegerList(b)
    aCharacterIntegerList = list(range(0, len(aCharacterList)))
    bCharacterIntegerList = list(range(0, len(bCharacterList)))
    bCharacterIntegerListSubsets = allSubsets(set(bCharacterIntegerList), len(aCharacterList))

    # Build all permutations of the reference character list
    permutations = allPermutations(aCharacterIntegerList)

    # For all permutations, compute the Levenshtein distance with all subsets
    # of the characters of the other string of the reference string's size
    if heuristic is None:
        smallestDistance = max(len(b), len(a))
        bestTransformedA = ""
        bestPerm = []
        bestSubset = []
    else:
        guessed_distance, guessed_perm = heuristic(a,b)
        smallestDistance = guessed_distance
        bestTransformedA = buildString(aIntegerList, guessed_perm)
        bestPerm = guessed_perm
        bestSubset = []
    for perm in permutations:
        transformedA = buildString(aIntegerList, perm)
        for sub in allSubs:
            transformedB = buildString(bIntegerList, sub)
            # below, weights=(1,1,1) for classical Levenshtein distance, weights=(1,1,10) for deletion/insertion
            # distance
            # dist = distance(transformedA, transformedB, weights=(1, 1, 1))
            # In this version I used the distance function from polyleven module, that takes into account a potential upper bound for the distance.
            dist = bounded_distance(transformedA, transformedB,smallestDistance)
            if dist < smallestDistance:
                smallestDistance = dist
                bestTransformedA = transformedA
                bestTransformedB = transformedB
                bestPerm = perm
                bestSubset = sub

    print("Smallest distance: " + str(smallestDistance))
    act_number = pair_name[-1]
    csv_row = {'Pair name': pair_name[:-2], 'Act Number': act_number, 'Distance': str(smallestDistance), 'Input_1': a,
               'Input_2': b,
               'Input 1 length': len(a), 'Input 2 length': len(b),
               'Input 1 renamed': bestTransformedA,
               "Renaming": list(zip(bestPerm, bestSubset)),
               'Computing time': str(time.time() - startTime)}
    # "bestSubset": str(bestSubset), "bestPerm": str(bestPerm),"renamed input 2": bestTransformedB,
    for x in csv_row:
        queue.put((x, csv_row[x]))
    return smallestDistance


def encode_scenes(scene1, scene2):
    """Given two files of plays, encodes them as parameterized words and saves the equivalences in dictionnaries."""
    u, d1 = normalize_scene(scene1, True)
    v, d2 = normalize_scene(scene2, True)
    return '', d1, d2, u, v


def keep_most_frequents(l, max_to_keep):
    # Dictionnary of letters occurence
    character_count = dict()
    for character in l:
        character_count[character] = character_count.get(character, 0) + 1
    # Getting the list of occurences and sorting it
    occurence_list = sorted([(character, character_count[character]) for character in character_count], reverse=True,
                            key=lambda x: x[1])
    character_to_keep = occurence_list[:max_to_keep]
    character_to_keep = [x[0] for x in character_to_keep]

    def replace_characters(x):
        if x not in character_to_keep:
            return f"Personnages_secondaires({max_to_keep})"
        else:
            return x

    return [replace_characters(x) for x in l]


def compare_pieces_content(acts1, acts2, pair_name, gwriter, timeout=60, fieldnames = fieldnames):
    """Given two plays, compare both plays act by act, and logs the results in a csv file.
    output csv and name of columns are given by gwriter and fieldnames.
    Computation are done with the specified timeout."""
    # Comparing number of acts of each play
    if len(acts1) != len(acts2):
        m = min(len(acts1), len(acts2))
        acts1, acts2 = acts1[:m], acts2[:m]
        print(f" Warning : {pair_name} do not have the same number of acts. Comparing only first {m} acts")

    # We compare act 1 with act 1, 2 with 2, etc
    for (act_number, (a1, a2)) in enumerate(zip(acts1, acts2)):
        # New act
        print(f'Act {act_number + 1}')

        # Encoding the acts as parameterized words
        input_name, d1, d2, normalized_a1, normalized_a2 = encode_scenes(a1, a2)
        smallest_alphabet_size = min(len(d1), len(d2))

        # Now we call the FPT algorithm
        # We execute it with a timeout. To do so, we use the multiprocessing library.
        print('Calling resolution algorithm ...')
        # What we want to get back from the call is a dictionnary. It is however unhashable, and therefore cannot be
        # obtained directly as a return of the call. Hence we use a queue to stock all the results and only put them
        # in a dictionnary aftewards.
        queue = multiprocessing.Queue()
        p1 = Process(target=parameterizedAlignment,
                     args=(normalized_a1, normalized_a2, queue, f'{pair_name}_{act_number + 1}'), name='FPTtry')
        p1.start()
        p1.join(timeout=timeout)
        # Case 1 : call has timed out, we fill the csv accordingly
        csv_row = {"Pair name": f'{pair_name}', "Act Number": f'{act_number + 1}', "Distance": None,
                   "Normalized_Distance":None,
                   "Input_1": normalized_a1,
                   "Input_2": normalized_a2,
                   "Input 1 length": len(normalized_a1),
                   "Input 2 length": len(normalized_a2),
                   "Input 1 renamed": None,
                   "Letters_play1":d1,
                   "Letters_play2": d2,
                   "Renaming": None}
        renaming = None
        if p1.exitcode is None:
            p1.kill()
            # TODO: Log the best solution found instead
            csv_row['Computing time'] = 'TIMEOUT'
            # "bestSubset": None,"bestPerm": None,"renamed input 2": None,
        # Case 2 : Success
        else:
            p1.kill()
            for i in range(len(fieldnames)):
                k, v = queue.get()
                if k == "Renaming":
                    renaming = v
                else:
                    csv_row[k] = v
            queue.empty()
        if renaming is not None:
            renaming = [(sat_instance.invert_dic(d1,chr(x+65)), sat_instance.invert_dic(d2,chr(y+65))) for (x,y) in renaming]
            renaming_string = str(renaming)
            csv_row["Renaming"] = renaming_string
            size_u,size_v = int(csv_row["Input 1 length"]), int(csv_row["Input 2 length"])
            csv_row["Normalized_Distance"] = round((int(csv_row["Distance"]) - abs(size_u-size_v) ) / (min(size_u,size_v)),2)
            print(csv_row["Computing time"])
        gwriter.writerow(csv_row)
        print('written')
        print(f'done for act {act_number + 1}')

def compare_heuristics(acts1, acts2, pair_name,  heuristic, timeout=60):
    """Given two plays, compare both plays act by act, and logs the results in a csv file.
    output csv and name of columns are given by gwriter and fieldnames.
    Computation are done with the specified timeout."""

    # We compare act 1 with act 1, 2 with 2, etc
    for (act_number, (a1, a2)) in enumerate(zip(acts1, acts2)):
        print(len(acts1))
        # New act
        print(f'Act {act_number + 1}')

        # Encoding the acts as parameterized words
        input_name, d1, d2, normalized_a1, normalized_a2 = encode_scenes(a1, a2)

        #Collecting initial heuristic guess
        guessed_score, guessed_renaming = heuristic(normalized_a1, normalized_a2)
        # print(guessed_renaming)
        guessed_renaming = [(sat_instance.invert_dic(d1,x), sat_instance.invert_dic(d2,y)) for (x,y) in guessed_renaming]

        # Now we call the FPT algorithm with a heuristic
        # We execute it with a timeout. To do so, we use the multiprocessing library.
        print('Calling resolution algorithm ...')
        # What we want to get back from the call is a dictionnary. It is however unhashable, and therefore cannot be
        # obtained directly as a return of the call. Hence we use a queue to stock all the results and only put them
        # in a dictionnary aftewards.
        queue = multiprocessing.Queue()
        p1 = Process(target=parameterizedAlignment,
                     args=(normalized_a1, normalized_a2, queue,  f'{pair_name}_{act_number + 1}', heuristic),  name='FPTtry')
        p1.start()
        p1.join(timeout=timeout)
        #["True Distance", "True Renaming", "Input_1", "Input_2", "Computing time"]
        # Case 1 : call has timed out, we fill the csv accordingly
        true_renaming, true_distance = None,None
        if p1.exitcode is None:
            p1.kill()
            computing_time = timeout
            normalized_distance = None
        # Case 2 : Success
        else:
            p1.kill()
            print('here')
            for i in range(queue.qsize()):
                print(f'getting {i}')
                k, v = queue.get()
                print(f' Got{i}')
                if k == "Renaming":
                    true_renaming = v
                elif k == "Computing time":
                    computing_time = v
                elif k == "Distance":
                    true_distance = v
                elif k == "Input 1 length":
                    size_u = v
                elif k == "Input 2 length":
                    size_v = v
                else:
                    print(k,v)
            print('for done')
        queue.empty()
        print('done')
        if true_renaming is not None:
            true_renaming = [(sat_instance.invert_dic(d1,chr(x+65)), sat_instance.invert_dic(d2,chr(y+65))) for (x,y) in true_renaming]
            normalized_distance = round((int(true_distance) - abs(size_u-size_v) ) / (min(size_u,size_v)),2)
    return guessed_score, guessed_renaming, computing_time, true_renaming, true_distance, normalized_distance
def compare_pieces(f1, f2, pair_name, gwriter, timeout=60, split_by_act = True):
    """Given two files of plays, run the parameterized matching comparison and logs the results.
    Logs the result in a csv file given by gwriter.
    Args:
        f1 (str): Path to first play
        f2 (str): Path to second play
        pair_name(str): Name of the two plays
        gwriter(csv.DictWriter) : Writer for the csv output
        timeout(int): Only used for logging purposes, when called with a timeout
        split_by_act (bool): should acts be treated separately or as one block
        """
    # Getting plays, titles, and acts
    piece1 = minidom.parse(open(f1, 'rb'))
    piece2 = minidom.parse(open(f2, 'rb'))
    # title1, title2 = unidecode(get_title(piece1)), unidecode(get_title(piece2))
    acts1, acts2 = get_all_acts_dialogues(piece1), get_all_acts_dialogues(piece2)
    if not split_by_act:
        acts1 = [[replique for act in acts1 for replique in act]]
        acts2 = [[replique for act in acts2 for replique in act]]
    compare_pieces_content(acts1, acts2, pair_name, gwriter, timeout)


def compare_pieces_corpus(folder, timeout=60, final_output_dir='Resultats FPT'):
    """Compare all pairs of plays in the specified folder by iterating compare_piece.
    Logs all the results in a csv file.
    Args:
        folder(str):path of the folder containing plays to compare. Must be a folder of folders containing 2 plays.
        timeout(int): How long to compute on each pair in seconds
        final_output_dir(str): path of the directory where to write the output
        """
    # Todo : Also log number of characters per play
    # Getting the folder of pairs to compare
    folders = os.listdir(folder)
    final_output_dir = os.path.join(os.getcwd(), final_output_dir)
    # Creating output csv file
    output_csv = open(os.path.join(final_output_dir, f'FPTcomparisons_tm{timeout}.csv'), 'w+')
    fieldnames = ["Pair name", "Distance", "Input_1", "Input_2", 'Input 1 length', 'Input 2 length', "Input 1 renamed",
                  "Renaming", 'Computing time']
    # "bestSubset", "bestPerm",, "renamed input 2"
    gwriter = csv.DictWriter(output_csv, fieldnames=fieldnames)
    gwriter.writeheader()

    folders.sort(key=lambda x: os.path.getsize(os.path.join(folder, x)))
    for f in folders:
        pair_name = f
        print(f'Comparing {pair_name}')
        folder_path = os.path.join(folder, f)
        if os.path.isdir(folder_path):
            plays = os.listdir(folder_path)
            play1, play2 = os.path.join(folder_path, plays[0]), os.path.join(folder_path, plays[1])
            compare_pieces(play1, play2, pair_name, gwriter, timeout)


# TODO : try heuristics on the whole corpora
if __name__ == "__main__":
    # Parameters
    # corpus_name = 'corpus11paires'
    # timeout = 1200
    #
    # corpus_folder = os.path.join(os.getcwd(), corpus_name)
    # print(f'comparing all plays of folder {corpus_name}')
    # compare_pieces_corpus(corpus_folder, timeout)
    # print('Done')

    # Temp : nostra/zoro
    # full_nostra = [char for scene in nostra_pp for char in scene ]
    # full_zoro = [char for scene in zoro_pp for char in scene ]
    # _,d_zoro,d_nostra,zoro_pword,nostra_pword = encode_scenes(full_nostra, full_zoro)
    # print(parameterizedAlignment(nostra_pword, zoro_pword,queue.Queue()))
    pass