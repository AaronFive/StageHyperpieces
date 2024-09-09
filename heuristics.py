import csv
import os
from polyleven import levenshtein
from collections import defaultdict
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
from basic_utilities import flatten_list,characterList,stringToIntegerList, count_letters

output_folder = 'Quick comparisons/Comparison results'
input_folder = 'Quick comparisons'
sources_couples_file = 'Quick comparisons/Couples-test.csv' #Quick comparisons/Imitation, création au théâtre - Couples.csv'
sources_plays_file = 'Quick comparisons/Imitation, création au théâtre - Pieces.csv'
sources_couples = open(sources_couples_file, 'r')
d = csv.DictReader(sources_couples)
sources = {x['id_piece_source']: x['id_piece_inspiree'] for x in d}
sources_plays = open(sources_plays_file, 'r', encoding='utf8')
d = csv.DictReader(sources_plays, dialect='unix')
plays = {x['id']: eval(x['succession_personnages']) for x in d if x['succession_personnages'] != ''}


def compare_heuristics(sources, heuristics, output_name, characters_to_keep=8, timeout=120):
    output_csv = open(os.path.join(output_folder, f'{output_name}.csv'), 'w', newline='', encoding='utf8')
    fieldnames = ["Pair name", "True Distance", "Normalized Distance"]
    for f in heuristics:
        heuristic_name = f.__name__
        fieldnames.append(f"Computing Time with {heuristic_name}")
        fieldnames.append(f"Value of {heuristic_name} estimation")
        fieldnames.append(f"Renaming guessed w/ {heuristic_name}")
    fieldnames = fieldnames + ["True Renaming", "Input_1", "Input_2"]
    gwriter = csv.DictWriter(output_csv, fieldnames=fieldnames)
    gwriter.writeheader()
    count_pairs = 0
    for i, play in enumerate(sources):
        print(play)
        if play in plays and sources[play] in plays:
            count_pairs += 1
            play1_name, play2_name = play, sources[play]
            pair_name = play1_name + play2_name
            acts_1, acts_2 = plays[play1_name], plays[play2_name]
            acts_1 = flatten_list(acts_1)
            acts_2 = flatten_list(acts_2)
            acts_1 = [fpt_alphabet_size.keep_most_frequents(acts_1, characters_to_keep)]
            acts_2 = [fpt_alphabet_size.keep_most_frequents(acts_2, characters_to_keep)]
            csv_row = {}
            csv_row["Pair name"] = pair_name
            csv_row["Input_1"] = acts_1
            csv_row["Input_2"] = acts_2
            for h in heuristics:
                guessed_score, guessed_renaming, computing_time, true_renaming, true_distance, normalized_distance = fpt_alphabet_size.compare_heuristics(
                    acts_1, acts_2, pair_name, h, timeout)
                if normalized_distance is not None:
                    csv_row["Normalized Distance"] = normalized_distance
                    csv_row["True Distance"] = true_distance
                    csv_row["True Renaming"] = true_renaming
                heuristic_name = h.__name__
                csv_row[f"Computing Time with {heuristic_name}"] = computing_time
                csv_row[f"Value of {heuristic_name} estimation"] = guessed_score
                csv_row[f"Renaming guessed w/ {heuristic_name}"] = guessed_renaming
            gwriter.writerow(csv_row)




def get_renaming_most_frequent(u, v):
    """Given two strings u and v, order letter by frequences, and returns a mapping of most frequent letters to most frequent letters
    The second renaming dcitionnary handles the case where u has more letters than v, and fixes the images of the unchanged letters."""
    letter_occurence_u, letter_occurence_v = count_letters(u), count_letters(v)
    letter_occurence_u, letter_occurence_v = list(letter_occurence_u.items()), list(letter_occurence_v.items())
    letter_occurence_u.sort(key=lambda x: -x[1])
    letter_occurence_v.sort(key=lambda x: -x[1])
    letter_occurence_u = [x[0] for x in letter_occurence_u]
    letter_occurence_v = [x[0] for x in letter_occurence_v]
    renaming = zip(letter_occurence_u, letter_occurence_v)
    renam_dict = dict()
    for x, y in renaming:
        renam_dict[x] = y
    extended_renam_dict = dict()
    for x in u:
        if x not in renam_dict:
            extended_renam_dict[x] = x
        else:
            extended_renam_dict[x] = renam_dict[x]
    return renam_dict, extended_renam_dict


def simple_frequence_heuristic(u, v):
    renam_dict, extended_renam_dict = get_renaming_most_frequent(u, v)
    renamed_u = ''.join([extended_renam_dict[x] for x in u])
    return levenshtein(renamed_u, v), list(renam_dict.items())


def count_sucessions(s):
    successions_matrix = dict()
    n = len(s)
    for (i, x) in enumerate(s):
        if i != n - 1:
            next_letter = s[i + 1]
            if x not in successions_matrix:
                successions_matrix[x] = dict()
            successions_matrix[x][next_letter] = successions_matrix[x].get(next_letter, 0) + 1


def count_bigrams(s):
    # Initialize a defaultdict to store the counts of each bi-gram
    bi_gram_counts = defaultdict(int)

    # Iterate over the string, considering pairs of adjacent characters
    for i in range(len(s) - 1):
        # Concatenate the current character and the next character to form a bi-gram
        bi_gram = s[i:i + 2]
        # Increment the count of this bi-gram in the dictionary
        bi_gram_counts[bi_gram] += 1

    # Return the dictionary containing the counts of each bi-gram
    return bi_gram_counts


def is_compatible(x, y, renaming, inverse_renaming, bijective=False):
    """ Checks if renaming x to y is compatible with the current renaming"""
    if bijective:
        return (x not in renaming and y not in inverse_renaming) or (x in renaming and renaming[x] == y)
    else:
        return x not in renaming or renaming[x] == y


def successions_heuristic(u, v, bijective=True):
    """ Heuristic to find a parameterized match between u and v.
    We try to match characters going by pairs, based on the frequency."""
    # Counting occurence of each pair of characters speaking, and ordering them
    bigrams_u, bigrams_v = count_bigrams(u), count_bigrams(v)
    ordered_bigrams_u, ordered_bigrams_v = list(bigrams_u.items()), list(bigrams_v.items())
    ordered_bigrams_u.sort(key=lambda x: -x[1])
    ordered_bigrams_v.sort(key=lambda x: -x[1])

    # We construct a renaming between letters of u and v. To simplify things, for the bijective case, we keep the inverse function of the renaming
    renaming = dict()
    inverse_renaming = dict()
    # We check every bigram of u, by decreasing frequency
    for i in range(len(ordered_bigrams_u)):
        current_u_bigram = ordered_bigrams_u[i][0]
        first_u_letter, second_u_letter = current_u_bigram[0], current_u_bigram[1]
        # We check every bigram of v, by decreasing frequency
        for j in range(len(ordered_bigrams_v)):
            current_v_bigram = ordered_bigrams_u[j][0]
            first_v_letter, second_v_letter = current_v_bigram[0], current_v_bigram[1]
            # Checking to see if it is possible to extend the renaming by matching the current u bigram with the current v bigram
            if is_compatible(first_u_letter, first_v_letter, renaming, inverse_renaming, bijective):
                if is_compatible(second_u_letter, second_v_letter, renaming, inverse_renaming, bijective):
                    # If possible, we update the renamings accordingly, and remove this bigram from the list
                    renaming[first_u_letter] = first_v_letter
                    inverse_renaming[first_v_letter] = first_u_letter
                    renaming[second_u_letter] = second_v_letter
                    inverse_renaming[second_v_letter] = second_u_letter
                    ordered_bigrams_v.pop(j)
                    break  # onto the next u bigram
    # Now that we have constructed a first renaming, some letters might still not be matched. We use a simple
    # frequency renaming.
    # Preparing u and v by only keeping the unmatched letters
    unmatched_u = "".join([x for x in u if x not in renaming])
    unmatched_v = "".join([x for x in v if x not in inverse_renaming])
    if unmatched_u.strip():
        print(f"Unmatched : {unmatched_u} ")
    if unmatched_v.strip():
        print(f"Unmatched : {unmatched_v}")
    unmatched_renaming, unmatched_renaming_extended = get_renaming_most_frequent(unmatched_u, unmatched_v)

    # Merging the two renamings
    for x in unmatched_renaming:
        if x in renaming:
            print(f"Something went wrong : {x} is renamed twice")
        renaming[x] = unmatched_renaming[x]
    for x in unmatched_renaming_extended:
        renaming[x] = ''
    renamed_u = ''.join([renaming[x] for x in u])
    return levenshtein(renamed_u, v), list(renaming.items())


def make_reverse_dictionary(d):
    """Given a dictionary d representing an injective function, compute the dictionnary representing the inverse function
    Prints an error if the function is not injective"""
    inverted_dict = dict()
    for k in d:
        v = d[k]
        if v in inverted_dict:
            print(f"Warning : dictionnary is not injective ({inverted_dict[v]} and {k} point to the same value {v}")
        else:
            inverted_dict[v] = k
    return inverted_dict

def relabelIntegerList(list, relabelingDictionary):
    """Relabels the integerList using the relabelingDictionary whose images are integers from 0 to n, labeling as n+1 all characters which are not keys of the relabeling dictionary.
    Args:
        list(list): a list of integers
        relabelingDictionary(dict): a dictionary which associates integers to integers from 0 to n
    Returns:
        list: integerList containing integers from 0 to n+1
    """
    newList = []
    maxOutputLetter = 0
    for inputLetter in relabelingDictionary:
        maxOutputLetter = max(maxOutputLetter, relabelingDictionary[inputLetter])

    for char in list:
        if char in relabelingDictionary:
            newList.append(relabelingDictionary[char])
        else:
            newList.append(maxOutputLetter + 1)
    return newList


def integerListToString(list):
    """Transforms an integer list into a string with letters starting from A for 0, B for 1, etc.
    Args:
        list(list): a list of integers
    Returns:
        str: string
    """
    string = ""
    for i in list:
        string += chr(65 + i)
    return string

degreedy_heuristic(a,b):
    """Greedy heuristic to find an injection between the smallest alphabet of the two input strings to the largest alphabet, hoping to minimize the distance between the two parameterized words, computed with the "alignment" Python library (https://pypi.org/project/alignment/) with match score 2, mismatch score -1, gap opening score -2; aims to find, for each character of the first string, the remaining character of the second string which provides the best mapping with the first one, taking into account previously mapped characters and mapping all remaining characters to a single letter
        Args:
            list(list): a list of integers
        Returns:
            str: string
        """
    # print("Looking for parameterized alignment for the following strings:")
    # print(a)
    # print(b)

    # Put the string with smallest alphabet in a
    aCharacterList = characterList(a)
    bCharacterList = characterList(b)

    if len(aCharacterList) > len(bCharacterList):
        a, b = b, a
        aCharacterList = characterList(a)
        bCharacterList = characterList(b)
    aIntegerList = stringToIntegerList(a)
    bIntegerList = stringToIntegerList(b)
    aCharacterIntegerList = list(range(0, len(aCharacterList)))
    bCharacterIntegerList = list(range(0, len(bCharacterList)))
    """
    print(aCharacterIntegerList)
    print(bCharacterIntegerList)
    print(stringToIntegerList(a))
    print(stringToIntegerList(b))
    """
    aRelabelingDictionary = {}
    bRelabelingDictionary = {}

    currentlyRelabeledCharacter = 0

    # Create a vocabulary to encode the sequences for their alignment
    v = Vocabulary()

    # While the set of remaining characters of the first string to map with characters of the second string is not empty
    while len(aCharacterIntegerList) > 0:
        aBestRelabelingDictionary = aRelabelingDictionary.copy()
        bBestRelabelingDictionary = bRelabelingDictionary.copy()
        bestChar1 = aCharacterIntegerList[0]
        bestChar2 = bCharacterIntegerList[0]
        aBestRelabelingDictionary[bestChar1] = currentlyRelabeledCharacter
        bBestRelabelingDictionary[bestChar2] = currentlyRelabeledCharacter
        highestNumberOfCurrentlyRelabeledCharacters = 0
        # Test each character of the first string to check if it provides the best alignment with some character of the second string
        for char1 in aCharacterIntegerList:
            aTempRelabelingDictionary = aRelabelingDictionary.copy()
            aTempRelabelingDictionary[char1] = currentlyRelabeledCharacter
            # print("Relabeling first word with dictionary " + str(aTempRelabelingDictionary))
            tempA = relabelIntegerList(stringToIntegerList(a), aTempRelabelingDictionary)
            aTempString = integerListToString(tempA)
            # print(aTempString)
            # Try to map each character of the second string with char1
            for char2 in bCharacterIntegerList:
                bTempRelabelingDictionary = bRelabelingDictionary.copy()
                bTempRelabelingDictionary[char2] = currentlyRelabeledCharacter;
                # print("Relabeling second word with dictionary " + str(bTempRelabelingDictionary))
                tempB = relabelIntegerList(stringToIntegerList(b), bTempRelabelingDictionary)
                bTempString = integerListToString(tempB)
                # print(bTempString)

                # Encode the sequences to align them.
                aEncoded = v.encodeSequence(Sequence([*aTempString]))
                bEncoded = v.encodeSequence(Sequence([*bTempString]))

                # Create a scoring and align the sequences using global aligner.
                scoring = SimpleScoring(2, -1)
                aligner = GlobalSequenceAligner(scoring, -2)
                score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)

                # Iterate over optimal alignments to count how often the currently relabeled character is properly aligned
                for encoded in encodeds:
                    alignment = v.decodeSequenceAlignment(encoded)
                    alignmentString = str(alignment).split("\n")
                    # Count the number of aligned currently relabeled characters (occurrences of the character corresponding to char1)
                    numberOfCurrentlyRelabeledCharacters = 0
                    charNumber = 0
                    for char in alignmentString[0]:
                        if ord(char) == 65 + currentlyRelabeledCharacter and ord(
                                alignmentString[1][charNumber]) == 65 + currentlyRelabeledCharacter:
                            numberOfCurrentlyRelabeledCharacters += 1
                        charNumber += 1
                    if (numberOfCurrentlyRelabeledCharacters > highestNumberOfCurrentlyRelabeledCharacters):
                        # Update information about the best character mapping found so far : char1 => char2
                        highestNumberOfCurrentlyRelabeledCharacters = numberOfCurrentlyRelabeledCharacters
                        aBestRelabelingDictionary = aTempRelabelingDictionary.copy()
                        bBestRelabelingDictionary = bTempRelabelingDictionary.copy()
                        bestChar1 = char1
                        bestChar2 = char2

                        # print("Best character mapping found so far for first string: " + str(aBestRelabelingDictionary))
                        # print(
                        #     "Best character mapping found so far for second string: " + str(bBestRelabelingDictionary))
                        # print("Number of aligned currently relabeled characters: " + str(
                        #     highestNumberOfCurrentlyRelabeledCharacters))
                        # print(alignment)
                        # print('Alignment score:', alignment.score)
                        print('Percent identity:', alignment.percentIdentity())

        # Prepare the next step, update the relabeling dictionary with the best pair char1 => char2 found in this step
        currentlyRelabeledCharacter += 1
        aRelabelingDictionary = aBestRelabelingDictionary.copy()
        bRelabelingDictionary = bBestRelabelingDictionary.copy()
        aCharacterIntegerList.remove(bestChar1)
        bCharacterIntegerList.remove(bestChar2)
        """
        print(aCharacterIntegerList)
        print(bCharacterIntegerList)

        print("Best character mapping found so far for first string: " + str(aRelabelingDictionary))
        print("Best character mapping found so far for second string: " + str(bRelabelingDictionary))
        """
    tempA = relabelIntegerList(stringToIntegerList(a), aRelabelingDictionary)
    tempB = relabelIntegerList(stringToIntegerList(b), bRelabelingDictionary)
    aTempString = integerListToString(tempA)
    bTempString = integerListToString(tempB)
    print("Strings obtained after applying the character relabeling found by the heuristic")
    print(aTempString)
    print(bTempString)
    aEncoded = v.encodeSequence(Sequence([*aTempString]))
    bEncoded = v.encodeSequence(Sequence([*bTempString]))

    # Get the renaming in the correct format
    print(aRelabelingDictionary)
    print(bRelabelingDictionary)
    renaming = dict()
    inverted_RenamingA, inverted_RenamingB = make_reverse_dictionary(aRelabelingDictionary), make_reverse_dictionary(bRelabelingDictionary)
    for i in inverted_RenamingA:
        renaming[inverted_RenamingA[i]] = inverted_RenamingB[i]
    # Create a scoring and align the sequences using global aligner.
    scoring = SimpleScoring(2, -1)
    aligner = GlobalSequenceAligner(scoring, -2)
    score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)
    alignment = v.decodeSequenceAlignment(encodeds[0])
    # print("Best alignment found (only a substring of the longest string may appear below):")
    # print(alignment)
    # print('Alignment score:', alignment.score)
    # print('Percent identity:', alignment.percentIdentity())
    renaming_table = str.maketrans(renaming)
    print(renaming)
    return levenshtein(a.translate(renaming_table), b.translate(renaming_table)), list(renaming.items())

if __name__ == "__main__":
    heuristics = [greedy_heuristic, successions_heuristic, simple_frequence_heuristic]
    compare_heuristics(sources, heuristics, 'heuristic_comparison_test.csv')
    # f1 = "Corpus\\Corpus Dramacode\\ouville_espritfolet.xml"
    # f2 = 'cal000025-la-dama-duende.tei.xml'
    # both_compare_pieces(f1,f2, 'dama_full_text', 0, 600, False)
