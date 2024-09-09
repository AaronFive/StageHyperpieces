"""Generic Python function, mostly on strings, used throughout the project"""
import pickle
import os

OUTPUT_DIR = "Outputs\\To categorize"
PICKLE_DIR = "Data\\Pickled saves"

title_dict = pickle.load(open(os.path.join(PICKLE_DIR, 'id_title_correspondance.pkl'), 'rb'))


def flatten_list(input_list):
    """Recursively flattens a list that contains lists and other elements.
    Args:
        input_list(list) : A list that contains list and/or other elements
    Returns:
        list : The flattened list"""
    result = []
    for item in input_list:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, list):
            result.extend(flatten_list(item))
    return result


def flatten_list_of_list(play):
    res = []
    for act in play:
        new_act = []
        for scene in act:
            for x in scene:
                new_act.append(x)
        res.append(new_act)
    return res


def characterList(a):
    """Returns the set of characters of string `a`"""
    charList = []
    for i in range(0, len(a)):
        if a[i] not in charList:
            charList.append(a[i])
    return charList


def stringToIntegerList(a):
    """Returns a as a list of integers, each letter corresponding to an integer, ordered by first appearance in a.
    Args:
        a(str)
    Returns:
        list: a as a list of integers
    """
    integerList = []
    charSet = {}
    charNb = 0
    for letter in a:
        if letter not in charSet:
            charSet[letter] = charNb
            charNb += 1
        integerList.append(charSet[letter])
    return integerList


def count_letters(s):
    """Given a string, returns a dictionary of the occurences of each letter
    Args:
        s(str)
    Returns:
        dict : the number of occurence of each letter of s
    """
    d = dict()
    for x in s:
        d[x] = d.get(x, 0) + 1
    return d


def pickle_file(object, file_name, output_directory=PICKLE_DIR):
    """Given an object, saves it using pickle in the given file.
    Args:
        object(any): The object to save
        file_name(str): The name of the save file to use. Should not end by .pkl, this is added by the function
        output_directory(str): The directory where to save"""
    save_file = open(os.path.join(output_directory, f"{file_name}.pkl"), 'wb')
    pickle.dump(object, save_file)
    print(f"{file_name} saved.")
