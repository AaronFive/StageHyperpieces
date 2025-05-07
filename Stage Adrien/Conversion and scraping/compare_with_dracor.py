import string
import pandas as pd

"""This file implements a quick way to gauge the number of plays in common between theatredoc and dracor 
Play titles and author names between both corpora are normalized, and then checked for exact match against each other."""

# Last result gave 387 common plays (done on 19/11/2024)
metadata_theatredoc = 'metadata theatredoc normalized.csv'
metadata_dracor = '../../fredracor-metadata.csv'

theatredoc_df = pd.read_csv(metadata_theatredoc)
dracor_df = pd.read_csv(metadata_dracor)

dracor_df = dracor_df[['title', 'firstAuthor']]
theatredoc_df = theatredoc_df[['Title', 'Author']]


# Comment comparer les deux :
# Pour chaque titre : voir s'il y en a un exactement pareil dans l'autre... vérifier si les auteurs/dates sont similaires ?
# On va commencer par un matching exact en faisant juste un join des dfs


def normalize_title(title):
    """ Normalize all titles with uppercase letter at the start of each word and removes punctuation"""
    title = ''.join(char for char in title if char not in string.punctuation)
    words = title.split()
    words = [word.lower().capitalize() for word in words]
    return ' '.join(words)


def normalize_author_theatredoc(author_name):
    author_parts = author_name.split()
    author_parts = [x for x in author_parts if x.isupper()]
    uppercase_author_name = ' '.join(author_parts)
    normalized_author = normalize_title(uppercase_author_name)
    return normalized_author


def normalize_author_dracor(author_name):
    return normalize_title(author_name)


theatredoc_df['Title'] = theatredoc_df['Title'].apply(normalize_title)
theatredoc_df['Author'] = theatredoc_df['Author'].apply(normalize_author_theatredoc)
dracor_df['title'] = dracor_df['title'].apply(normalize_title)
dracor_df['firstAuthor'] = dracor_df['firstAuthor'].apply(normalize_author_dracor)

if __name__ == "__main__":
    merge = pd.merge(theatredoc_df, dracor_df, left_on=['Title','Author'], right_on=['title', 'firstAuthor'])
    merge.to_csv('common_plays_dracor_theatredoc_v1.csv')
    print(len(merge))
