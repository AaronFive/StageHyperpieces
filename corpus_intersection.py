import os
import pickle
import re
import sys
from collections import defaultdict

import pandas as pd
from Levenshtein import distance

from compare_titles import vectorize, distance_title
from play_parsing import get_play_from_file, get_full_text, get_author
from utils import get_title, normalize_title

corpus_Dracor = None
corpus_bibdramatique = "Corpus\\Corpus Bibdramatique"
corpus_TC = "Corpus\\CorpusTC"
corpus_TD = "Corpus\\CorpusTD"


def compare_corpus_vectors(corpus_1_name, corpus_1_path, corpus_2_name, corpus_2_path, cutoff=0.98):
    """ Compares corpus intersection by using Flaubert embeddings"""
    data = {
        f'{corpus_1_name} play': [],
        f'In {corpus_2_name}': [],
        f'Best match in {corpus_2_name}': [],
        'Score': []
    }

    corpus_dict = {corpus_1_name: corpus_1_path, corpus_2_name: corpus_2_path}
    title_dicts = {corpus_1_name: dict(), corpus_2_name: dict()}
    # Vectoriser les titres des 2 corpus
    for corp_name in corpus_dict:
        corp_path = corpus_dict[corp_name]
        for file in os.listdir(corp_path):
            path = os.path.join(corp_path, file)
            play = get_play_from_file(path)
            title = normalize_title(get_title(play))
            print(f"Vectorizing {title}")
            vector = vectorize(title)
            title_dicts[corp_name][title] = vector
    try:
        save_file = f'Outputs\\Intersection of corpus\\{corpus_1_name}_vs_{corpus_2_name}_title_dict.pkl'
        pickle.dump(title_dicts, open(save_file, 'wb'))
    except:
        pass

    # Double boucle pour trouver les meilleurs matchs

    for title_1 in title_dicts[corpus_1_name]:
        print(title_1)
        best_match_title = None
        best_score = -1
        found_match = False
        for title_2 in title_dicts[corpus_2_name]:
            similarity_score = distance_title(title_dicts[corpus_1_name][title_1], title_dicts[corpus_2_name][title_2])
            if similarity_score > best_score:
                best_score = similarity_score
                best_match_title = title_2
                found_match = True
            if best_score > cutoff:
                break

        # Add data to the lists for DataFrame
        data[f'{corpus_1_name} play'].append(title_1)
        data[f'Best match in {corpus_2_name}'].append(best_match_title)
        data['Score'].append(best_score)
        if found_match and best_score > 0.98:
            data[f'In {corpus_2_name}'].append(True)
        else:
            data[f'In {corpus_2_name}'].append(False)

        # Save DataFrame to CSV
    df = pd.DataFrame(data)
    csv_filename = f'Outputs\\Intersection of corpus\\{corpus_1_name}_vs_{corpus_2_name}_NLP_comparison.csv'
    df.to_csv(csv_filename, index=False)

    # Return the size of the intersection and some metadata
    intersection_size = df[df[f'In {corpus_2_name}']].shape[0]
    return intersection_size


# Objectif : récupérer les textes bruts puis faire du Levensthein
# Essayer avec tous les autres textes, jusqu'à en trouver un qui soit close àun certain cutoff prêt
# Genre 0.99 ù taille du texte ?


def compare_corpus(corpus_1_name, corpus_1_path, corpus_2_name, corpus_2_path, preprocess_function, comparison_function,
                   cutoff=0.9):
    """ Computes the intersection between the corpus 1 and the corpus 2, given as lists. """
    data = {
        f'{corpus_1_name} play': [],
        f'{corpus_1_name} author': [],
        f'In {corpus_2_name}': [],
        f'Best match in {corpus_2_name}': [],
        f'Best author match in {corpus_2_name}': [],
        'Score': []
    }
    corpus1_list = os.listdir(corpus_1_path)
    corpus2_list = os.listdir(corpus_2_path)

    # Pre-processing

    corpus_dict = {corpus_1_name: corpus_1_path, corpus_2_name: corpus_2_path}
    title_dicts = {corpus_1_name: dict(), corpus_2_name: dict()}
    # Vectoriser les titres des 2 corpus
    title_seen = 0
    for corp_name in corpus_dict:
        corp_path = corpus_dict[corp_name]
        print(f"Pre-processing ({title_seen})")
        for file in os.listdir(corp_path):
            path = os.path.join(corp_path, file)
            play = get_play_from_file(path)
            title = get_title(play)
            author = get_author(play)
            title_seen += 1
            result = preprocess_function(title, author)

            # print(result)
            title = f"{title_seen} - {title}"
            title_dicts[corp_name][title] = result
    print("Pre-processing done")

    # Comparing both corpora
    for title_1 in title_dicts[corpus_1_name]:
        print(title_1)
        best_match_title, best_match_author, best_score, found_match = None, None, -1, False
        author_1 = title_dicts[corpus_1_name][title_1][1]
        for title_2 in title_dicts[corpus_2_name]:
            similarity_score = comparison_function(title_dicts[corpus_1_name][title_1],
                                                   title_dicts[corpus_2_name][title_2])
            if similarity_score > best_score:
                best_score = similarity_score
                best_match_title = title_2
                best_match_author = title_dicts[corpus_2_name][title_2][1]  # TMP
            if best_score > cutoff:
                found_match = True
                break

        # Add data to the lists for DataFrame
        data[f'{corpus_1_name} play'].append(title_1)
        data[f'{corpus_1_name} author'].append(author_1)
        data[f'Best match in {corpus_2_name}'].append(best_match_title)
        data[f'Best author match in {corpus_2_name}'].append(best_match_author)
        data['Score'].append(best_score)
        data[f'In {corpus_2_name}'].append(found_match)

    # Save DataFrame to CSV
    df = pd.DataFrame(data)
    csv_filename = f'Outputs\\Intersection of corpus\\{corpus_1_name}_vs_{corpus_2_name}_{comparison_function.__name__}_comparison_tmp.csv'
    df.to_csv(csv_filename, index=False)

    # Return the size of the intersection and some metadata
    intersection_size = df[df[f'In {corpus_2_name}']].shape[0]
    return intersection_size


def get_raw_text(doc):
    full_text = get_full_text(doc)
    raw_text = ' '.join([f'{locutor} : {speech}' for (locutor, speech) in full_text])
    return raw_text


def compare_texts(text1, text2):
    d = distance(text1, text2, score_cutoff=0.1 * min(len(text1), len(text2)))
    s1, s2 = len(text1), len(text2)
    alignment = (s1 + s2 - d) / 2
    res = alignment / max(s1, s2)
    return res


def preprocess_title(title):
    title = normalize_title(title)
    return title


def preprocess_title_and_authors(title, author):
    title = normalize_title(title)
    # author = normalize_author(author)
    return title, author


def title_levensthein(title_1, title_2):
    if min(len(title_1), len(title_2)) >= 7 and (title_1 in title_2 or title_2 in title_1):
        return 1
    d = distance(title_1, title_2)
    return 1 - d / (max(len(title_1), len(title_2)))


def levensthein_author_and_title(couple1, couple2):
    title1, author1 = couple1
    title2, author2 = couple2
    d_title = title_levensthein(title1, title2)
    d_author = title_levensthein(author1, author2)
    return (0.3 * d_author + 0.7 * d_title)


def wikidata_merge(corpus1, corpus2):
    pass


def handle_authors_strings(s):
    print(s)
    name_or_surnames = s.split(" ")
    first_names = []
    surnames = []
    current_name = ""
    in_first_names = True
    total_authors = 0
    for nos in name_or_surnames:
        if nos.isupper() and in_first_names:
            in_first_names = False
            first_names.append(current_name.strip())
            total_authors = len(first_names)
            current_name = ""
        if in_first_names:
            if nos == "-":
                first_names.append(current_name.strip())
                current_name = ""
            else:
                current_name += f'{nos} '
        else:
            if nos == "DE":
                surnames.append(current_name)
                total_authors -= 1
                current_name = "DE "
            elif nos == "D’":
                surnames.append(current_name)
                total_authors -= 1
                current_name = "D’"
            else:
                current_name += f'{nos}'
                if total_authors > 0:
                    surnames.append(current_name)
                    current_name = ""
                    total_authors -= 1
                else:
                    current_name += " "
    if not in_first_names:
        surnames.append(current_name.strip())
    return first_names, surnames


def normalize_all_author_strings(lst_authors_file):
    lst_authors = open(lst_authors_file, 'r', encoding='utf8').read().splitlines()
    res = []
    for author in lst_authors:
        res.append(handle_authors_strings(author))
    print(res)


### On prend deux csv, on les mets en df
# Dans chacun, il faut regarder la colonne wikidata author id et comparer
# On fait un match exact dessus
# On filtre les lignes en ne gardant que celle ou le titre ressemble assez


############################################# Comparison with wikidata IDs
# First,
# Pre-processing : get all 4-grams of all plays

# Open each xml
# Get the raw txt (play parsing)
folder = os.path.abspath(os.path.dirname(sys.argv[0]))
corpus = 'CorpusDracor - new'
outputDir = 'Output'
TDCorpusFolder = os.path.join(folder, 'Corpus', 'CorpusTD')
BDCorpusFolder = os.path.join(folder, 'Corpus', 'CorpusBibDramatique')
DracorCorpusFolder = os.path.join(folder, 'Corpus', 'CorpusDracor - new')


def save_raw_texts(corpus_name, save_name=None):
    if save_name is None:
        save_name = corpus_name
    output_dir_name = os.path.join('Data', f'Textes bruts {corpus_name}')
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    for file_name in os.listdir(corpus_name):
        file = os.path.join(corpus_name, file_name)
        play = get_play_from_file(file)
        txt = get_raw_text(play)
        # title = get_title(play)
        print(file_name)
        raw_text_file_name = os.path.join(output_dir_name, f'{file_name}_raw.txt')
        raw_text_file = open(raw_text_file_name, 'w', encoding='utf8')
        raw_text_file.write(txt)


# Represent it as a dictionary which keys are 4-grams and values are number of occurences
# How to compare 2 such dictionaries : size of intersection over size of union
# That is to say : for each key, sum of the min of each values (use defaultdict 0)

def get_n_grams(txt, n):
    """Given a text, returns the dictionnary of the n-grams"""
    dict = defaultdict(int)
    words = txt.split(" ")
    for i in range(len(words) - 3):
        dict[' '.join(words[i: i + n])] += 1
    return dict


def compare_ngrams_dict(dict1, dict2):
    """Compare two ngrams dict by computing the size of the intersection over the size of the union"""
    intersection_size = 0
    total_size_1 = 0
    for key in dict1:
        intersection_size += min(dict1[key], dict2[key])
        total_size_1 += dict1[key]
    total_size_2 = sum([dict2[key] for key in dict2])
    return 2 * intersection_size / (total_size_1 + total_size_2)


def precompute_n_grams(corpus, n, corpus_name):
    output_dir_name = os.path.join('Data', f'{n}-gramme {corpus_name}')
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    for file in os.listdir(corpus):
        txt = open(os.path.join(corpus, file), 'r', encoding='utf8').read().strip()
        n_gram_dict = get_n_grams(txt, n)
        file_name = re.sub('\.xml_raw\.txt', '.pkl', file)
        print(file_name)
        save_file_ngram = open(os.path.join(output_dir_name, file_name), 'wb')
        pickle.dump(n_gram_dict, save_file_ngram)


def merge_df(ref_df, df_to_merge, corpus_string):
    rows_to_add = [ref_df]
    for index, row_to_merge in df_to_merge.iterrows():
        # Get the wikidata ID of the author
        wikidata_author_id = row_to_merge['wikidataId Author']
        title = row_to_merge['Title']
        print(title)
        # Getting the 4 grams dictionary
        raw_name = f"{row_to_merge['Raw Name']}.pkl"  # TODO : ADD THOSE COLUMNS FOR TC
        n_grams_file = os.path.join('Data', f'4-gramme {corpus_string}', raw_name)
        n_grams_dict_tomerge = pickle.load(open(n_grams_file, 'rb'))

        # If there is a wikidata ID, filter the reference corpus for only this author.
        if wikidata_author_id:
            filtered_df = ref_df[ref_df['wikidataId Author'] == wikidata_author_id]
        else:
            filtered_df = ref_df
        # What happens if there is nothing in the filtered reference dataframe ?
        # For now I check if there is a played called exactly the same in the reference, in case I missed an author
        # A more exhaustive way to do it would be to do the 4-grams comparison against ALL plays from the reference
        # But that would be very slow. If I have time, I could try it.
        if len(filtered_df) == 0:
            filtered_df = ref_df[ref_df['Title'] == title]
        # Keeping in memory the most ressembling row from the reference dataframe
        best_match, best_d = None, 0
        for index2, row_ref in filtered_df.iterrows():
            if title.title() == row_ref['Title'].title():  # Found exact match ?
                # TODO: Incorporate 4-grams check
                best_match, best_d, best_index = row_ref, 1, index2
                print(f'Found exact match {title} with similarity {best_d}')
                break
            else:
                # Compare n-grams

                # First we need to get the 4-grams dictionnary :
                raw_name_TD = f"{row_ref['Raw Name']}.pkl"

                # THIS IS SKETCHY : IT TRIES TO FIND THE GOOD DIRECTORY TO OPEN THE 4 GRAMS FILE, but it depends on the order
                # TODO : fix it with try/excepts ?
                if not (pd.isna(row_ref['in_TD'])):
                    folder_ngram = 'TD'
                elif not (pd.isna(row_ref['in_Dracor'])):
                    folder_ngram = 'Dracor'
                elif not (pd.isna(row_ref['in_BD'])):
                    folder_ngram = 'BD'
                n_grams_file = os.path.join('Data', f'4-gramme {folder_ngram}', raw_name_TD)
                n_grams_dict_TD = pickle.load(open(n_grams_file, 'rb'))

                # Now we do the comparison
                d = compare_ngrams_dict(n_grams_dict_tomerge, n_grams_dict_TD)
                # Experimentally, 4-grams similarity is <0.1 for two random plays and around 0.4 for similar plays.
                # The cutoff is put at 0.3 to play it safe, I haven't witnessed false positives yet.
                if d > max(best_d, 0.3):
                    best_match, best_d, best_index = row_ref, d, index2

        if best_match is not None:
            # If we found a match, we update the reference sheet with the id of the play.
            ref_df.at[best_index, f'in_{corpus_string}'] = row_to_merge[f'{corpus_string} id']
            # Trying to bring in as much information as possible
            for col_name in ref_df.columns:
                if col_name in df_to_merge.columns and str(best_match[col_name]).strip() in ['[vide]', '', 'NA',
                                                                                             'N/A']:  # Yeah...
                    ref_df.at[best_index, col_name] = row_to_merge[col_name]
            print(f'Best match found with similarity {best_d} : {best_match["Title"]} ')
        else:
            # No match found : we create a new row to append to our df
            row_to_merge[f'in_{corpus_string}'] = row_to_merge[f'{corpus_string} id']
            row_to_merge = row_to_merge[ref_df.columns.intersection(row_to_merge.index)]
            rows_to_add.append(row_to_merge.to_frame().T)
    ref_df = pd.concat(rows_to_add, ignore_index=True)  # Adding all new plays we found
    ref_df.to_csv(os.path.join('Outputs', f'Merge TD {corpus_string}  V4.csv'), index=False)
    print(f'New rows : {len(rows_to_add) - 1}')


# then by similarity with 4-grams. Is similarity is above some threhsold (50% ?) keep it
# If not found, add it as a new entry in the reference corpus
# If there is no wikidata ID, try to find by title

def join_dataframes(bd_df, merged_df):
    """Perform an SQL left join on dataframes.

    Here the example is on key BD id and we want to keep Years """
    bd_df = bd_df.rename(columns={'BD id': 'in_BD', 'Date': 'Normalized Year'})
    bd_subset = bd_df[['in_BD', 'Normalized Year']]
    # Merge with suffixes to keep both temporarily
    merged_df = pd.merge(merged_df, bd_subset, on='in_BD', suffixes=('_dracor', '_bd'), how='left')

    # Create a single column by filling NaNs from one with the other
    merged_df['Normalized Year'] = merged_df['Normalized Year_dracor'].combine_first(merged_df['Normalized Year_bd'])

    # Drop the two original columns
    merged_df = merged_df.drop(columns=['Normalized Year_dracor', 'Normalized Year_bd'])

    merged_df['Normalized Year'] = merged_df['Normalized Year'].astype('Int64')
    merged_df['in_BD'] = merged_df['in_BD'].astype('Int64')
    print(merged_df.columns)
    merged_df.to_csv(os.path.join(DataFolder, 'Supersheet_merged.csv'))

def count_year_occurrences(dataframe):
    # Assuming "Genre" is the name of the column containing genre strings
    genre_counts = {}
    genres = dataframe['Normalized Year']

    for genre in genres:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1

    return genre_counts

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DataFolder = os.path.join('Data', 'Corpus comparison')
    # reference_dracor = os.path.join(DataFolder, 'Metadata Dracor.csv')
    # reference_bd = os.path.join(DataFolder, 'Metadata Bibdramatique.csv')
    reference_supersheet = os.path.join(DataFolder, 'Supersheet.csv')

    supersheet_df = pd.read_csv(reference_supersheet)
    # dracor_df = pd.read_csv(reference_dracor)
    # bd_df = pd.read_csv(reference_bd)

    dict_years = count_year_occurrences(supersheet_df)
    print(min(dict_years.keys()))
    print(max(dict_years.keys()))
    siecles_dict = dict()
    pas = 100
    for siecle in range(1100, 2018, pas):
        for year in dict_years:
            if siecle <= year <= siecle + pas -1:
                siecles_dict[f'{siecle} - {siecle + pas - 1}'] = siecles_dict.get(f'{siecle} - {siecle + pas - 1}',0) + dict_years[year]
    print(siecles_dict)
    plt.bar(siecles_dict.keys(), siecles_dict.values())
    plt.show()





    # #### CODE TO MERGE BIBDRAMATIQUE, TD, AND DRACOR :
    #
    # # Metadata sheets
    # DataFolder = os.path.join('Data', 'Corpus comparison')
    # reference_dracor = os.path.join(DataFolder, 'Metadata Dracor.csv')
    # reference_bd = os.path.join(DataFolder, 'Metadata Bibdramatique.csv')
    # reference_td_merged = os.path.join('Outputs', 'Merge TD Dracor  V4.csv')
    # #reference_td_merged = os.path.join('Outputs', 'Merge TD BD  V3.csv')
    #
    # # Putting stuff in dataframes
    # ref_df = pd.read_csv(reference_td_merged)
    # dracor_df = pd.read_csv(reference_dracor)
    # bd_df = pd.read_csv(reference_bd)
    #
    # # Normalizing
    # dracor_df['Raw Name'] = dracor_df['Raw Name'].apply(lambda x: re.sub(r'\s|\.xml', '', x))
    # bd_df['Raw Name'] = bd_df['Raw Name'].apply(lambda x: re.sub(r'\s|\.xml', '', x))
    # ref_df['Raw Name'] = ref_df['Raw Name'].apply(lambda x: re.sub(r'\s', '', x))
    #
    # # Declaring variables
    # df_ref, corpus_ref_string = ref_df, "TD"
    # df_to_merge, corpus_to_merge_string = bd_df, "BD"
    #
    # df_ref[f'in_{corpus_to_merge_string}'] = None
    # merge_df(df_ref, df_to_merge, corpus_to_merge_string)
