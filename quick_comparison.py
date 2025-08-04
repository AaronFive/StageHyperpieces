import re
import csv
import os
from polyleven import levenshtein
from collections import defaultdict
from heuristics import greedy_heuristic,successions_heuristic,simple_frequence_heuristic
from basic_utilities import flatten_list, flatten_list_of_list
import fpt_alphabet_size
import sat_instance
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

output_folder = 'Quick comparisons/Comparison results'
input_folder = 'Quick comparisons'


def both_compare_pieces(f1, f2, pair_name, timeout_sat=800, timeout_bruteforce=60, split_by_act=True):
    """ Given twofile names f1 and f2 of plays performs a parameterized matching comparison of both,
    and save the output in a csv file, which name is given by pair_name
     Args:
         f1 (str): Path to first play
         f2 (str): Path to second play
         timeout_sat (int) : timeout for the sat approach (default 800). Set to skip it
         timeout_bruteforce (int) : timeout for the bruteforce approach
         """
    # SAT comparison
    final_output_dir = os.path.join('Quick comparisons', 'Comparison results')
    logs_files = open(os.path.join(final_output_dir, f'{pair_name}_logs'), 'w')
    output_csv = open(os.path.join(final_output_dir, f'{pair_name}.csv'), 'w', encoding='utf8')
    fieldnames = ['Pair name', 'Act Number', 'Distance', 'Normalized_Distance', 'Input_1', 'Input_2', 'Renaming',
                  'Input 1 renamed',
                  'Input 1 length', 'Input 2 length', 'Computing time', 'Letters_play1', 'Letters_play2']
    gwriter = csv.DictWriter(output_csv, fieldnames=fieldnames)
    gwriter.writeheader()
    if timeout_sat > 0:
        sat_instance.compare_pieces(f1, f2, pair_name, logs_files, gwriter, final_output_dir, timeout_sat)
    if timeout_bruteforce > 0:
        fpt_alphabet_size.compare_pieces(f1, f2, pair_name, gwriter, timeout_bruteforce, split_by_act)


# We compare plays from a csv
# First we have to get the pairs : the id of the plays, and the id of its source (from one file)
# We have to get, for each, the list of its speakers
# Then , for each couple of plays, get their list of speakers, and compare them (using which function ?)
# Return this in a csv file : Play 1/ Play 2/ Distance / Statistics about it
sources_couples_file = 'Quick comparisons/Imitation, création au théâtre - Couples.csv'
sources_plays_file = 'Quick comparisons/Imitation, création au théâtre - Pieces.csv'

sources_couples = open(sources_couples_file, 'r')
d = csv.DictReader(sources_couples)
sources = {x['id_piece_source']: x['id_piece_inspiree'] for x in d}


def convert_string_to_list(s):
    """Given a string of a play of the form [[[A,B,C],[B,C]], """
    res = []
    acts = s.split(']]')
    for act in acts:
        act_res = []
        scenes = act.split('["')
        scenes = [re.sub(r'\[|]]?', "", x) for x in scenes]
        scenes = [x for x in scenes if x != '']
        for x in scenes:
            persos = x.split(',')
            persos = [re.sub('"|]', '', x) for x in persos]
            persos = [x for x in persos if x not in ['', ' ']]
            act_res.append(persos)
        res.append(act_res)
    return res


# Getting characters from spanish plays
sources_plays = open(sources_plays_file, 'r', encoding='utf8')
d = csv.DictReader(sources_plays, dialect='unix')
# output = open('extraction_dracor','w')
# for play in d:
# Getting characters from Dracor
# Already executed
#     if 'DRACOR' in play['succession_personnages']:
#         if 'dracor' in play['url_xml']:
#             dracor_id = re.match('https://dracor\.org/(?:api/corpora/)?fre/(?:play/)?([^/]*)(:?/tei)?',play['url_xml'])
#             if dracor_id:
#                 file_name = dracor_id.group(1)
#                 file_path = os.path.join('Corpus','CorpusDracor',f'{file_name}.xml')
#                 file_play = open(file_path,'r')
#                 doc_play = minidom.parse(file_play)
#                 character_succession = play_parsing.get_all_acts_dialogues(doc_play,split_by_act=True)
#                 output.writelines(f"""
# {file_name}
#                 """)
#                 output.writelines(str(character_succession))
plays = {x['id']: eval(x['succession_personnages']) for x in d if x['succession_personnages'] != ''}


# Uncomment this part to get extract the text from a specific play
# corpusfolder = os.path.join('corpus10pairs','La Belle Egyptienne') # PATH OF FOLDER
# output = open('successionsallebray.txt', 'w') # NAME OF OUTPUT
# corpus = os.getcwd()
# f = os.path.join(corpusfolder,'fre001316-sallebray-belle-egyptienne.tei.xml') #NAME OF FILE
# doc = minidom.parse(open(f, 'r'))
# title = play_parsing.get_title(doc)
# succession = play_parsing.get_all_acts_dialogues(doc, split_by_act=True)
# succession_text = str(succession).replace('\'', '"')
# output.writelines(f"""{title}:
#
#         {succession_text}
#
#
#         """)






# Représentation alignement
# Couper les n moins fréquents OK
# Correspondances personnages OK
# Normaliser par la taille : (d- ||v|-|u||)/min(|u|,|v|)

# Adapter le bruteforce aux ensembles

# This is the csv header for all the information we want to log for tests on parameterized matching computation
fieldnames = ["Pair name", "Act Number", "Normalized_Distance", "Renaming", "Distance", "Input_1", "Input_2",
              "Input 1 length",
              "Input 2 length", "Input 1 renamed", "Letters_play1", "Letters_play2", "Computing time"]


def run_on_corpus(sources, outputname, fieldnames, timeout=600):
    output_csv = open(os.path.join(output_folder, f'{outputname}.csv'), 'w', newline='', encoding='utf8')
    gwriter = csv.DictWriter(output_csv, fieldnames=fieldnames)
    gwriter.writeheader()
    count_pairs = 0
    for i, play in enumerate(sources):
        print(play)
        if play in plays and sources[play] in plays:
            count_pairs += 1
            play1_name, play2_name = play, sources[play]
            acts_1, acts_2 = plays[play1_name], plays[play2_name]
            acts_1 = flatten_list(acts_1)
            acts_2 = flatten_list(acts_2)
            acts_1 = [fpt_alphabet_size.keep_most_frequents(acts_1, 8)]
            acts_2 = [fpt_alphabet_size.keep_most_frequents(acts_2, 8)]
            print(acts_1)
            print(acts_2)
            fpt_alphabet_size.compare_pieces_content(acts_1, acts_2, play1_name + play2_name, gwriter, timeout,
                                                     fieldnames)


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




if __name__ == "__main__":
    # heuristics = [successions_heuristic, simple_frequence_heuristic] #greedy_heuristic,
    # compare_heuristics(sources, heuristics, 'heuristic_comparison_test.csv')
    f1 = "Corpus\\Corpus Dramacode\\ouville_espritfolet.xml"
    f2 = 'cal000025-la-dama-duende.tei.xml'
    both_compare_pieces(f1,f2, 'dama_full_text', 0, 600, False)
