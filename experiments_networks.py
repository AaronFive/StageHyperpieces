import csv
import json
import os
import pickle
import re
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import unidecode as unidecode
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer

import play_parsing
from corpus_intersection import compare_ngrams_dict

data_dir = 'Data\\Network comparison'
ngrams_dir = 'Data\\4-gramme Dracor'
french_stopwords_file = f'{data_dir}\\StoplistFrench.txt'
dracor_metadata_path = 'Data\\fredracor-metadata.csv'
dracor_corpus_path = 'Corpus\\CorpusDracor - new'
close_cast_csv_path = f'{data_dir}\\plays_with_cast_intersection_atleast6_ngramsimilarity.csv'
dracor_metadata = pd.read_csv(dracor_metadata_path)
with open(french_stopwords_file, 'r') as stopwords_file:
    french_stopwords = stopwords_file.read().splitlines()
play_file_1 = os.path.join(data_dir, 'DESJARDINS_FAVORI.xml')
play_file_2 = os.path.join(data_dir, '108_JODELLE_DIDON.xml')

name_1 = re.sub(r'.xml', r'.html', play_file_1)
# test_file = os.path.join(data_dir, 'fre001019-moliere-misanthrope.spoken-text-by-character.json')

# Load the CSV file
# file_path = os.path.join(data_dir, 'moliere-tartuffe.xml')
# file_path = os.path.join(data_dir, 'fre001019-moliere-misanthrope.network.csv')

# df = pd.read_csv(file_path, sep=',')  # Use sep=',' if it's comma-separated
# exclude = ['#l-exempt', '#loyal']
# # exclude = ['basque', 'dubois', 'garde']
# # Filter out rows where Source or Target is in the exclude list
# df = df[~df['Source'].isin(exclude) & ~df['Target'].isin(exclude)]

play_1 = play_parsing.get_play_from_file(play_file_1)
play_2 = play_parsing.get_play_from_file(play_file_2)


def get_character_text_from_file(file):
    """Parse the dracor speaker_text files. Returns a dict which keys are character ids and values are their text

    The Dracor files are said to be json but are actually csv..."""
    characters_text = dict()
    with open(file, newline='', encoding='utf8') as f:
        speakers = csv.DictReader(f)
        for row in speakers:
            characters_text[row['ID']] = row['Text'].replace('\n', ' ')
    return characters_text


def get_speakers_in_scene(scene):
    """Given a scene, returns a set of speakers (character identifiers) who speak in the scene."""
    speaker_list = scene.getElementsByTagName('sp')
    speakers = set()
    for speaker_node in speaker_list:
        locutor_id = speaker_node.getAttribute('who')
        if locutor_id:
            speakers.add(locutor_id)
    return speakers


def get_scene_texts(scene):
    """Given a scene, returns a dictionary mapping speakers to their spoken text."""
    translation_dict = {'\x81': None, '\x84': None, '\x85': '...', '\x91': '\'', '\x93': '\"', '\x95': '•',
                        '\x96': '-', '\uff0c': ',', '\u0144': 'ń', '\u0173': 'ų',
                        '\u2015': '―', '\xa0': ' '}

    speaker_list = scene.getElementsByTagName('sp')
    scene_text = {}

    for speaker_node in speaker_list:
        sentences = []
        sentences.extend(speaker_node.getElementsByTagName('l'))
        sentences.extend(speaker_node.getElementsByTagName('s'))
        # sentences.extend(speaker_node.getElementsByTagName('stage')) Uncomment for stage directions

        text = []
        for sentence in sentences:
            for child in sentence.childNodes:
                if child.nodeValue is not None:
                    text.append(child.nodeValue.strip())
                    break

        full_text = ' '.join(text).translate(str.maketrans(translation_dict))
        locutor_id = speaker_node.getAttribute('who')

        if locutor_id:
            if locutor_id in scene_text:
                scene_text[locutor_id] += ' ' + full_text
            else:
                scene_text[locutor_id] = full_text

    return scene_text


def get_all_scenes(play):
    """Given the whole play, returns a list of scene nodes (<div> or <div2> with @type="scene")."""
    scenes = []
    for tag_name in ['div', 'div2']:
        divs = play.getElementsByTagName(tag_name)
        for div in divs:
            if div.getAttribute('type') == 'scene':
                scenes.append(div)
    return scenes


def build_pairwise_texts(play, top_n_characters=None, only_keep=None):
    """Given a play, returns:
    - dict {(A, B): text} where A spoke while B was on stage
    - dict {A: full text spoken by A}

    Characters considered can be filtered.
    If top_n_characters is specified, only the n characters with the most occurrences are considered.
    If only_keep is specified instead, only the characters in this list are kept.
    """
    pairwise_texts = {}
    total_texts = {}
    scenes = get_all_scenes(play)

    # First pass: count scene occurrences
    scene_counter = Counter()
    for scene in scenes:
        speakers_in_scene = get_speakers_in_scene(scene)
        scene_counter.update(speakers_in_scene)
    # print({char for char in scene_counter})
    if top_n_characters is not None:
        # Select only the top N characters
        most_common_chars = set(char for char, _ in scene_counter.most_common(top_n_characters))
    elif only_keep is not None:
        most_common_chars = set([char for char in scene_counter if normalize_charname(char) in only_keep])
    else:
        most_common_chars = set(scene_counter.keys())
    # print(len(most_common_chars))
    # print(most_common_chars)
    if not most_common_chars:
        raise ValueError('No most common char')
    for scene in scenes:
        speakers_in_scene = get_speakers_in_scene(scene)
        # Filter speakers to only selected characters
        filtered_speakers = speakers_in_scene.intersection(most_common_chars)
        scene_texts = get_scene_texts(scene)

        for speaker_A, text_A in scene_texts.items():
            if speaker_A not in most_common_chars:
                continue  # skip unwanted characters
            # Update total text spoken by A
            if speaker_A in total_texts:
                total_texts[speaker_A] += ' ' + text_A
            else:
                total_texts[speaker_A] = text_A

            # Update text spoken by A when B is present
            for speaker_B in filtered_speakers:
                if speaker_A != speaker_B:
                    pair_key = (speaker_A, speaker_B)
                    if pair_key in pairwise_texts:
                        pairwise_texts[pair_key] += ' ' + text_A
                    else:
                        pairwise_texts[pair_key] = text_A

    return pairwise_texts, total_texts


def compute_tfidf(characters_text, n, printing=False):
    """Given a list of texts said by/to different characters, returns the n words with the highest tf-idf for each

    Args:
        characters_text(dict): A dictionary which keys are character names and values are associated text
        n(int): Number of words to keep for each character
        printing(bool): If True, results are pretty printed
    Returns:
        dict"""
    # Get character names and their texts
    character_names = list(characters_text.keys())
    corpus = list(characters_text.values())
    # Create the vectorizer and compute TF-ID
    vectorizer = TfidfVectorizer(stop_words=french_stopwords, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Get feature (word) names
    feature_names = vectorizer.get_feature_names_out()

    # Store results in a dictionary
    top_words_per_character = {}

    for i, character in enumerate(character_names):
        # Get the row corresponding to the character
        row = tfidf_matrix[i].toarray().flatten()

        # Get indices of the top 10 TF-IDF scores
        top_indices = row.argsort()[-n:][::-1]

        # Get the corresponding words and their scores
        top_words = [(feature_names[j], row[j]) for j in top_indices if row[j] > 0]

        top_words_per_character[character] = top_words

    if printing:
        # Print results
        for character, words in top_words_per_character.items():
            print(f"\n{character}:")
            for word, score in words:
                print(f"  {word}: {score:.4f}")
    return top_words_per_character


def get_top_words_per_pair(pairwise_text, character_text, ntfidf=10):
    """Given the list of lines said between each pair of characters, computes the most representative words between each pair using tf-idf

    The tf-idf computation is done in the following way : for each character A, the corpus consists of a document
    for each other character B. In every one of these document is what A is saying while B is on stage. """
    top_words_per_pair = dict()
    characters = character_text.keys()
    for char in characters:
        char_text = {y: pairwise_text[(char, y)] for y in characters if y != char and (char, y) in pairwise_text}
        tfidf_char = compute_tfidf(char_text, ntfidf)
        for y in tfidf_char:
            top_words_per_pair[(char, y)] = tfidf_char[y]
    return top_words_per_pair


def create_edges_network_df(pairwise_text, min_weight=0.2):
    """Create a dataframe containing all the information for the creation of a network

    Filters the edges to keep only those with normalized minimum weight >=0.2"""
    # Prepare data for DataFrame
    data = []
    if pairwise_text:
        max_length = max(len(text) for text in pairwise_text.values())
    else:
        max_length = 1
        print("Warning : no text")

    for (char_A, char_B), text in pairwise_text.items():
        weight = len(text) / max_length
        if weight > min_weight:  # normalize
            data.append({
                'Source': char_A,
                'Type': 'Directed',
                'Target': char_B,
                'Weight': weight
            })

    # Create the DataFrame
    links_df = pd.DataFrame(data, columns=['Source', 'Type', 'Target', 'Weight'])
    return links_df


def make_graph(df, top_words_per_character, top_words_per_pair):
    """Creates a directed graph, tagged by the top words for each character and each character pair

    TODO: Set properly the parameters for the visualization"""
    # Create a graph
    G = nx.DiGraph()  # Because it's directed

    # Add weighted edges to the graph
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    # Create Pyvis network
    net = Network(height='750px', width='100%', directed=True, bgcolor='#222222', font_color='white')
    net.from_nx(G)

    # Improve layout and interactivity
    for node in net.nodes:
        node['title'] = node['id']
        node['label'] = node['id']
        node['size'] = 20 + G.degree(node['id']) * 2  # Size based on degree

        # Add the top words as a tooltip for each node
        if node['id'] in top_words_per_character:
            # Create a string of top words and join them with commas
            top_words = '\n'.join([str(x) for x in top_words_per_character[node['id']]])
            node['title'] += f"Top Words: \n {top_words}"

    for edge in net.edges:
        edge['value'] = edge['width']
        edge['smooth'] = {
            'enabled': True,
            'type': 'curvedCW'
        }
        source, target = edge['from'], edge['to']
        top_words = '\n'.join([str(x) for x in top_words_per_pair[(source, target)]])
        edge['title'] = f"Top Words for {source} to  {target} : \n {top_words} "

    # Optional: Use physics for dynamic layout
    net.show_buttons(filter_=['physics'])
    #     net.set_options("""
    # const options = {
    #   "physics": {
    #     "barnesHut": {
    #       "springConstant": 0
    #     },
    #     "minVelocity": 0.75
    #   }
    # }""")

    return net


def normalize_charname(text):
    """
    Normalize text by:
    - Lowercasing
    - Removing accents
    - Stripping punctuation
    - Normalizing whitespace
    """
    text = unidecode.unidecode(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'#le-', '#', text)
    text = re.sub(r'#la-', '#', text)
    text = re.sub(r'#les-', '#', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_renaming(character_text_1, character_text_2):
    """Computes a renaming between two lists of characters and returns it

    The renaming is done by first matching the characters that have the same normalized name
    Then mapping remaining characters by volume of text said.

    Both plays are supposed to have the same number of characters"""
    characters_1, characters_2 = character_text_1.keys(), character_text_2.keys()
    characters_1_normalized, characters_2_normalized = {c: normalize_charname(c) for c in characters_1}, {
        normalize_charname(c): c for c in
        characters_2}
    renaming = dict()
    # For character that are in both lists, we map them together
    for x in characters_1:
        normalized_x = characters_1_normalized[x]
        if normalized_x in characters_2_normalized:
            renaming[x] = characters_2_normalized[normalized_x]
    # For the remaining characters, we match them by descending volume of text.
    # We assume there are exactly the same number of characters in each text.
    remaining_1 = [x for x in characters_1 if x not in renaming]
    remaining_2 = [characters_2_normalized[x] for x in characters_2_normalized if x not in renaming]

    # Compute total text lengths for remaining characters
    lengths_1 = {x: len(character_text_1.get(x, '')) for x in remaining_1}
    lengths_2 = {x: len(character_text_2.get(x, '')) for x in remaining_2}

    # Sort both by text length (descending)
    sorted_1 = sorted(lengths_1.items(), key=lambda item: item[1], reverse=True)
    sorted_2 = sorted(lengths_2.items(), key=lambda item: item[1], reverse=True)

    # Pair them in order
    for (x1, _), (x2, _) in zip(sorted_1, sorted_2):
        renaming[x1] = x2

    return renaming


def compare_dict(dict1, dict2):
    """
    Returns the sum of a dictionary where each key is from the union of dict1 and dict2,
    and the value is dict1[key] - dict2[key], treating missing keys as 0.
    """
    all_keys = set(dict1.keys()) | set(dict2.keys())
    diff = {
        key: abs(dict1.get(key, 0) - dict2.get(key, 0))
        for key in all_keys
    }
    return sum(diff.values()) / len(all_keys)


def compute_dict_intersection(dict1, dict2):
    """Returns the size of the intersection of keys of both dict, normalized between 0 and 1"""
    common_keys = set(dict1.keys()) & set(dict2.keys())
    total = len(dict1)
    return 1 - len(common_keys) / total


def compute_list_intersection(lst1, lst2):
    # TODO : normalize
    intersection = set([word for (word, weight) in lst1]) & set([word for (word, weight) in lst2])
    size = len(intersection)
    return size


def make_character_play_dict(csvFolder):
    """Computes the occurrences of each characters in each play, and keeps the display name"""
    character_play_dict = dict()
    title_play_dict = dict()
    for csv_file in os.listdir(csvFolder):
        print(csv_file)
        # Match play identifier (e.g., 'fre000123') from the filename
        match = re.match(r'(fre\d+)_(.*)\.csv', csv_file)
        play_id, play_name = match.group(1), match.group(2)
        title_play_dict[play_id] = play_name
        csv_path = os.path.join(csvFolder, csv_file)
        # Load the CSV file into a DataFrame
        try:
            characters_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue  # Skip if file is completely empty or invalid

        if characters_df.empty:
            print(f'Empty play {play_id}')
            continue  # Skip if the DataFrame is empty (no rows)
        for _, row in characters_df.iterrows():
            char_id = normalize_charname(row['id'])
            display_name = row['name']
            if char_id not in character_play_dict:
                character_play_dict[char_id] = []
            character_play_dict[char_id].append((play_id, display_name))

    return character_play_dict, title_play_dict


def get_title_from_id(dracor_id, title_play_dict):
    return title_play_dict[dracor_id]


def compare_with_ngrams(playname1, playname2):
    ngrams_1 = pickle.load(open(os.path.join(ngrams_dir, f'{playname1}.pkl'), 'rb'))
    ngrams_2 = pickle.load(open(os.path.join(ngrams_dir, f'{playname2}.pkl'), 'rb'))
    similarity = compare_ngrams_dict(ngrams_1, ngrams_2)
    return similarity


def find_plays_with_common_chars(charact_dict, csvFolder, title_play_dict, cutoff=6):
    """Generate a csv tracking the list of plays that have at least cutoff common characters """
    seen = set()
    result = []
    for csv_file in os.listdir(csvFolder):
        print(csv_file)
        match = re.match(r'(fre\d+)_(.*)\.csv', csv_file)
        play_id, play_name = match.group(1), match.group(2)
        csv_path = os.path.join(csvFolder, csv_file)
        # Load the CSV file into a DataFrame
        try:
            characters_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue  # Skip if file is completely empty or invalid
        common_plays = dict()

        for _, row in characters_df.iterrows():
            char_id = normalize_charname(row['id'])
            for (play, name) in charact_dict[char_id]:
                if play != play_id:
                    if play not in common_plays:
                        common_plays[play] = [char_id]
                    else:
                        common_plays[play].append(char_id)
        # Filtering to keep only most common
        common_plays = {x: common_plays[x] for x in common_plays if len(common_plays[x]) >= cutoff}
        if common_plays:
            for x in common_plays:
                detected_pair = tuple(sorted([play_id, x]))
                if detected_pair not in seen:
                    seen.add(detected_pair)
                    name_similar = get_title_from_id(x, title_play_dict)
                    n_gram_similarity = compare_with_ngrams(play_name, name_similar)
                    result.append({'id': play_id, 'Name': play_name, 'id_similar': x,
                                   'Name_similar': name_similar,
                                   'Intersection_size': len(common_plays[x]),
                                   'Intersection': json.dumps(common_plays[x]), 'ngram Similarity': n_gram_similarity,
                                   'Similar': n_gram_similarity > 0.3})
    result_list = pd.DataFrame(result)
    result_list.to_csv(f'{data_dir}\\plays_with_cast_intersection_atleast{cutoff}_ngramsimilarity.csv', index=False)
    print(len(seen))


def make_graph_matching(words_per_character1, words_per_character2):
    """Finds the maximum matching in the graph where vertices are characters and edges weight is given by intersection of tf-idf words"""
    # Graph construction
    G = nx.Graph()
    nodes_1, nodes_2 = [char_id + '1' for char_id in words_per_character1], [char_id + '2' for char_id in
                                                                             words_per_character2]
    G.add_nodes_from(nodes_1, bipartite=0)
    G.add_nodes_from(nodes_2, bipartite=1)
    edges = [(char1, char2, intersection_size)
             for char1 in nodes_1 for char2 in nodes_2 if (
                 intersection_size := compute_list_intersection(words_per_character1[char1[:-1]],
                                                                words_per_character2[char2[:-1]])) > 0]
    G.add_weighted_edges_from(edges)
    matching = nx.max_weight_matching(G)
    return matching


def test_tfidf_on_close_casts(close_cast_csv, corpus_path, n_tfidf):
    close_casts = pd.read_csv(close_cast_csv)
    matchings = []
    matching_scores = []

    for _, row in close_casts.iterrows():
        play_1_file, play_2_file = f"{row['Name']}.xml", f"{row['Name_similar']}.xml"
        print(play_1_file, play_2_file)
        intersection = json.loads(row['Intersection'])

        play_1_path = os.path.join(corpus_path, play_1_file)
        play_2_path = os.path.join(corpus_path, play_2_file)

        play_1 = play_parsing.get_play_from_file(play_1_path)
        play_2 = play_parsing.get_play_from_file(play_2_path)

        try:
            pairwise_text_1, character_text_1 = build_pairwise_texts(play_1, only_keep=intersection)
            pairwise_text_2, character_text_2 = build_pairwise_texts(play_2, only_keep=intersection)
        except ValueError:
            print(f'Problem with the couple {play_1_file} and {play_2_file}')
            matchings.append(None)
            matching_scores.append(None)
            continue

        top_words_per_char1 = compute_tfidf(character_text_1, n_tfidf)
        top_words_per_char2 = compute_tfidf(character_text_2, n_tfidf)

        # Perform max weight matching
        matching = make_graph_matching(top_words_per_char1, top_words_per_char2)

        # Normalize and sort matching pairs
        normalized_matching = {(normalize_charname(x), normalize_charname(y)) for (x, y) in matching}
        normalized_matching = {tuple(sorted(c, key=lambda x: x[-1])) for c in normalized_matching}

        # Compute identity matching
        normalized_chars = [normalize_charname(c) for c in intersection]
        identity_matching = {(f'{c}1', f'{c}2') for c in normalized_chars}

        # Compute matching score as ratio of matches that are identity matches
        correct_matches = normalized_matching.intersection(identity_matching)
        score = len(correct_matches) / max(1, len(identity_matching))  # avoid division by zero

        matchings.append(json.dumps(list(normalized_matching)))
        matching_scores.append(round(score, 3))

    close_casts['Matching'] = matchings
    close_casts['MatchingScore'] = matching_scores

    # close_casts.to_csv(f'{data_dir}\\cast_intersection_automatic_matchings.csv', index=False)

    return close_casts


def compare_networks(words_per_char_1, words_per_char_2, words_per_pair_1, words_per_pair_2, renaming, distance):
    """Compare two plays networks, once a renaming and a distance has been fixed.
    WIP"""
    distance_chars = {(x, renaming[x]): distance(words_per_char_1[x], words_per_char_2[renaming[x]]) for x in
                      words_per_char_1}
    distance_pairs = {((x, y), (renaming[x], renaming[y])): distance(words_per_pair_1[(x, y)],
                                                                     words_per_pair_2[(renaming[x], renaming[y])]) for
                      (x, y) in words_per_pair_1}
    total_distance_chars = sum(distance_chars.values())
    total_distance_pairs = sum(distance_pairs.values())
    for x in distance_chars:
        print(x, distance_chars[x])
    for x in distance_pairs:
        print(x, distance_pairs[x])


def truncate_tfidf(tfidf_dict, n):
    return {char: words[:n] for char, words in tfidf_dict.items()}


def test_tfidf_on_close_casts_over_n(close_cast_csv, corpus_path, min_n=1, max_n=50, step=1):
    close_casts = pd.read_csv(close_cast_csv)
    summary_scores = {}

    os.makedirs("tf-idf graphs", exist_ok=True)

    for _, row in close_casts.iterrows():
        play_1_file, play_2_file = f"{row['Name']}.xml", f"{row['Name_similar']}.xml"
        print(play_1_file, play_2_file)
        intersection = json.loads(row['Intersection'])
        play_1_path = os.path.join(corpus_path, play_1_file)
        play_2_path = os.path.join(corpus_path, play_2_file)

        try:
            play_1 = play_parsing.get_play_from_file(play_1_path)
            play_2 = play_parsing.get_play_from_file(play_2_path)

            pairwise_text_1, character_text_1 = build_pairwise_texts(play_1, only_keep=intersection)
            pairwise_text_2, character_text_2 = build_pairwise_texts(play_2, only_keep=intersection)
        except ValueError:
            print(f'Problem with the couple {play_1_file} and {play_2_file}')
            continue

        # Precompute max TF-IDF (n=50) once
        tfidf_1_all = compute_tfidf(character_text_1, n=max_n)
        tfidf_2_all = compute_tfidf(character_text_2, n=max_n)

        scores = []
        for n in range(min_n, max_n + 1, step):
            tfidf_1 = truncate_tfidf(tfidf_1_all, n)
            tfidf_2 = truncate_tfidf(tfidf_2_all, n)
            matching = make_graph_matching(tfidf_1, tfidf_2)
            # Normalize and sort matching pairs
            normalized_matching = {(normalize_charname(x), normalize_charname(y)) for (x, y) in matching}
            normalized_matching = {tuple(sorted(c, key=lambda x: x[-1])) for c in normalized_matching}
            # Compute identity matching
            normalized_chars = [normalize_charname(c) for c in intersection]
            identity_matching = {(f'{c}1', f'{c}2') for c in normalized_chars}

            # Compute matching score as ratio of matches that are identity matches
            correct_matches = normalized_matching.intersection(identity_matching)
            score = len(correct_matches) / max(1, len(identity_matching))  # avoid division by zero
            scores.append(score)

        summary_scores[(row['Name'], row['Name_similar'])] = scores

        # Plot for this pair
        plt.figure()
        plt.plot(range(min_n, max_n + 1, step), scores, marker='o')
        plt.title(f"TF-IDF Matching Score: {row['Name']} vs {row['Name_similar']}")
        plt.xlabel("n_tfidf")
        plt.ylabel("Matching Score")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(f"tf-idf graphs/{row['Name']}_vs_{row['Name_similar']}.png")
        plt.close()

    # Compute the average matching score for each n_tfidf value
    average_scores = []
    for n in range(min_n, max_n + 1, step):
        avg_score = sum([scores[n - 1] for scores in summary_scores.values()]) / len(summary_scores)
        average_scores.append(avg_score)

    # Plot the average score across all pairs
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_n, max_n + 1, step), average_scores, marker='o', label='Average Matching Score')
    plt.title("Average TF-IDF Matching Score across All Pairs")
    plt.xlabel("n_tfidf")
    plt.ylabel("Average Matching Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tf-idf graphs/_average_summary.png")
    plt.close()


if __name__ == "__main__":
    # Finding the plays with common characters
    # charac_play_dict, title_dict = make_character_play_dict('dracor_data')
    # find_plays_with_common_chars(charac_play_dict, 'dracor_data', title_dict)

    # Comparing close plays
    # test_tfidf_on_close_casts(close_cast_csv_path, dracor_corpus_path, 30)

    # Testing for variable values of n
    # test_tfidf_on_close_casts_over_n(close_cast_csv_path, dracor_corpus_path, min_n=1, max_n=150)

    # Get data for play 1
    pairwise_text, character_text = build_pairwise_texts(play_1, top_n_characters=6)
    top_words_per_char = compute_tfidf(character_text, 10)
    top_words_per_pair = get_top_words_per_pair(pairwise_text, character_text)
    #
    # # Get data for play 1
    # pairwise_text2, character_text2 = build_pairwise_texts(play_2, 6)
    # top_words_per_char2 = compute_tfidf(character_text2, 10)
    # top_words_per_pair2 = get_top_words_per_pair(pairwise_text2, character_text2)
    #
    # renaming = compute_renaming(character_text, character_text2)
    # print(renaming)
    #
    # compare_networks(dict(top_words_per_char), dict(top_words_per_char2), dict(top_words_per_pair), dict(top_words_per_pair2), renaming,compare_dict)
    #
    network_df = create_edges_network_df(pairwise_text)
    net = make_graph(network_df, top_words_per_char, top_words_per_pair)
    net.show(f"{name_1}", notebook=False)
