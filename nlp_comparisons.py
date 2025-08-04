import itertools
import json
import math
import os
import pickle
import random
import re
import time
from collections import defaultdict
from heapq import nlargest

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from basic_utilities import pickle_file, OUTPUT_DIR, PICKLE_DIR
from pickled_data import get_full_text_corpus
from play_parsing import get_play_from_file, get_dracor_id, get_title, get_full_text

# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords

# camembert = CamembertForMaskedLM.from_pretrained('camembert-base')
# camembert = camembert.cuda()
# tokenizer = AutoTokenizer.from_pretrained('camembert-base')


# print('Loading tokenizer')
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
print('Loading model ...')
# model = BertModel.from_pretrained('bert-base-multilingual-cased')
sentence_model = SentenceTransformer("dangvantuan/sentence-camembert-base")
print('Model loaded')


def average_embeddings(embeddings, attention_mask):
    return (attention_mask[..., None] * embeddings).mean(1)


# def similarity_score_sentences(sentence1, sentence2):
#     batch_sentences = [sentence1, sentence2]
#     tokenizer_output = tokenizer(
#         batch_sentences,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     )
#     inputs_ids, attention_masks = tokenizer_output.input_ids, tokenizer_output.attention_mask
#     with torch.no_grad():
#         model_output = camembert(inputs_ids, attention_masks, output_hidden_states=True)
#     token_embeddings = model_output.hidden_states[-1]
#     avg_sentence_representations = average_embeddings(token_embeddings, tokenizer_output.attention_mask)
#     avg_similarity_score = F.cosine_similarity(avg_sentence_representations[0], avg_sentence_representations[1], dim=-1)
#     return avg_similarity_score


# def split_into_parts(sentence, max_token_length):
#     """Splits a sentence into parts of length at most max_token_length.
#     WARNING : Can give an infinite loop on small values of max_token_length.
#     Should not happen when max_token_length > 50 """
#     # Find whitespace-based breakpoints for splitting the sentence
#     breakpoints = [match.end() for match in re.finditer(r'\s', sentence)]
#
#     parts = []
#     start_index = 0
#
#     current_index = 0
#     i = 0
#     while i < len(breakpoints):
#         if breakpoints[i] - start_index < max_token_length:
#             current_index = breakpoints[i]
#             i += 1
#         else:
#             parts.append(sentence[start_index:current_index].strip())
#             start_index = current_index
#
#     # Add the last part
#     if start_index < len(sentence):
#         parts.append(sentence[start_index:].strip())
#
#     return parts

## JADT work
def split_into_parts(s):
    delimiters = r'[\|\.\?!]'
    sentences = r'[^\|\.\?!]+'
    repliques = re.findall(f'{sentences}{delimiters}', s)
    parts = [re.sub('\s*\|\s*', ' ', part) for part in repliques if part.strip()]
    parts = [part.strip() for part in parts if part.strip()]
    return parts


line = 0


def process_long_sentence(sentence, max_token_length=512, overlap=0):
    global line
    print(line)
    line += 1
    # Break the long sentence into smaller  parts
    parts = split_into_parts(sentence)
    if not parts:
        return
    # Tokenize each part
    tokenized_parts = tokenizer(parts, return_tensors="pt", padding="max_length")
    # To visualize the tokens :
    tokens = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in tokenized_parts['input_ids']]

    with torch.no_grad():
        model_outputs = camembert(**tokenized_parts, output_hidden_states=True)
    # Extract token embeddings from the last hidden state for each part
    token_embeddings = model_outputs.hidden_states[-1]
    return token_embeddings, tokens


def similarity_tokens(t1, t2):
    return F.cosine_similarity(t1, t2, dim=-1).item()


def get_token(v):
    return tokenizer.convert_ids_to_tokens(v)


# semicolon_token_id = tokenizer( [';'], add_special_tokens = False)
# semicolon_token = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in semicolon_token_id['input_ids']][0][0]
# print(type(semicolon_token))
# print(semicolon_token[1])

# punct = string.punctuation
# punct_tokens_ids = tokenizer([x for x in punct], add_special_tokens=False)
# punct_tokens = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in punct_tokens_ids['input_ids']]
# punct_tokens = set([x for l in punct_tokens for x in l])
#
# french_stopwords = fr_stop
# stopwords_tokens_ids = tokenizer([x for x in french_stopwords], add_special_tokens=False)
# stopwords_tokens = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in stopwords_tokens_ids['input_ids']]
# stopwords_tokens = set([x for l in stopwords_tokens for x in l])
# stopwords_tokens = stopwords_tokens.union(
#     {'▁Flo', '▁MAN', '▁Manuel', '▁Carr', 'ille', 'TRI', 'OS', 'ANGE', '▁Angel', '▁Ange', '▁Ang', '▁Luc', '▁RO', '▁Li',
#      '▁JEAN', 'ISA', 'BEL', '▁Isabel', '▁Isabelle'})


def to_be_ignored(t):
    if t and t[0] == '▁':
        stripped_t = t[1:]
    else:
        stripped_t = t
    if len(stripped_t) == 1 or t == ' ' or t in [tokenizer.cls_token,
                                                 tokenizer.sep_token] or stripped_t in punct or stripped_t.lower() in french_stopwords or t in stopwords_tokens:
        return True
    return False


line = 0


def get_k_closest_tokens(s1, s2, tok_list_1, tok_list_2, k=5):
    global line
    similarities = []
    print(line)
    line += 1
    for (index_sentence1, tlist1) in enumerate(tok_list_1):
        for (index_token1, t1) in enumerate(tlist1):
            if not to_be_ignored(t1):
                for (index_sentence2, tlist2) in enumerate(tok_list_2):
                    for (index_token2, t2) in enumerate(tlist2):
                        if not to_be_ignored(t2):
                            similarity = round(
                                similarity_tokens(s1[index_sentence1][index_token1], s2[index_sentence2][index_token2]),
                                2)
                            similarities.append((similarity, t1, t2))
    return nlargest(k, similarities, key=lambda x: x[0])


def tmp_print(tensor):
    if len(tensor) >= 0:
        for sentence in tensor:
            for t in sentence:
                print(t)
                print(tokenizer.convert_ids_to_tokens(t))
    else:
        print('empty tensor')


def remove_pads(token_list_list):
    res = []
    pad_token_id = tokenizer.pad_token_id
    pad_token = tokenizer.convert_ids_to_tokens(pad_token_id)
    if type(token_list_list) is not float:
        for (i, token_list) in enumerate(token_list_list):
            # print(f'{i} : {token_list}')
            try:
                pad_start_index = token_list.index(pad_token)
            except ValueError:
                print(f'pad not found in {token_list}')
                pad_start_index = len(token_list)
            # vectors_list_list[i][:] = vectors_list_list[i][:pad_start_index]
            res.append(token_list[:pad_start_index])
    return res


def get_chunk_embedding(chunk, model):
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(chunk)).unsqueeze(0)
    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return embedding


# Using tokenized corpus

def vectorize_tokenized_play(tokenized_play):
    """ Computes embeddings for every line of an already tokenized play.
    If there are too many tokens, cut them into chunks, compute embeddings, and average them.
    Args:
        model(any): BERT model
        tokenized_play(list): List of all tokenized lines in the play. Assumed to be of the form (locutor, tokenized_line)
    Returns:
        list: The list of embeddings of all lines"""
    vectorized_full_text = []
    for (locutor, tokens) in tokenized_play:
        chunks = chunk_tokens(tokens)
        embeddings = []
        for chunk in chunks:
            embedding = get_chunk_embedding(chunk)
            embeddings.append(embedding)
        line_embedding = sum(embeddings) / len(embeddings)  # TODO : check if the average of all embeddings is correct
        vectorized_full_text.append((locutor, line_embedding))
    return vectorized_full_text


# TODO : Check if adding CLS and SEP at every extremity is good usage.
def chunk_tokens(tokens, chunk_size=512, overlap=32):
    """Divides a list of tokens into a list of chunks, with an overlap between each successive chunk.
    Adds a CLS and SEP tokens at each extremity of the chunks. Chunk size should be given without taking them into account.
    Args:
        tokens(list): the list of tokens
        chunk_size(int): size of each chunk. Typically, 512.
        overlap(int): How many tokens should overlap between two successive chunks. Should be less than chunk_size
    Returns:
        list: The list of tokens in chunks."""
    chunk_size = chunk_size - 2  # Make room for 'CLS' and 'SEP' tokens
    chunked_tokens = []
    end = overlap
    while end < len(tokens) or len(chunked_tokens) == 0:
        start = end - overlap
        end = start + chunk_size
        chunk = ['CLS'] + tokens[start:end]
        chunk.append('SEP')
        chunked_tokens.append(chunk)
    return chunked_tokens


def tokenize_corpus(corpusFolder, name='tokenized_plays_dracor'):
    """Computes the tokenization of all plays in the corpus and saves it.
    Args:
        corpusFolder(str): path of the folder containing the corpus
    Returns:
        dict: The dictionnary containing all tokenized plays. Keys are play ID and values are of the form [(locutor, tokenized_line)]"""
    tokenized_plays = dict()
    for file in os.listdir(corpusFolder):
        play = get_play_from_file(os.path.join(corpusFolder, file))
        title = get_title(play)
        print(title)
        identifier = get_dracor_id(play)
        full_text = get_full_text(play, remove_punctuation=True)
        tokenized_text = [(locutor, tokenizer.tokenize(text)) for (locutor, text) in full_text]
        tokenized_plays[identifier] = tokenized_text
        print("tokenizing done")
    pickle_file(tokenized_plays, name)
    return tokenized_plays


# Uses input from function tokenize corpus.
def make_bar_plot_tokenized_corpus(tokenized_plays_pickle_file):
    """Creates a bar plot to visualize the repartition of token-length of lines in the corpus
    Args:
        tokenized_plays_pickle_file(str): Path of the save file of tokenized plays"""
    size_of_tokens = dict()
    tokenized_plays = pickle.load(open(tokenized_plays_pickle_file, 'rb'))
    # Counting token-size of lines
    for id in tokenized_plays:
        tokenized_text = tokenized_plays[id]
        for (_, tokens) in tokenized_text:
            token_number = len(tokens)
            size_of_tokens[token_number] = size_of_tokens.get(token_number, 0) + 1
    sorted_data = dict(sorted(size_of_tokens.items()))

    # Extract keys and values
    keys = list(sorted_data.keys())

    max_token_size = max(keys)
    bins = [512 * i for i in range(int(max_token_size / 512) + 1)]

    # Compute cumulative sum of occurrences
    values = [0 for i in range(len(bins))]
    for number in sorted_data:
        slot = int(number / 512)
        values[slot] += sorted_data[number]

    # Plotting the occurences in log-scale
    plt.bar(bins, values, width=359, color='b', log=True, align='center')

    # Adding labels and title
    plt.xlabel('Number of Tokens')
    plt.ylabel('Occurrences')
    plt.xticks(bins)
    plt.title('Number of lines of a certain token-length ')
    plt.show()


# TODO
def vectorize_tokenized_corpus(tokenized_corpus, save_file, save_every=50):
    """Computes the embeddings of each play in the corpus, line by line."""
    try:
        with open(save_file, 'rb') as save:
            vectorized_plays = pickle.load(save)
    except:
        vectorized_plays = dict()
    print(len(vectorized_plays))
    for identifier in tokenized_corpus:
        if identifier not in vectorized_plays:
            tokenized_play = tokenized_corpus[identifier]
            print(f'{len(vectorized_plays) + 1}/{len(tokenized_corpus)}')
            vectorized_play_item = vectorize_tokenized_play(tokenized_play)
            vectorized_plays[identifier] = vectorized_play_item
            if len(vectorized_plays) % save_every == 0:
                with open(save_file, 'wb') as save:
                    pickle.dump(vectorized_plays, save)
                print(f"Saved at {len(vectorized_plays)} plays")
    print("Done, saving...")
    with open(save_file, 'wb') as save:
        pickle.dump(vectorized_plays, save)
    return vectorized_plays


def compare_embeddings(embeddings1, embeddings2):
    """Computes the semantic similarity between two vectors, using a cosine similarity
    Args:
        embeddings1(torch.Tensor): First vector
        embeddings2(torch.Tensor): Second vector
    Returns:
        float: A number between 0 and 1. Closer to 1 is higher similarity."""
    # Calculate similarity
    similarity_score = cosine_similarity(embeddings1, embeddings2)
    return similarity_score.item()


## Using sentenceBert (no tokenization)

# def vectorize_sentence(sentence):
#     embedding = sentence_model.encode(sentence)
#     return embedding


def vectorize_sentence_play(play):
    locutors, sentences = [l for (l, _) in play], [s for (_, s) in play]
    vectorized_sentences = sentence_model.encode(sentences)
    return [(l, v) for (l, v) in zip(locutors, vectorized_sentences)]  # zip(locutors, vectorized_sentences)


def vectorize_sentence_corpus(corpus, save_file, save_every=10):
    """Computes the embeddings of each play in the corpus, line by line, and saves it"""
    try:
        with open(save_file, 'rb') as save:
            vectorized_plays = pickle.load(save)
    except:
        vectorized_plays = dict()
    print(len(vectorized_plays))

    # Computation time measurements
    total_time = 0
    computation_times = dict()
    save_file_time = open(os.path.join(PICKLE_DIR, 'Computation_times_SBERT_Dracor_vectorization.txt'), 'w')
    for identifier in corpus:
        if identifier not in vectorized_plays:
            play = corpus[identifier]
            print(f'{len(vectorized_plays) + 1}/{len(corpus)}')
            t_1 = time.time()
            vectorized_play_item = vectorize_sentence_play(play)
            time_to_vectorize = time.time() - t_1
            computation_times[identifier] = time_to_vectorize
            total_time += time_to_vectorize
            vectorized_plays[identifier] = vectorized_play_item
            if len(vectorized_plays) % save_every == 0:
                with open(save_file, 'wb') as save:
                    pickle.dump(vectorized_plays, save)
                save_file_time.write(
                    f'Average computation time : {total_time / len(vectorized_plays)} \n {json.dumps(computation_times)}')
                print(
                    f"Saved at {len(vectorized_plays)} plays, Average computation time : {total_time / len(vectorized_plays)} ")
    print("Done, saving...")
    with open(save_file, 'wb') as save:
        pickle.dump(vectorized_plays, save)
        save_file_time.write(
            f'Average computation time : {total_time / len(vectorized_plays)} \n {json.dumps(computation_times)}')
        print("Average computation time : {total_time / len(vectorized_plays)} ")
    return vectorized_plays


def compare_vectorized_plays(play1, play2, mode='levensthein', normalizing=True, max_distance=None):
    """ Given two already vectorized plays, computed the alignment and distance of both plays
    Args:
        play1(list): The first play
        play2(list) : The second play
    Returns:
        int: The distance between both plays
        list: The alignment between both plays"""
    embeddings1 = [vector for (locutor, vector) in play1]
    embeddings2 = [vector for (locutor, vector) in play2]
    total_length = len(embeddings1) + len(embeddings2)
    if mode == 'levensthein':
        distance, alignment = levenshtein_distance_and_alignment(embeddings1, embeddings2)
        if normalizing:
            distance = distance / total_length
        return distance, alignment
    elif mode == 'smith-waterman':
        alignment, score = smith_waterman_with_model(embeddings1, embeddings2, sentence_model, log_gap_cost)
        return score, alignment


def compare_vectorized_corpus(vectorized_corpus, save_file_permutation=None, save_file_dicts=None, save_every=500):
    """Compute the distance and alignment of all pairs of plays in the corpus,
    and find the most similar play for each play, for each of the 4 metrics.
    Saves progress periodically to allow resumption.
    """

    def create_empty_results():
        return {}, {}, {}, {}

    if save_file_dicts is None:
        match_percentages, average_costs, similarities, renaming_weights = create_empty_results()
        metric_sums = {'match': 0, 'cost': 0, 'similarity': 0, 'renaming': 0}
        similarity_count = 0
    else:
        with open(save_file_dicts, 'rb') as f:
            match_percentages, average_costs, similarities, renaming_weights, metric_sums = pickle.load(f)

    if save_file_permutation is None:
        init_iterations = 0
        all_ids = list(vectorized_corpus.keys())
        all_pairs = list(itertools.combinations(all_ids, 2))
        random.shuffle(all_pairs)
        nb_pairs = len(all_pairs)
        similarity_count = 0
    else:
        with open(save_file_permutation, 'rb') as f:
            l = pickle.load(f)
            init_iterations, nb_pairs, all_pairs = l[:3]
            similarity_count = l[3] if len(l) > 3 else 0
            if len(l) == 3:
                metric_sums['similarity'] = 0

    full_texts = get_full_text_corpus()

    for iterations in range(init_iterations, nb_pairs):
        dracor_id1, dracor_id2 = all_pairs[iterations]
        if (iterations + 1) % 500 == 0:
            print(f'{iterations + 1} / {nb_pairs}')

        play1, play2 = vectorized_corpus[dracor_id1], vectorized_corpus[dracor_id2]
        if play1 and play2:
            full_text_play_1 = full_texts[dracor_id1]
            full_text_play_2 = full_texts[dracor_id2]

            similarity, alignment = compare_vectorized_plays(play1, play2, mode='levensthein', normalizing=True)
            alignment_df = visualize_levenshtein_alignement(alignment, full_text_play_1, full_text_play_2, save=False)
            match_percentage, average_cost = evaluate_alignment(alignment_df)

            _, _, _, _, renaming_weight = get_character_renaming_from_alignment(full_text_play_1, full_text_play_2, alignment)

            metric_sums['match'] += match_percentage
            metric_sums['cost'] += average_cost
            metric_sums['similarity'] += similarity
            metric_sums['renaming'] += renaming_weight
            similarity_count += 1

            for id_a, id_b in [(dracor_id1, dracor_id2), (dracor_id2, dracor_id1)]:
                if match_percentage > match_percentages.get(id_a, (None, -1))[1]:
                    match_percentages[id_a] = (id_b, match_percentage)
                if average_cost < average_costs.get(id_a, (None, float('inf')))[1]:
                    average_costs[id_a] = (id_b, average_cost)
                if similarity > similarities.get(id_a, (None, -1))[1]:
                    similarities[id_a] = (id_b, similarity)
                if renaming_weight > renaming_weights.get(id_a, (None, -1))[1]:
                    renaming_weights[id_a] = (id_b, renaming_weight)

        if (iterations + 1) % save_every == 0 or (iterations + 1) == nb_pairs:
            print(f'Saving at {iterations + 1}...')
            with open(os.path.join('Data', 'Pickled saves', 'Alignments SBERT on Dracor', 'save_permutation.pkl'), 'wb') as f_perm:
                pickle.dump([iterations + 1, nb_pairs, all_pairs, similarity_count], f_perm)

            with open(os.path.join('Data', 'Pickled saves', 'Alignments SBERT on Dracor', 'save_dicts.pkl'), 'wb') as f_dict:
                pickle.dump([match_percentages, average_costs, similarities, renaming_weights, metric_sums], f_dict)

            print("Current averages:")
            print(f"  Match percentage: {metric_sums['match'] / (iterations + 1):.4f}")
            print(f"  Average cost: {metric_sums['cost'] / (iterations + 1):.4f}")
            print(f"  Renaming weight: {metric_sums['renaming'] / (iterations + 1):.4f}")
            print(f"  Similarity: {metric_sums['similarity'] / similarity_count:.4f} (based on {similarity_count} pairs)")
            print('Saving done')

    return match_percentages, average_costs, similarities, renaming_weights, metric_sums, similarity_count


def save_similarity_summary_to_csv(vectorized_corpus, output_csv='similarity_summary.csv'):
    """
    Computes similarity metrics across a corpus and saves a CSV summary.

    Args:
        vectorized_corpus (dict): A dictionary of play_id -> vectorized play.
        output_csv (str): Output file path.
    """
    match_percentages, average_costs, similarities, renaming_weights = compare_vectorized_corpus(vectorized_corpus)

    rows = []
    all_play_ids = vectorized_corpus.keys()

    for play_id in all_play_ids:
        row = {
            'Play Name': play_id,
            'Best Match (Match %)': match_percentages[play_id][0],
            'Match %': round(match_percentages[play_id][1], 3) if match_percentages[play_id][1] is not None else None,

            'Best Match (Avg Cost)': average_costs[play_id][0],
            'Avg Cost': round(average_costs[play_id][1], 3) if average_costs[play_id][1] is not None else None,

            'Best Match (Similarity)': similarities[play_id][0],
            'Similarity': round(similarities[play_id][1], 3) if similarities[play_id][1] is not None else None,

            'Best Match (Renaming)': renaming_weights[play_id][0],
            'Renaming Weight': round(renaming_weights[play_id][1], 3) if renaming_weights[play_id][
                                                                             1] is not None else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved similarity summary to {output_csv}")


def compare_sentence_embeddings(embedding1, embedding2):
    """Given two already encoded sentences, computes their similarity"""
    sim = sentence_model.similarity(embedding1, embedding2)
    return sim.item()


# Result is 0.19828132870063347
def get_average_distance(vectorized_corpus, n, bins_number=20):
    """Approximate the average distances between two random lines in the corpus and makes a graph of the distribution of distances.

    Computation is done with n iterations.
    Args:
        vectorized_corpus(dict): The corpus of vectorized plays
        n(int): the number of iterations to do to compute the average
        bins_number(int):number of categories
    Returns:
        float: the average distance"""
    full_text_dict = get_full_text_corpus()
    keys = list(vectorized_corpus.keys())
    avg_similarity = 0
    avg_of_squares = 0
    bins = [k * 1 / bins_number for k in range(bins_number)]
    values = [0 for _ in range(len(bins))]
    for _ in range(n):
        i1, i2 = random.sample(keys, 2)
        p1, p2 = vectorized_corpus[i1], vectorized_corpus[i2]
        index1, index2 = random.randint(0, len(p1) - 1), random.randint(0, len(p2) - 1)
        random_line1, random_line2 = p1[index1][1], p2[index2][1]
        # TMP : filtre pour garder uniquement des phrases un peu longues
        sentence_1, sentence_2 = full_text_dict[i1][index1][1], full_text_dict[i2][index2][1]
        if len(sentence_1) > 0 and len(sentence_2) > 0:
            sim = compare_sentence_embeddings(random_line1, random_line2)
            sim = max(0, sim)
            if sim > 0.9:
                print('Ces deux répliques sont similaires :')
                print(full_text_dict[i1][index1])
                print(full_text_dict[i2][index2])
            slot = int((bins_number - 0.00001) * sim)
            values[slot] += 1
            avg_similarity += sim
            avg_of_squares += sim ** 2
    avg_similarity = avg_similarity / n
    avg_of_squares = avg_of_squares / n
    print(f"Average similarity {avg_similarity}")
    print(f"Standard dev : {math.sqrt(avg_of_squares - avg_similarity ** 2)}")
    plt.xlim(0, 1.1)
    values_to_plot = values
    plt.bar(bins, values_to_plot, width=0.9 / bins_number, align='edge')
    plt.xlabel('Similarity Score', fontsize=22)
    plt.ylabel('Number of line pairs', fontsize=22)
    plt.xticks(np.arange(0, 1, step=5 / bins_number), fontsize=18)
    plt.yticks(fontsize=20)
    plt.title(f'Approximation of the density probability of similarity between embeddings, n= {n}', fontsize=22)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Density probability of similarity score.png'), format='png', dpi=300)
    plt.show()
    return avg_similarity / n

  # plt.title("Distribution of genre by corpus", fontsize=22)
  #   plt.xlabel("Genre", fontsize=22)
  #   plt.ylabel("Number of plays", fontsize=20)
  #   plt.xticks(rotation=45, fontsize=20)
  #   plt.yticks(fontsize=20)
  #   plt.legend(title="Corpus", fontsize=18, title_fontsize=20)


# There are 4 empty plays
def remove_empty_plays_from_corpus(vectorized_plays):
    """ Sanitize corpus of plays to remove empty plays"""
    empty_plays = 0
    vectorized_plays_cleaned = dict()
    for id in vectorized_plays:
        play = vectorized_plays[id]
        if len(play) == 0:
            empty_plays += 1
        else:
            vectorized_plays_cleaned[id] = play
    print(f"Empty plays : {empty_plays}")
    return vectorized_plays_cleaned


def similarity_to_levenshtein_cost(similarity, max_cutoff=0.65, min_cutoff=0.1):
    """ Transforms a similarity score into a substitution cost, to be used for a levenshtein distance computation.

    Uses a piecewise linear interpolation : extreme values before and after the cutoffs, and linear in between """
    if similarity >= max_cutoff:
        return 0
    elif similarity <= min_cutoff:
        return 2
    else:
        return 2 - 2 * (similarity - min_cutoff) / (
                max_cutoff - min_cutoff)  # Linear interpolation between min and max cutoff


# Base GPT
def levenshtein_distance_and_alignment(seq1, seq2, max_distance=None, /, insertion_cost=(lambda x: 1),
                                       deletion_cost=(lambda x: 1),
                                       substitution_cost=similarity_to_levenshtein_cost):
    """
    Computes the Levenshtein distance and alignment between two iterables.

    insertion, deletion, and substitution cost are parameterizable.

    Parameters:
    - seq1: The first iterable (e.g., string or list).
    - seq2: The second iterable (e.g., string or list).

    Returns:
    - distance: The Levenshtein distance between the two iterables.
    - alignment: A tuple of two aligned sequences with insertions/deletions represented as '-'.
    """
    # Initialize the distance matrix
    len1, len2 = len(seq1), len(seq2)
    dist = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Precompute all distances
    all_similarities = list(
        sentence_model.similarity(seq1, seq2))  # Compute all pairs of similarities for all embeddings
    all_similarities = [[substitution_cost(cost.item()) for cost in lines] for lines in
                        all_similarities]  # Convert them to a substitution cost

    # Fill the base cases for deletions/insertions
    for i in range(1, len1 + 1):
        dist[i][0] = dist[i - 1][
                         0] + 1  # deletion_cost(seq1[i - 1]) TODO: put back parametrization, done to try speeding the process up
    for j in range(1, len2 + 1):
        dist[0][j] = dist[0][j - 1] + 1  # insertion_cost(seq2[j - 1])

    # print('computing distance...')
    total_time = 0
    # Compute distances
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # embedding_1, embedding_2 = seq1[i - 1], seq2[j - 1] # Uncomment to allow parameterization of insertion/deletion cost
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                # Deletion. To allow parameterization by deletion cost, replace 1 by deletion_cost(embedding_1)
                dist[i][j - 1] + 1,
                # Insertion. To allow parameterization by insertion cost, replace 1 by insertion_cost(embedding_2)
                dist[i - 1][j - 1] + all_similarities[i - 1][j - 1]  # Substitution
            )
        current_max_distance = min(dist[i])
        if max_distance is not None and current_max_distance >= max_distance:  # If the distance goes over the threshold, exit early
            return -1, None

    # Backtrack to find the alignment
    i, j = len1, len2
    alignment = []

    # print("computing alignment")
    while i > 0 or j > 0:
        if i > 0 and dist[i][j] == dist[i - 1][j] + deletion_cost(seq1[i - 1]):
            alignment.append((i - 1, '-', deletion_cost(seq1[i - 1])))
            i -= 1  # Deletion
        elif j > 0 and dist[i][j] == dist[i][j - 1] + insertion_cost(seq2[j - 1]):
            alignment.append(('-', j - 1, insertion_cost(seq2[j - 1])))
            j -= 1  # Insertion
        else:
            alignment.append((i - 1, j - 1, all_similarities[i - 1][j - 1]))
            i -= 1
            j -= 1  # Substitution or match

    # Reverse the aligned sequences because we backtracked
    alignment.reverse()

    return dist[len1][len2], alignment


# Base with ChatGPT
def smith_waterman(seq1, seq2, similarity_matrix, gap_cost_fn=None):
    """
    Smith-Waterman local alignment algorithm.

    Parameters:
    - seq1, seq2: sequences to align
    - similarity_matrix: 2D list of similarity scores [len(seq1)][len(seq2)]
    - gap_cost_fn: a function f(gap_length) returning cost. For linear, f(x) = x * cost

    Returns:
    - alignment: list of tuples (i, j) representing aligned positions
    - max_score: alignment score
    """
    len1, len2 = len(seq1), len(seq2)
    H = [[0] * (len2 + 1) for _ in range(len1 + 1)]  # score matrix
    traceback = [[None] * (len2 + 1) for _ in range(len1 + 1)]  # backtrack pointers

    max_score = 0
    max_pos = (0, 0)

    # Fill in the score and traceback matrices
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            match = H[i - 1][j - 1] + similarity_matrix[i - 1][j - 1]
            if gap_cost_fn is None:  # Default to linear gap of cost 1 if not provided
                delete = H[i - 1][j] - 1
                insert = H[i][j - 1] - 1
            else:
                delete = max([H[i - k][j] - gap_cost_fn(k) for k in range(1, i + 1)])
                insert = max([H[i][j - k] - gap_cost_fn(k) for k in range(1, j + 1)])
            score = max(0, match, delete, insert)

            H[i][j] = score

            if score == 0:
                traceback[i][j] = None
            elif score == match:
                traceback[i][j] = (i - 1, j - 1, similarity_matrix[i - 1][j - 1])
            elif score == delete:
                for k in range(1, i + 1):
                    if H[i - k][j] - k == score:  # Change to gap_cost_fn(k)
                        traceback[i][j] = (i - k, j, k)  # Change the k when switching to different gap scores
                        break
            else:  # insert
                for k in range(1, j + 1):
                    if H[i][j - k] - k == score:  # Change to gap_cost_fn(k)
                        traceback[i][j] = (i, j - k, k)  # Change the k when switching to different gaps scores
                        break

            if score > max_score:
                max_score = score
                max_pos = (i, j)

    # Traceback
    alignment = []
    i, j = max_pos
    while traceback[i][j] is not None:
        score = traceback[i][j][2]
        alignment.append((i - 1, j - 1, score))  # subtract 1 to get original indices
        i, j, _ = traceback[i][j]
    alignment.reverse()
    print(alignment)
    return alignment, max_score


def smith_waterman_with_model(seq1, seq2, sentence_model, gap_cost_fn=None):
    all_similarities = list(sentence_model.similarity(seq1, seq2))
    similarity_matrix = [[cost.item() for cost in row] for row in all_similarities]
    return smith_waterman(seq1, seq2, similarity_matrix, gap_cost_fn)


def normalize_charname(charname):
    """Removes stage indications from charname"""
    if charname is None:
        print('Warning : empty charname')
        return ''
    charname = re.sub(r',.*', '', charname)
    charname = re.sub(r'\.', '', charname)
    charname = re.sub(r'#', '', charname)
    return charname


def log_gap_cost(k):
    return 0.5 * math.log1p(k)  # log1p(k) = log(1 + k)


# GPT
def get_character_renaming_from_alignment(play_1_text, play_2_text, alignment):
    """
    Args:
        play_1_text (dict): maps line_number -> character_name for play 1
        play_2_text (dict): maps line_number -> character_name for play 2
        alignment (list): list of (line_number_1, line_number_2, cost) tuples

    Returns:
        character_pair_counts (dict): {(char1, char2): count}
        most_common_for_1 (dict): {char1: char2}
        most_common_for_2 (dict): {char2: char1}
        optimal_matching (set): set of (char1, char2) tuples from max weight matching
    """
    character_pair_counts = defaultdict(int)

    for play_1_line_number, play_2_line_number, cost in alignment:
        if '-' not in (play_1_line_number, play_2_line_number):  # it's a match
            char1 = normalize_charname(play_1_text[play_1_line_number][0]) + '1'
            char2 = normalize_charname(play_2_text[play_2_line_number][0]) + '2'
            character_pair_counts[(char1, char2)] += 1
    # Most frequent match for each character in play 1
    most_common_for_1 = {}
    for char1, _ in character_pair_counts:
        matches = {pair[1]: character_pair_counts[pair] for pair in character_pair_counts if pair[0] == char1}
        if matches:
            most_common_for_1[char1] = max(matches.items(), key=lambda x: x[1])[0]

    # 2. Most frequent match for each character in play 2
    most_common_for_2 = {}
    for a, char2 in character_pair_counts:
        matches = {pair[0]: character_pair_counts[pair] for pair in character_pair_counts if pair[1] == char2}
        if matches:
            most_common_for_2[char2] = max(matches.items(), key=lambda x: x[1])[0]

    # Optimal bipartite matching
    G = nx.Graph()
    characters_play_1, characters_play_2 = {char1 for (char1, _) in character_pair_counts}, {char2 for (_, char2) in
                                                                                             character_pair_counts}
    if '' in characters_play_1 or '' in characters_play_2:
        print('Warning : empty charname')
    G.add_nodes_from(characters_play_1, bipartite=0)
    G.add_nodes_from(characters_play_2, bipartite=1)
    G.add_weighted_edges_from([(char1, char2, weight) for (char1, char2), weight in character_pair_counts.items()])
    optimal_matching = nx.algorithms.matching.max_weight_matching(G)
    optimal_matching = {tuple(sorted(c, key=lambda x: x[-1])) for c in
                        optimal_matching}  # Normalizing to put in char1 : char 2 format

    # The matching weight reflects what percentage of character lines are correctly aligned, according to the matching found
    total_weight = sum(character_pair_counts.values())
    matching_weight = sum(character_pair_counts[(char1, char2)] for (char1, char2) in optimal_matching)
    matching_score = matching_weight / total_weight

    return character_pair_counts, most_common_for_1, most_common_for_2, optimal_matching, matching_score


def visualize_levenshtein_alignement(alignment, play_1_text, play_2_text, play_1_title='Play_1', play_2_title='Play_2',
                                     output_dir='Outputs/Alignements', save=True):
    """Given an already computed alignement, generate a csv to visualize all operations"""
    rows = []
    for (play_1_element, play_2_element, cost) in alignment:
        operation_type = None
        if play_1_element == '-':
            operation_type = "Deletion"
            line_1 = ""
        else:
            line_number = play_1_element
            line_1 = play_1_text[line_number]

        if play_2_element == '-':
            operation_type = "Insertion"
            line_2 = ""
        else:
            line_number = play_2_element
            line_2 = play_2_text[line_number]

        if operation_type is None:
            if round(cost, 2) == 0:
                operation_type = "Match"
            else:
                operation_type = "Substitution"

        # rows.append({'Play 1': line_1, 'Play2': line_2, 'Operation Type': operation_type})
        # temp :
        rows.append(
            {play_1_title: line_1, play_2_title: line_2, 'Cost': round(cost, 2), 'Operation Type': operation_type})
    visualization = pd.DataFrame(rows)
    if save:
        visualization.to_csv(f'{output_dir}/alignement_{play_1_title}_{play_2_title}.csv')
    return visualization


def visualize_smith_waterman_alignment(seq1, seq2, alignment, play_1_title, play_2_title,
                                       output_dir='Outputs/Alignements'):
    """
    Save the Smith-Waterman alignment to a CSV file using pandas.

    Parameters:
    - seq1, seq2: original sequences (strings or list of tokens)
    - alignment: list of tuples (i, j, score) representing alignment positions and scores
    - filename: path to the output CSV file
    """
    data = []
    prev_i, prev_j = -1, -1

    for i, j, score in alignment:
        # Fill vertical (seq1) gaps
        for gap_i in range(prev_i + 1, i):
            data.append([seq1[gap_i], '-', 0])
        # Fill horizontal (seq2) gaps
        for gap_j in range(prev_j + 1, j):
            data.append(['-', seq2[gap_j], 0])

        token1 = seq1[i] if i >= 0 else "-"
        token2 = seq2[j] if j >= 0 else "-"
        data.append([token1, token2, score])
        prev_i, prev_j = i, j

    df = pd.DataFrame(data, columns=[play_1_title, play_2_title, "Score"])
    df.to_csv(f'{output_dir}/SWlogalignement_{play_1_title}_{play_2_title}.csv', index=False)
    return df


def compare_from_full_text(full_text_1, full_text_2, title1='Title1', title2='Title2', mode='levensthein'):
    """Given the full text of two plays, computes an alignment between both plays, saves it, and returns the associated character renamings.

    Args:
        full_text_1(list): The first play in format [(locutor, line)]
        full_text_2(list): The second play in format [(locutor, line)]
        title1(str): Title of the first play (optional)
        title2(str): Title of the second play (optional)
    Returns:
        dict: The dictionary counting of many times each pairs if characters is aligned between play1 and play2
        dict: The dictionary giving the renaming obtained by considering the most frequently corresponding character for each character of play 1
        dict: The dictionary giving the renaming obtained by considering the most frequently corresponding character for each character of play 2
        dict: The dictionary giving the renaming obtained by computing a maximum matching"""
    # This is what takes time : if you intend on calling this function often, you should pre-compute the vectorizations once instead
    print("Vectorizing ...")
    t1 = time.time()
    vec_1, vec_2 = vectorize_sentence_play(full_text_1), vectorize_sentence_play(full_text_2)
    print(f"Vectorizing in {time.time() - t1}")

    # Computing the distance and alignment
    dist, alignement = compare_vectorized_plays(vec_1, vec_2, mode)
    if mode == 'levensthein':
        print(f'Distance: {dist}')
    else:
        print(f'Score : {dist}')

    # Computing the associated renaming
    pairs_occurences, frequency_renaming_1, frequency_renaming_2, matching_renaming, matching_weight = get_character_renaming_from_alignment(
        full_text_1, full_text_2, alignement)

    # Saving the alignment
    if mode == "levensthein":
        alignment = visualize_levenshtein_alignement(alignement, full_text_1, full_text_2, title1, title2)
    else:
        alignment = visualize_smith_waterman_alignment(full_text_1, full_text_2, alignement, title1, title2)

    match_percentage, average_cost = evaluate_alignment(alignment)

    output_file = os.path.join('Outputs', 'Alignements', f'{title1}_{title2}_stats.txt')
    output = open(output_file, 'w')
    output.write(f"""
    Normalized Distance : {dist}
    Renaming : {matching_renaming}
    With score : {matching_weight}
    Match percentage : {match_percentage}
    Average cost : {average_cost}
    """)
    return pairs_occurences, frequency_renaming_1, frequency_renaming_2, matching_renaming


def compare_play_from_files(file1, file2):
    """Given two xml files of plays, computes an alignment between them, saves it, and returns associated renamings.
    Args:
        file1(str): path of the first play
        file2(str): path of the second play
    """
    play1, play2 = get_play_from_file(file1), get_play_from_file(file2)
    title1, title2 = get_title(play1) + '1', get_title(play2) + '2'
    full_text_1, full_text_2 = get_full_text(play1), get_full_text(play2)
    compare_from_full_text(full_text_1, full_text_2, title1, title2)


def get_plays_corpus_from_csv(csv_file, corpus_path):
    """Given a csv with couples of close plays, generates a dictionnary which keys are play identifiers and values are parsed plays
    Args:
        csv_file(str): path of the folder containing the couples of plays
        corpus_path(str) : the directory in which to find the xml files of the plays
    Returns:
        dict: The dictionary containing all parsed plays. Keys are play ID and values are of the form [(locutor, line)]"""
    close_casts = pd.read_csv(csv_file)
    resulting_corpus = dict()
    for _, row in close_casts.iterrows():
        play_1_file, play_2_file = f"{row['Name']}.xml", f"{row['Name_similar']}.xml"
        play_1_id, play_2_id = f"{row['id']}.xml", f"{row['id_similar']}.xml"
        if play_1_id not in resulting_corpus:
            play_1_path = os.path.join(corpus_path, play_1_file)
            play_1 = get_play_from_file(play_1_path)
            resulting_corpus[play_1_id] = get_full_text(play_1)
        if play_2_id not in resulting_corpus:
            play_2_path = os.path.join(corpus_path, play_2_file)
            play_2 = get_play_from_file(play_2_path)
            resulting_corpus[play_2_id] = get_full_text(play_2)
    return resulting_corpus


def compare_corpus_pairs(csv_file, vectorized_plays, corpus_path, mode='levensthein'):
    close_casts = pd.read_csv(csv_file)
    distances, renamings, pairs, matching_weights, match_percentages, avg_costs = [], [], [], [], [], []
    for _, row in close_casts.iterrows():
        # Getting the title and text of each play (needed for the alignment visualization
        play_1_title, play_2_title = row['Name'], row['Name_similar']
        play_1_file, play_2_file = f"{play_1_title}.xml", f"{play_2_title}.xml"

        print(play_1_title, play_2_title)
        play_1_path, play_2_path = os.path.join(corpus_path, play_1_file), os.path.join(corpus_path, play_2_file)
        play_1_text, play_2_text = get_full_text(get_play_from_file(play_1_path)), get_full_text(
            get_play_from_file(play_2_path))

        # Getting the vectorized plays (pre-computed)
        play_1_id, play_2_id = f"{row['id']}.xml", f"{row['id_similar']}.xml"
        vectorized_play_1, vectorized_play_2 = vectorized_plays[play_1_id], vectorized_plays[play_2_id]
        print(len(vectorized_play_1), len(vectorized_play_2))

        # Computing the alignement, distance, and visualization
        distance, alignment = compare_vectorized_plays(vectorized_play_1, vectorized_play_2, mode)
        print(f'Distance {play_1_title} et  {play_2_title} : {distance}')
        if mode == 'levensthein':
            alignment_df = visualize_levenshtein_alignement(alignment, play_1_text, play_2_text, play_1_title,
                                                            play_2_title,
                                                            'Outputs/Alignements/Close Casts')
        else:
            alignment_df = visualize_smith_waterman_alignment(play_1_text, play_2_text, alignment, play_1_title,
                                                              play_2_title,
                                                              'Outputs/Alignements/Close Casts')
        frequencies_pair, _, _, matching_renaming, matching_weight = get_character_renaming_from_alignment(play_1_text,
                                                                                                           play_2_text,
                                                                                                           alignment)
        match_percentage, avg_cost = evaluate_alignment(alignment_df)
        distances.append(distance)
        renamings.append(matching_renaming)
        pairs.append(frequencies_pair)
        matching_weights.append(matching_weight)
        match_percentages.append(match_percentage)
        avg_costs.append(avg_cost)

    close_casts = close_casts[['id', 'Name', 'id_similar', 'Name_similar', 'Intersection']]
    close_casts['Matching weight'] = matching_weights
    close_casts["Distances"] = distances
    close_casts["Renamings"] = renamings
    close_casts["Percent of matches"] = match_percentages
    close_casts["Average cost of operations"] = avg_costs
    # close_casts["Alignment Pair frequencies"] = pairs
    return close_casts


def fix_corpus_speakers(csv, vectorized_plays, corpus_path):
    """TMP"""
    close_casts = pd.read_csv(csv)
    for _, row in close_casts.iterrows():
        play_1_title, play_2_title = row['Name'], row['Name_similar']
        play_1_file, play_2_file = f"{play_1_title}.xml", f"{play_2_title}.xml"
        play_1_id, play_2_id = f"{row['id']}.xml", f"{row['id_similar']}.xml"

        print(play_1_title, play_2_title)
        play_1_path, play_2_path = os.path.join(corpus_path, play_1_file), os.path.join(corpus_path, play_2_file)
        play_1_text, play_2_text = get_full_text(get_play_from_file(play_1_path)), get_full_text(
            get_play_from_file(play_2_path))
        vectorized_play_1, vectorized_play_2 = vectorized_plays[play_1_id], vectorized_plays[play_2_id]
        vectorized_play_1 = [(fixed_locutor, vector) for ((fixed_locutor, _), (_, vector)) in
                             zip(play_1_text, vectorized_play_1)]
        vectorized_plays[play_1_id] = vectorized_play_1
        vectorized_play_2 = [(fixed_locutor, vector) for ((fixed_locutor, _), (_, vector)) in
                             zip(play_2_text, vectorized_play_2)]
        vectorized_plays[play_2_id] = vectorized_play_2
    return vectorized_plays


def evaluate_alignment(alignment):
    """Evaluates an alignement by returning the percentage of matches and the average cost of operations"""
    if type(alignment) is str:
        df = pd.read_csv(alignment)
    else:
        df = alignment
    total_lines = len(df)
    # Total number of "Match"
    match_count = (df['Operation Type'] == 'Match').sum()
    # Convert 'Cost' to numeric, forcing errors to NaN, and drop those
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
    average_cost = df['Cost'].mean()

    return match_count / total_lines, average_cost


import pandas as pd
import matplotlib.pyplot as plt


def plot_similarity(csv_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Ensure necessary columns exist
    required_columns = {'line', 'similarity', 'Average Score'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The CSV must contain columns: {required_columns}")

    # Map similarity to colors: 1 → green, 0 → red
    color_map = {1: 'green', 0: 'red'}
    colors = df['similarity'].map(color_map)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['line'], df['Average Score'], c=colors)
    plt.xlabel('Line')
    plt.ylabel('Average Score')
    plt.title('Average Score by Line Colored by Similarity')
    plt.grid(True)
    plt.show()


# TMP : Visualizing dama alignment quality
def plot_similarity(csv_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Ensure necessary columns exist
    required_columns = {'Line', 'Marked similar (after algorithm)', 'Average Score'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The CSV must contain columns: {required_columns}")

    # Replace commas with periods in 'Average Score' and convert to float
    df['Average Score'] = (
        df['Average Score']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

    # Drop rows with missing values in required columns
    df = df.dropna(subset=['Line', 'Marked similar (after algorithm)', 'Average Score'])

    # Map similarity to colors: 1 → green, 0 → red
    color_map = {1: 'green', 0: 'red'}
    colors = df['Marked similar (after algorithm)'].map(color_map)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Line'], df['Average Score'], c=colors)
    plt.xlabel('Line')
    plt.ylabel('Token Similarity Score')
    plt.title('Average token similarity score per line')
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label='Lines marked similar by annotators', markerfacecolor='green', markersize=10),
    #     Line2D([0], [0], marker='o', color='w', label='Lines not marked similar by annotators', markerfacecolor='red', markersize=10)
    # ]
    #
    # plt.legend(handles=legend_elements, title="Similarity")
    plt.grid(True)
    plt.show()
    plt.savefig('Dama Alignment visualization.png')

if __name__ == "__main__":

    # csv_file = "Imitation, création au théâtre - Dama Duende - Alignement annoté.csv"
    # plot_similarity(csv_file)
    # # Metadata
    # SBERT_comparison_dir = data_dir = os.path.join('Data', 'NLP Comparisons SBERT')
    # csv_close_casts = os.path.join(SBERT_comparison_dir, 'plays_with_cast_intersection_atleast6_ngramsimilarity.csv')
    # dracor_corpus_path = 'Corpus\\CorpusDracor - new'
    # save_file = os.path.join(PICKLE_DIR, 'SBERT_vectorized_Dracor.pkl')
    #
    # # # To compute the vectorization of all plays in the corpus
    # # corpus = get_full_text_corpus()
    # # vectorized_plays = vectorize_sentence_corpus(corpus, save_file, 30)
    #
    # # To load the vectorization of all plays, if already computed
    # print('Loading vectorized corpus ...')
    # vectorized_plays = pickle.load(open(save_file, 'rb'))
    # print('Corpus loaded.')
    #
    # save_perm = os.path.join('Data', 'Pickled saves', 'Alignments SBERT on Dracor', 'save_permutation.pkl')
    # save_dict = os.path.join('Data', 'Pickled saves', 'Alignments SBERT on Dracor', 'save_dicts.pkl')
    #
    # save_dict_fixed = os.path.join('Data', 'Pickled saves', 'Alignments SBERT on Dracor',
    #                                'save_dicts_similarity_fixed.pkl')
    #
    # compare_vectorized_corpus(vectorized_plays, save_file_permutation=save_perm, save_file_dicts=save_dict,
    #                           save_every=1000)

    # # To check the results of the corpus comparison
    # dicts = pickle.load(open(save_dict, 'rb'))
    # match_percentages, average_costs, similarities, renaming_weights, metric_sums = dicts
    # with open(save_perm, 'rb') as f:
    #     init_iterations, nb_pairs, all_pairs = pickle.load(f)
    # print(init_iterations)
    # print(renaming_weights.values())

    # result = compare_corpus_pairs(csv_close_casts, vectorized_plays, dracor_corpus_path, mode='levensthein')
    # result.to_csv(f'Outputs\\Alignements\\Close Casts\\close_casts_metadata.csv', index=False)

    # To compare two plays directly from their file (no need to pre-vectorize, slow)
    # data_dir = os.path.join('Data', 'NLP Comparisons SBERT')
    # play_file_1 = os.path.join(data_dir, 'moliere-precieuses-ridicules.xml')
    # play_file_2 = os.path.join(data_dir, 'somaize-veritables-precieuses.xml')
    # compare_play_from_files(play_file_1, play_file_2)

    # # Quick and dirty way to test pairs of plays in Corpus Piece Tres Proches
    # closePlaysFolder = os.path.join(os.getcwd(), 'Corpus', 'Corpus pieces tres proches')
    # corpus = dict()
    # full_text_dict = get_full_text_corpus()
    # play_names = os.listdir(closePlaysFolder)
    # for number in range(1, len(play_names)):
    #     plays_to_compare = [os.path.join(os.getcwd(), 'Corpus', 'Corpus pieces tres proches', plays) for plays in
    #                         play_names if plays[0] == str(number)]
    #     if plays_to_compare:
    #         play_1, play_2 = plays_to_compare
    #         compare_play_from_files(play_1, play_2)

    # # Testing the code on the close plays corpus
    # # For all possible pairs in the test corpus (trying to do it quickly)
    # for file in os.listdir(closePlaysFolder):
    #     play = get_play_from_file(os.path.join(closePlaysFolder, file))
    #     id = get_dracor_id(play)
    #     if id not in full_text_dict:
    #         text = get_full_text(play)
    #         full_text_dict[id] = text
    #     corpus[id] = full_text_dict[id]
    # close_vectorized_plays = vectorize_sentence_corpus(corpus, 'vectorized_close_plays.pkl')
    # title_dict = get_title_dict()
    # distances, alignments = compare_vectorized_corpus(close_vectorized_plays)
    # df = pd.DataFrame(distances)
    # new_column_names = {dracor_id: title_dict[dracor_id] for dracor_id in df.columns}
    # df.rename(columns=new_column_names, index=new_column_names, inplace=True)
    # df.to_csv('close_plays_distances.csv')
    # df = pd.DataFrame(alignments)
    # df.rename(columns=new_column_names, index=new_column_names, inplace=True)
    # df.to_csv('close_plays_alignments.csv')

    # Code to generate average_distance
    vec_corpus_path = os.path.join(PICKLE_DIR, 'SBERT_vectorized_Dracor.pkl')
    print('Opening...')
    file = open(vec_corpus_path, 'rb')
    print('Loading ...')
    vectorized_corpus = pickle.load(file)
    print('Loaded.')
    # vectorized_corpus = remove_empty_plays_from_corpus(vectorized_corpus)
    vectorized_corpus_cleaned = remove_empty_plays_from_corpus(vectorized_corpus)
    print(len(vectorized_corpus_cleaned))
    get_average_distance(vectorized_corpus_cleaned, 1000000, bins_number=100)
    #
    # # Running Levensthein corpus comparison
    # cutoff = 100
    # title_dict = get_title_dict()
    # distances, alignments = compare_vectorized_corpus(vectorized_corpus, cutoff)
    # df = pd.DataFrame(distances)
    # new_column_names = {dracor_id: title_dict.get(dracor_id, dracor_id) for dracor_id in df.columns}
    # df.rename(columns=new_column_names, index=new_column_names, inplace=True)  # replacing ids by titles in the output
    # df.to_csv(os.path.join(OUTPUT_DIR, f'levensthein_sentence_distances_{cutoff}.csv'), index=True, encoding='utf-8')
    #
    # cp = 'Textes par locuteur Dracor'
    # output_cp = 'BERT encoding per speaker'
    # for play in os.listdir(cp):
    #     print(play)
    #     file = os.path.join(cp, play)
    #     output_file = open(os.path.join(output_cp, play), 'wb')
    #     pk = open(file, 'rb')
    #     d = pickle.load(pk)
    #     for speaker in d:
    #         # full_speech = " ".join(d[speaker])
    #         # print(full_speech)
    #         t1 = time.perf_counter()
    #         averages = [process_long_sentence(sentence) for sentence in d[speaker]]
    #         t2 = time.perf_counter()
    #         print(f'processed {speaker} in {t2 - t1}')
    #         avg = sum(averages) / len(averages)
    #         d[speaker] = avg
    #     # print(d)
    #     pickle.dump(d, output_file)
