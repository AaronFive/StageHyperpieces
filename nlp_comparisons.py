import os
import pickle
import random
import re
from heapq import nlargest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

from basic_utilities import pickle_file, OUTPUT_DIR
from pickled_data import get_tokenized_corpus, get_title_dict, get_full_text_corpus
from play_parsing import get_play_from_file, get_dracor_id, get_title, get_full_text
from sentence_transformers import SentenceTransformer
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords

# camembert = CamembertForMaskedLM.from_pretrained('camembert-base')
# camembert = camembert.cuda()
# tokenizer = AutoTokenizer.from_pretrained('camembert-base')


print('Loading tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
print('Loading model')
#model = BertModel.from_pretrained('bert-base-multilingual-cased')
sentence_model = SentenceTransformer("dangvantuan/sentence-camembert-large")



# TODO:
# Pour chaque réplique, segmenter en ensembles de phrases de moins de 512 caractères
# Tokenisation, vectorisation par token
# Sauvegarder les tokens
# (Enlever les mots vides)
# Pour chaque couple identifié, regarder les 10 meilleurs mots en commun (en regardant sur toutes les paires)
# Puis sur toutes les répliques


# BELLE EGYPTIENNE
# Prendre toutes les paires de répliques dans chacune, enlever les mots vides, et calculer un score de nombre de mots en communs
# Vérifier si les couples qu'on a identifiés à la main sont parmis le top

def average_embeddings(embeddings, attention_mask):
    return (attention_mask[..., None] * embeddings).mean(1)


def similarity_score_sentences(sentence1, sentence2):
    batch_sentences = [sentence1, sentence2]
    tokenizer_output = tokenizer(
        batch_sentences,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    inputs_ids, attention_masks = tokenizer_output.input_ids, tokenizer_output.attention_mask
    with torch.no_grad():
        model_output = camembert(inputs_ids, attention_masks, output_hidden_states=True)
    token_embeddings = model_output.hidden_states[-1]
    avg_sentence_representations = average_embeddings(token_embeddings, tokenizer_output.attention_mask)
    avg_similarity_score = F.cosine_similarity(avg_sentence_representations[0], avg_sentence_representations[1], dim=-1)
    return avg_similarity_score


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

def vectorize_sentence(sentence):
    embedding = sentence_model.encode(sentence)
    return embedding.item()

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
def vectorize_corpus(tokenized_corpus, save_file, save_every=50):
    """Computes the embeddings of each play in the corpus, line by line."""
    try:
        with open(save_file, 'rb') as save:
            vectorized_plays = pickle.load(save)
    except Exception:
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


def compare_vectorized_plays(play1, play2):
    embeddings1 = [vector for (locutor, vector) in play1]
    embeddings2 = [vector for (locutor, vector) in play2]
    distance, alignment = levenshtein_distance_and_alignment(embeddings1, embeddings2)
    return distance, alignment


def compare_vectorized_corpus(vectorized_corpus):
    seen_plays = set()
    distances = dict()
    alignments = dict()
    for dracor_id1 in vectorized_corpus:
        print(dracor_id1)
        seen_plays.add(dracor_id1)
        distances[dracor_id1] = dict()
        alignments[dracor_id1] = dict()
        for dracor_id2 in vectorized_corpus:
            if dracor_id2 not in seen_plays:
                print(f'Computing distance with {dracor_id2}')
                play1, play2 = vectorized_corpus[dracor_id1], vectorized_corpus[dracor_id2]
                similarity, alignment = compare_vectorized_plays(play1, play2)
                distances[dracor_id1][dracor_id2] = similarity
                alignments[dracor_id1][dracor_id2] = alignment
    return distances, alignments

def compare_sentence_embeddings(embedding1, embedding2):
    sim = sentence_model.similarity(embedding1, embedding2)
    return sim.item()

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


# Result is 0.8335
def get_average_distance(vectorized_corpus, n, bins_number=50):
    """Approximate the average distances between two random lines in the corpus and makes a graph of the distribution of distances.

    Computation is done with n iterations.
    Args:
        vectorized_corpus(dict): The corpus of vectorized plays
        n(int): the number of iterations to do to compute the average
    Returns:
        float: the average distance"""
    keys = list(vectorized_corpus.keys())
    avg_similarity = 0
    bins = [k * 1 / bins_number for k in range(bins_number)]
    values = [0 for _ in range(len(bins))]
    for _ in range(n):
        i1, i2 = random.sample(keys, 2)
        p1, p2 = vectorized_corpus[i1], vectorized_corpus[i2]
        index1, index2 = random.randint(0, len(p1) - 1), random.randint(0, len(p2) - 1)
        random_line1, random_line2 = p1[index1][1], p2[index2][1]
        sim = compare_embeddings(random_line1, random_line2)
        if sim == 1:
            print('Ces deux répliques sont similaires :')
            # print(full_text_dict[i1][index1])
            # print(full_text_dict[i2][index2])
            # print()
        slot = int((bins_number - 0.001) * sim)  # The 0.001 is
        values[slot] += 1
        avg_similarity += sim
    print(f"Average similarity {avg_similarity / n}")
    plt.xlim(0, 1.1)
    values = [v for v in values]
    plt.bar(bins, values, width=0.9 / bins_number, align='edge')
    plt.xlabel('Similarity Score')
    plt.ylabel('Number of line pairs')
    plt.xticks(np.arange(0, 1, step=2 / bins_number))
    plt.title(f'Approximation of the density probability of similarity between embeddings, n= {n}')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Density probability of similarity score.png'), format='png', dpi=300)
    plt.show()
    return avg_similarity / n


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


def similarity_to_levenshtein_cost(similarity, max_cutoff, min_cutoff):
    """ Transforms a similarity score into a substitution cost, to be used for a levenshtein distance computation.

    Uses a piecewise linear interpolation : extreme values before and after the cutoffs, and linear in between """
    if similarity >= max_cutoff:
        return 0
    elif similarity <= min_cutoff:
        return 2
    else:
        return 2 - 2 * (similarity - min_cutoff) / (
                max_cutoff - min_cutoff)  # Linear interpolation between min and max cutoff


def substitution_cost_levenshtein(embedding1, embedding2, max_cutoff=0.96, min_cutoff=0.7, sentence=False):
    """ Given two embeddings, compute the substitution cost to transform one into the other"""
    if sentence:
        similarity = sentence_model.similarity(embedding1, embedding2)
    else:
        similarity = compare_embeddings(embedding1, embedding2)
    cost = similarity_to_levenshtein_cost(similarity, max_cutoff, min_cutoff)
    return cost


def levenshtein_distance_and_alignment(seq1, seq2, max_distance=None, /, insertion_cost=(lambda x: 1),
                                       deletion_cost=(lambda x: 1),
                                       substitution_cost=substitution_cost_levenshtein):
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

    # Fill the base cases for deletions/insertions
    for i in range(1, len1 + 1):
        dist[i][0] = dist[i - 1][
                         0] + 1  # deletion_cost(seq1[i - 1]) TODO: put back parametrization, done to try speeding the process up
    for j in range(1, len2 + 1):
        dist[0][j] = dist[0][j - 1] + 1  # insertion_cost(seq2[j - 1])

    print('computing distance...')

    # Compute distances
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            embedding_1, embedding_2 = seq1[i - 1], seq2[j - 1]
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                # deletion_cost(embedding_1),  # Deletion TODO: put back parametrization, done to try speeding the process up
                dist[i][j - 1] + 1,  # insertion_cost(embedding_2),  # Insertion
                dist[i - 1][j - 1] + substitution_cost(embedding_1, embedding_2)  # Substitution
            )

    # Backtrack to find the alignment
    i, j = len1, len2
    alignment = []

    print("computing alignment")
    while i > 0 or j > 0:
        if i > 0 and dist[i][j] == dist[i - 1][j] + deletion_cost(seq1[i - 1]):
            alignment.append((i - 1, '-', deletion_cost(seq1[i - 1])))
            i -= 1  # Deletion
        elif j > 0 and dist[i][j] == dist[i][j - 1] + insertion_cost(seq2[j - 1]):
            alignment.append(('-', j - 1, insertion_cost(seq2[j - 1])))
            j -= 1  # Insertion
        else:
            alignment.append((i - 1, j - 1, substitution_cost(seq1[i - 1], seq2[j - 1])))
            i -= 1
            j -= 1  # Substitution or match

    # Reverse the aligned sequences because we backtracked
    alignment.reverse()

    return dist[len1][len2], alignment

def visualize_levenshtein_alignement(alignment, play_1_text, play_2_text, play_1_title='Play_1', play_2_title='Play_2'):
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
        rows.append({'Play 1': line_1, 'Play2': line_2, 'Op symbol1': play_1_element, 'Op symbol2': play_2_element,
                     'Cost': round(cost, 2), 'Operation Type': operation_type})
    visualization = pd.DataFrame(rows)
    visualization.to_csv(f'Outputs/Alignements/alignement_{play_1_title}_{play_2_title}.csv')


def compare_play_from_files(file1, file2):
    play1,play2 = get_play_from_file(file1), get_play_from_file(file2)
    title1,title2 = get_title(play1), get_title(play2)
    full_text_1, full_text_2 = get_full_text(play1, True), get_full_text(play2, True)
    tok1, tok2 = [(locutor, tokenizer.tokenize(text)) for (locutor, text) in full_text_1], [(locutor, tokenizer.tokenize(text)) for (locutor, text) in full_text_2]
    vec_1, vec_2 = vectorize_tokenized_play(tok1), vectorize_tokenized_play(tok2)
    dist, alignement = compare_vectorized_plays(vec_1, vec_2)
    print(f'Distance: {dist}')
    visualize_levenshtein_alignement(alignement, full_text_1, full_text_2, title1, title2)

if __name__ == "__main__":

    # # Generate alignment from 2 plays
    # file_1 = 'Corpus/Corpus pieces tres proches/8 - Didon se sacrifiant.xml'
    # file_2 = 'Corpus/Corpus pieces tres proches/8-JODELLE-DIDON.xml'
    # compare_play_from_files(file_1, file_2)


    # Test on punctuation
    # test_text_1 =  "Écoutez, et vous saurez comment Son trépas seul m'oblige à cet éloignement. Après six ans passés, depuis notre voyage, Dans les plus grands plaisirs qu'on goûte au mariage, Mon père, tout caduc, émouvant ma pitié, Je conjurai Médée, au nom de l'amitié..."
    # test_text_1_punct = "Écoutez, et vous saurez comment Son trépas seul m'oblige à cet éloignement. Après six ans passés, depuis notre voyage, Dans les plus grands plaisirs qu'on goûte au mariage, Mon père, tout caduc, émouvant ma pitié, Je conjurai Médée, au nom de l'amitié."
    # test_text_2 = "Écoutez, et vous saurez comment Son trépas seul me force à cet éloignement. Après six ans passés, depuis notre voyage, Dans les plus grands plaisirs qu'on goûte au mariage, Mon père tout caduc émouvant ma pitié, Je conjurai Médée, au nom de l'amitié."
    test_text_1 = 'Ma constance éprouvée en choses plus ardues   Choses qui font trembler les plus forts  entendues   Me purge du soupçon de telle lâcheté   La gloire au plus haut prix j’ai toujours acheté  Ennemi du repos  ennemi des délices   Mais quand nous nous sentons de cruautés complices  Quand il est question de rompre une amitié  Envers nos bienfaiteurs plus dignes de pitié  Ha   Cieux   ha   justes Cieux  alors la conscience Jette un trouble dans l’âme affreux d’impatience  Nous portons contre nous de terribles témoins  Et les plus généreux  alors le sont le moins '
    test_text_2 =  'Encor oublions nous  qu outre l ailé Mercure   Plus sûrs encor nous doit rendre un céleste augure   Alors qu au sac piteux notre Troie était pleine  Du feu  de pleurs  de meurtre  une flamme soudaine  Vint embraser mon chef  qui comme notre Anchise  L expliqua  nous chassait hors de la Troie prise   Je jure par l honneur de cette même tête   Par celle de mon père  et par la neuve fête  Que le tombeau d Anchise ajoute à notre année   Qu un même embrasement m a cette matinée  Donné le même signe   et qu on nous tient promesse  De revenger bientôt la Troie de la Grèce  '

    # test_text_1 =  'La crainte du futur  du futur  que les Dieux  Sous l’ombre d’un repos dérobent à nos yeux '
    # test_text_2 = 'La peur du futur  de l\'avenir  que les Hommes  Sous l’ombre d’un repos dérobent à nos yeux '
    tok1 = [('_',tokenizer.tokenize(test_text_1))]
    tok2 = [('_',tokenizer.tokenize(test_text_2))]
    ch1 = chunk_tokens(tok1[0][1])
    ch2 = chunk_tokens(tok2[0][1])
    print(ch1)
    print(ch2)
    v1 = vectorize_tokenized_play(tok1)
    v2 = vectorize_tokenized_play(tok2)
    sim = compare_embeddings(v1[0][1], v2[0][1])
    print(sim)
    print(similarity_to_levenshtein_cost(sim, 0.96, 0.7))

    # # Testing the code on the close plays corpus
    # closePlaysFolder = os.path.join(os.getcwd(), 'Corpus', 'Corpus pieces tres proches')
    # tokenized_close_plays = tokenize_corpus(closePlaysFolder, 'tokenized_close_plays')
    # close_vectorized_plays = vectorize_corpus(tokenized_close_plays, 'vectorized_close_plays.pkl')
    # title_dict = get_title_dict()
    # distances, alignments = compare_vectorized_corpus(close_vectorized_plays)
    # print(distances)
    # df = pd.DataFrame(distances)
    # new_column_names = {dracor_id: title_dict[dracor_id] for dracor_id in df.columns}
    # df.rename(columns = new_column_names, index = new_column_names, inplace=True)
    # df.to_csv('close_plays_distances.csv')
    # df = pd.DataFrame(alignments)
    # df.rename(columns=new_column_names, index=new_column_names, inplace=True)
    # df.to_csv('close_plays_alignments.csv')

    # # Code to generate average_distance
    # vec_corpus_path = 'vectorized_corpus.pkl'
    # print('Opening...')
    # file = open(vec_corpus_path, 'rb')
    # print('Loading ...')
    # vectorized_corpus = pickle.load(file)
    # print('Loaded.')
    # vectorized_corpus_cleaned = remove_empty_plays_from_corpus(vectorized_corpus)
    # print(len(vectorized_corpus_cleaned))
    # get_average_distance(vectorized_corpus_cleaned, 1000000)

    ''''# Running Levensthein corpus comparison
    distances, alignments = compare_vectorized_corpus(vectorized_corpus)
    df = pd.DataFrame(distances)
    df.to_csv(os.path.join(OUTPUT_DIR, 'levensthein_distances.csv'), index=True, encoding='utf-8')'''

    # d = pd.read_csv(open('to_merge.csv', 'r', newline=''))
    # d = d['similarity']
    # distances_df = pd.read_csv(open('dama_close_tokens.csv', 'r', encoding='utf8'))
    # merged_df = pd.merge(d, distances_df, left_index=True, right_index=True, how='inner')
    # print(merged_df.columns)
    # merged_df['Unnamed: 0'] = merged_df['Unnamed: 0'] + 1
    # merged_df.rename(columns={'Unnamed: 0': 'Line'}, inplace=True)
    # merged_df = merged_df[['Line', 'Google Trad', 'Modernisé', 'similarity', '5_closest']]
    # print(merged_df.head(5))
    # merged_df.to_csv(open('dama_close_tokens_new.csv', 'w', newline='', encoding='utf8'), index=False)

    # # Tokenizing text and saving in a pickle file
    # # # Loading data and formatting it correctly
    # pickle_save = open("vectors_and_tokens_dama.pkl", 'rb')
    # dama_df = pickle.load(pickle_save)
    # # dama_df = pd.read_csv(dama).fillna("")
    # columns_to_vectorize = ["Esprit Folet", "Modernisé"]
    # # dama_df = dama_df[columns_to_keep]
    # save_file = open("vectors_and_tokens_dama_updated.pkl", 'wb')
    # for c in columns_to_vectorize:
    #     dama_df[c] = dama_df[c].astype('str')
    #     dama_df[[f'{c}_vectors', f'{c}_tokens']] = dama_df[c].apply(lambda x: pd.Series(process_long_sentence(x)))
    #     line = 0
    # pickle.dump(dama_df, save_file)
    # for c in columns_to_vectorize:
    #     dama_df[f'{c}_tokens'] = dama_df[f'{c}_tokens'].apply(remove_pads)
    # dama_df.to_csv(open('full_tokens_and_vectors_dama.csv', 'w', newline='', encoding='utf8'))
    #
    # # Opening pickle file and saving in a csv, for visualization purpose
    # t = time.perf_counter()
    # dama_df = pickle.load(open('vectors_and_tokens_dama.pkl', 'rb'))
    # loading_time = time.perf_counter() - t
    # print(f'loaded in {loading_time}')
    # print(dama_df.columns)
    # dama_df["Google Trad_tokens"] = dama_df["Google Trad_tokens"].apply(remove_pads)
    # dama_df.to_csv(open("vectors_and_tokens_dama_full.csv", 'w', newline='', encoding='utf8'))
    # print(f'done writing')
    #
    # # Reading df and comparing closest tokens
    # print('here')
    # dama_df = pickle.load(open("dama_vectors_token_unpadded.pkl", 'rb'))
    # print('loaded')
    # print(dama_df)
    # # Removing padding
    # for columns in ["Google Trad_tokens", "Modernisé_tokens", "Esprit Folet_tokens"]:
    #     dama_df[columns] = dama_df[columns].apply(remove_pads)
    #     pickle.dump(dama_df, open("dama_vectors_token_unpadded.pkl", 'wb'))
    # k = 5
    # dama_df[f'{k}_closest'] = dama_df.apply(
    #     lambda row: get_k_closest_tokens(row["Google Trad_vectors"], row["Modernisé_vectors"],
    #                                      row["Google Trad_tokens"], row["Modernisé_tokens"], k), axis=1)
    # pickle.dump(dama_df, open("dama_distances.pkl", 'wb'))
    # small_df = dama_df[["Google Trad", "Modernisé", f'{k}_closest']]
    # small_df.to_csv(open("dama_close_tokens.csv", 'w', newline='', encoding='utf8'))
    # for i, l in enumerate(dama_df[f'{k}_closest']):
    #     print(f'Ligne {i} :')
    #     for x in l:
    #         print(x)
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
