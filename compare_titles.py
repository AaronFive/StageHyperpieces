"""This file gathers all titles of the corpus, and find the most similar ones.
The output is a csv file with, for each title, the (semantically) closest different title in the corpus.
Duplicates in the input means multiple plays of the same name appear in the corpus. """
import math
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import graphing
from play_parsing import corpusFolder, get_play_from_file, get_title, get_dracor_id

# Load the BERT tokenizer and model
# modelname = 'flaubert/flaubert_large_cased'
# print('loading model ...')
# model = FlaubertModel.from_pretrained(modelname)
# print('loading tokenizer ...')
# tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)

#sentence_model = SentenceTransformer("dangvantuan/sentence-camembert-base")


def get_all_titles():
    """Returns the list of titles in the corpus selected in play_parsing module
     Returns:
        list: The list of all the titles of plays """
    titles = dict()
    for file in os.listdir(corpusFolder):
        play = get_play_from_file(os.path.join(corpusFolder, file))
        title = get_title(play)
        print(title)
        id = get_dracor_id(play)
        titles[id] = {'Title': title}
        # #For testing purposes on smaller datasets
        # if len(titles)>=10:
        #     break
    print('titles collected')
    return titles


def compare_all_titles(title_dict, similarity):
    """ Computes the closest pairs of titles
    Args:
        title_list (lst): the list of titles of plays in the corpus
        similarity (str*str -> int): a function computing the similarity between two strings
    Returns:
        dict: A dictionary whose keys are titles and values are the closest different title and the associated similarity """
    # Vectorizing all the titles once
    print('Vectorizing...')
    for id in title_dict:
        title_dict[id]['Vector'] = vectorize(title_dict[id]['Title'], model='SBERT')
    print('Vectorizing done.')
    ids_seen = set()
    avg, nb_pairs = 0, 0
    # We loop on every couple of title in the corpus
    for id in title_dict:
        title = title_dict[id]['Title']
        ids_seen.add(id)  # adding right away to prevent a title to be compared to itself
        print(f'{len(ids_seen)}/{len(title_dict)} : {title}')
        best_similarity = title_dict[id].get('Closest_similarity',
                                             0)  # Trying to get the closest similarity found so far, or 0 if there is none.
        for id_2 in title_dict:
            if id_2 not in ids_seen:  # Only checking for titles that haven't been seen yet
                title_2 = title_dict[id_2]['Title']
                current_similarity = similarity(title_dict[id]['Vector'], title_dict[id_2]['Vector'])
                # If the similarity is higher, update accordingly
                if current_similarity > best_similarity:
                    best_similarity = current_similarity
                    title_dict[id]['Closest_Title'] = title_2
                    title_dict[id]['Closest_id'] = id_2
                    title_dict[id]['Closest_similarity'] = best_similarity
                # Do the same for title_2 (useless as it is, but can be useful if the computation is too long or similarity
                # is lazy.
                if current_similarity > title_dict[id_2].get('Closest_similarity', 0):
                    title_dict[id_2]['Closest_Title'] = title
                    title_dict[id_2]['Closest_similarity'] = current_similarity
                    title_dict[id_2]['Closest_id'] = id
                nb_pairs += 1
                avg += current_similarity
    print(f'Average similarity : {avg / nb_pairs}')
    return title_dict


def vectorize(title, model='Flaubert'):
    """Converts a string into its vector representation using Flaubert model."""
    if model == 'Flaubert':
        # Tokenize the input text and add special tokens
        inputs = tokenizer(title, return_tensors="pt", padding=True, truncation=True)

        # Ensure all required inputs are passed to the model
        input_ids = inputs["input_ids"]  # Token IDs
        attention_mask = inputs["attention_mask"]  # Attention mask for padding

        # Obtain the Flaubert embeddings
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Extract the embeddings from the last hidden state
            last_hidden_states = outputs.last_hidden_state
            # Take the [CLS] token embedding (first token)
            cls_embedding = last_hidden_states[:, 0, :]  # Batch size 1, [CLS] token position
    else:
        cls_embedding = sentence_model.encode(title)

    return cls_embedding


# May return values slightly greater than one because of floating point errors.
def distance_title(embeddings1, embeddings2, model='SBERT'):
    """Computes the semantic similarity between two vectors, using a cosine similarity
    Args:
        embeddings1(torch.Tensor): First vector
        embeddings2(torch.Tensor): Second vector
    Returns:
        float: A number between 0 and 1. Closer to 1 is higher similarity."""
    # Calculate similarity
    if model == 'Flaubert':
        similarity_score = cosine_similarity(embeddings1, embeddings2)
        return similarity_score[0][0]
    else:
        return sentence_model.similarity(embeddings1, embeddings2).item()





if __name__ == "__main__":

    # Plotting similarity scores
    similarity_df = pd.read_csv('title_comparison_new.csv')
    similarity_scores = similarity_df['Closest_similarity'].to_list()
    print(similarity_scores)
    graphing.plot_similarities(similarity_scores, xlabel='Max Score', ylabel='Number of plays', title='Distribution of maximal title similarity scores across Dracor')

    # # Getting all titles and making the comparisons
    # d = distance_title
    # titles = get_all_titles()
    # distance_dict = compare_all_titles(titles, d)
    # print('comparison done')
    # # Preparing dataframe to export as csv
    # df = pd.DataFrame(distance_dict)
    # df = df.transpose()
    # df = df.drop('Vector', axis=1)
    # df['Closest_similarity'] = df['Closest_similarity'].astype(float).round(5)
    # df.index.name = 'Id'
    # # Saving output
    # f = open('to_delete.csv', 'w', encoding='utf8', newline='')
    # output = f
    # df.to_csv(output)
    # print('done')
