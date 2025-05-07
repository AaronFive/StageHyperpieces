""" This file provides an example to modernize a Text written in Old French, using the model from Rachel Bawden
Accessible at https://huggingface.co/rbawden/modern_french_normalisation
NB 1: Something this to be offline so the example given in this Hugging Face page doesn't work unless you add the arguments
to suppress the post-processing
NB 2 : In some cases I don't understand, the normaliser gives IndexError. This happened because of the character "ô", because there was no space after a comma,
 or for other mysterious reasons"""

import re
import pandas
from transformers import pipeline


#### Functions used to modernize 17th century French into Modern French ###
def normalise_translated_text(s, remove_speakers=False):
    repliques = s.split("|")
    to_remove_re = r"Sc[eè]ne .* §|ACTE .* §|\[.*\]|\{.*\}|\*"
    repliques = [re.sub(to_remove_re, "", s).strip() for s in repliques]
    repliques = [re.sub(r',(?![^a-zA-Z])', ", ", s).strip() for s in repliques]
    repliques = [s.replace("’", "'") for s in repliques if s]
    repliques = [s.replace("-", " ") for s in repliques if s]
    if repliques and remove_speakers:
        repliques.pop(0)
    return repliques


line = 0


def log_normaliser(s, separator="|"):
    global line
    print(line)
    if s:
        try:
            new_s = normaliser(s)
            new_s = [d['text'] for d in new_s]
        except IndexError:
            print(f'{line} : {s}')
            new_s = s
    else:
        new_s = s
    new_s = separator.join(new_s)
    line += 1
    # print(new_s)
    return new_s


def log_normaliser_into_full_text(s, full_text):
    global line
    print(line)
    if s:
        try:
            new_s = normaliser(s)
            new_s = [d['text'] for d in new_s]
        except IndexError:
            print(f'{line} : {s}')
            new_s = s
    else:
        new_s = s
    speaker = new_s[0]
    rest = new_s[1:]
    full_text.append((speaker, rest))
    line += 1
    # print(new_s)
    return new_s


s1 = ['Veille-je,ou si je dors ? Dieux je ne puis penser', "Que mon frere ait jamais eu dessein d'offencer",
      "Un amy que j'estime à l'esgal de moy-mesme :", "Mon frere,vous devez aymer celuy que j'ayme ;",
      "Que j'en sçache la cause."]

if __name__ == "__main__":
    # normaliser = pipeline(model="rbawden/modern_french_normalisation", no_postproc_lex=True, no_post_clean=True,
    #                       batch_size=32, beam_size=5, cache_file="./cache.pickle", trust_remote_code=True)
    normaliser = pipeline(model="rbawden/modern_french_normalisation",
                          batch_size=256, beam_size=5, cache_file="./cache.pickle", trust_remote_code=True)
    # s2 = ['Quelle folie, ô Dieux ! Des enchanteurs ? De mesme.']
    # new_s = normaliser(s2)

    # Read the CSV file and format columns correctly
    dama = open("DamaDuende Alignement Modernized.csv", "r", encoding='utf8')
    dama_df = pandas.read_csv(dama).fillna("")
    dama_df["Esprit Folet"] = dama_df["Esprit Folet"].astype(str)
    dama_df["Esprit Folet"] = dama_df["Esprit Folet"].apply(normalise_translated_text)

    # # Normalising the Old French text
    # dama_df["Modernisé"] = dama_df["Esprit Folet"].apply(log_normaliser)

    # Getting the full text for dama duende
    full_text = []
    dama_df["Modernisé"] = dama_df["Esprit Folet"].apply(lambda x: log_normaliser_into_full_text(x, full_text))
    print(full_text)

    # # Save into a new file
    # dama_df.to_csv(open("DamaDuende Alignement Modernized.csv", 'w', encoding='utf8', newline=''))
