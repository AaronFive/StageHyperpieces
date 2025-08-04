import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from basic_utilities import OUTPUT_DIR
from play_parsing import get_play_from_file, get_scenes

def get_corpus_category(row):
    sources = {
        "TD": pd.notna(row["in_TD"]) and row["in_TD"] != "",
        "BD": pd.notna(row["in_BD"]) and row["in_BD"] != "",
        "Dracor": pd.notna(row["in_Dracor"]) and row["in_Dracor"] != ""
    }
    active = [k for k, present in sources.items() if present]
    if len(active) == 1:
        return active[0]
    elif len(active) >= 2:
        return "Multiple"
    else:
        return "None"

def plot_genre_distribution(df):
    # Liste des genres cibles
    target_genres = ["Comedy", "Tragedy", "Vaudeville", "Tragi-comedy"]

    # Nettoyage de la colonne "Genre (bis)"
    df["Genre (bis)"] = df["Genre (bis)"].fillna("").astype(str)

    def map_genre(genre):
        if genre == "":
            return "Unreferenced"
        elif genre in target_genres:
            return genre
        else:
            return "misc."

    df["Genre_grouped"] = df["Genre (bis)"].apply(map_genre)

    # Déterminer le corpus
    df["Corpus"] = df.apply(get_corpus_category, axis=1)

    # Comptage
    genre_corpus_counts = df.groupby(["Genre_grouped", "Corpus"]).size().unstack(fill_value=0)
    for col in ["TD", "BD", "Dracor", "Multiple"]:
        if col not in genre_corpus_counts.columns:
            genre_corpus_counts[col] = 0
    genre_corpus_counts = genre_corpus_counts[["TD", "BD", "Dracor", "Multiple"]]

    # Couleurs pastel
    pastel_colors = {
        "TD": "#74a9cf",
        "BD": "#fdae6b",
        "Dracor": "#d95f0e",
        "Multiple": "#9e9ac8"
    }

    # Ordre des genres
    genre_totals = genre_corpus_counts.sum(axis=1)
    if "Unreferenced" in genre_totals.index:
        ordered_genres = genre_totals.drop("Unreferenced").sort_values(ascending=False).index.tolist()
        ordered_genres.append("Unreferenced")
    else:
        ordered_genres = genre_totals.sort_values(ascending=False).index.tolist()

    genre_corpus_counts = genre_corpus_counts.loc[ordered_genres]

    # Plot avec polices augmentées
    ax = genre_corpus_counts.plot(
        kind="bar",
        stacked=True,
        color=pastel_colors,
        edgecolor="black",
        figsize=(12, 7)
    )

    plt.title("Distribution of genre by corpus", fontsize=22)
    plt.xlabel("Genre", fontsize=22)
    plt.ylabel("Number of plays", fontsize=20)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(title="Corpus", fontsize=18, title_fontsize=20)
    plt.tight_layout()
    plt.show()




def plot_century_distribution(df):
    # Couleurs par corpus (identiques à la fonction précédente)
    pastel_colors = {
        "TD": "#74a9cf",
        "BD": "#fdae6b",
        "Dracor": "#d95f0e",
        "Multiple": "#9e9ac8"
    }

    # Nettoyage des années
    df = df.copy()
    df["Normalized Year"] = pd.to_numeric(df["Normalized Year"], errors="coerce")
    df = df[df["Normalized Year"].notna()]
    # Restriction à la période 1500–2000
    df = df[(df["Normalized Year"] >= 1500) & (df["Normalized Year"] < 2000)]

    # Tranches de siècles (1500s à 1900s)
    bins = list(range(1500, 2100, 100))
    labels = [f"{start}s" for start in bins[:-1]]
    df["Century"] = pd.cut(df["Normalized Year"], bins=bins, right=False, labels=labels)
    df["Corpus"] = df.apply(get_corpus_category, axis=1)

    # Grouper par siècle et corpus
    counts = df.groupby(["Century", "Corpus"]).size().unstack(fill_value=0)

    # S’assurer que toutes les colonnes de corpus soient présentes
    for col in ["TD", "BD", "Dracor", "Multiple"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts[["TD", "BD", "Dracor", "Multiple"]]

    # Plot
    ax = counts.plot(kind="bar", stacked=True, color=pastel_colors, edgecolor="black", figsize=(10, 6))
    plt.title("Distribution of plays by century and corpus")
    plt.xlabel("Century")
    plt.ylabel("Number of plays")
    plt.xticks(rotation=45)
    plt.legend(title="Corpus")
    plt.tight_layout()
    plt.savefig('distribution_years_merged_corpus.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_scene_count_by_corpus(df, directory):
    pastel_colors = {
        "TD": "#74a9cf",
        "BD": "#fdae6b",
        "Dracor": "#d95f0e",
        "Multiple": "#9e9ac8"
    }

    df = df.copy()
    df["Corpus"] = df.apply(get_corpus_category, axis=1)

    # Filtrer les pièces non vides
    df = df[~df["Empty text"].astype(str).str.lower().isin(["true"])]

    data = []
    for idx, row in df.iterrows():
        raw_name = str(row["Raw Name"]).strip()
        file_path = os.path.join(directory, raw_name + ".xml")
        if not os.path.exists(file_path):
            continue
        if idx % 100 == 0:
            print(f"{idx}: {raw_name}")
        try:
            play = get_play_from_file(file_path)
            scenes = get_scenes(play)
            num_scenes = len(scenes)
            num_scenes = "70+" if num_scenes > 70 else num_scenes
            if num_scenes == 1:
                print(raw_name)
            data.append((row["Corpus"], num_scenes))
        except Exception as e:
            print(f"Erreur pour {raw_name}: {e}")

    if not data:
        print("Aucune donnée à afficher.")
        return

    scene_df = pd.DataFrame(data, columns=["Corpus", "Number of Scenes"])
    scene_counts = scene_df.groupby("Corpus")["Number of Scenes"].value_counts().unstack(fill_value=0)

    for col in ["TD", "BD", "Dracor", "Multiple"]:
        if col not in scene_counts.index:
            scene_counts.loc[col] = 0
    scene_counts = scene_counts.loc[["TD", "BD", "Dracor", "Multiple"]]

    scene_counts = scene_counts.transpose()
    scene_counts = scene_counts.sort_index(key=lambda x: [int(i) if i != "70+" else 999 for i in x])

    ax = scene_counts.plot(kind="bar", stacked=True, color=pastel_colors, edgecolor="black", figsize=(10, 6))
    plt.title("Distribution of number of scenes per play (grouping 70+), by corpus")
    plt.xlabel("Number of scenes")
    plt.ylabel("Number of plays")
    plt.xticks(rotation=0)
    plt.legend(title="Corpus")
    plt.tight_layout()
    plt.savefig("distribution_scenes_per_play_70_grouped_no_empty.png", dpi=300)
    plt.show()


def plot_scene_distribution_binned(df, directory):
    # Couleurs par corpus
    pastel_colors = {
        "TD": "#74a9cf",
        "BD": "#fdae6b",
        "Dracor": "#d95f0e",
        "Multiple": "#9e9ac8"
    }

    # Nettoyage des dates
    df = df.copy()
    df["Normalized Year"] = pd.to_numeric(df["Normalized Year"], errors="coerce")
    df = df[df["Normalized Year"].notna()]
    df = df[(df["Normalized Year"] >= 1650) & (df["Normalized Year"] <= 1725)]

    # Filtrage des pièces vides
    df = df[df["Empty text"] != True]

    from collections import defaultdict
    scene_counts = defaultdict(lambda: defaultdict(int))

    for i, row in df.iterrows():
        raw_name = str(row["Raw Name"]).strip()
        file_path = os.path.join(directory, raw_name + ".xml")
        if not os.path.exists(file_path):
            continue

        try:
            play = get_play_from_file(file_path)
            n_scenes = len(get_scenes(play))

            if n_scenes == 1:
                bin_label = "1"
            elif n_scenes > 70:
                bin_label = "70+"
                print(f"{raw_name} : {n_scenes} scenes")
            else:
                lower = 5 * ((n_scenes - 1) // 5) + 1
                upper = lower + 4
                if lower == 1:
                    lower = 2
                bin_label = f"{lower}-{upper}"

            corpus = get_corpus_category(row)
            scene_counts[bin_label][corpus] += 1

            if i % 100 == 0:
                print(f"{i}: {raw_name}, {n_scenes} scenes")

        except Exception as e:
            print(f"Error processing {raw_name}: {e}")

    scene_df = pd.DataFrame(scene_counts).T.fillna(0).astype(int)

    def sort_key(label):
        if label == "1":
            return 1
        if label == "70+":
            return 999
        return int(label.split("-")[0])

    scene_df = scene_df.loc[sorted(scene_df.index, key=sort_key)]

    for col in ["TD", "BD", "Dracor", "Multiple"]:
        if col not in scene_df.columns:
            scene_df[col] = 0
    scene_df = scene_df[["TD", "BD", "Dracor", "Multiple"]]

    ax = scene_df.plot(
        kind="bar",
        stacked=True,
        color=pastel_colors,
        edgecolor="black",
        figsize=(14, 6)
    )
    #plt.ylim(0, 500)
    plt.title("Scene count per play grouped by 5, by corpus", fontsize =22)
    plt.xlabel("Number of scenes (grouped)", fontsize=22)
    plt.ylabel("Number of plays", fontsize=22)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(title="Corpus", fontsize = 18, title_fontsize=20)
    plt.tight_layout()
    plt.savefig("distribution_scenes_per_play_binned_max500.png", dpi=300)
    plt.show()




def plot_similarities(to_plot, bins_number=20, title='ADD TITLE',
                      xlabel='', ylabel='', max_value=None, ticks_every=1):
    """Plots data in variable number of bins

    Computation is done with n iterations.
    Args:
        to_plot (lst): data to plot
        bins_number(int):number of categories
        title(str) : title of plot
    Returns:
        float: the average distance"""
    n = len(to_plot)
    avg, avg_of_squares = 0, 0
    if max_value is None:
        max_value = max(to_plot)  # Will give irregular bins
    bins = [k * max_value / bins_number for k in range(bins_number)]
    values = [0 for _ in range(len(bins))]
    for similarity in to_plot:
        sim = similarity
        slot = int(((bins_number - 0.00001) * sim) / max_value)
        values[slot] += 1
        avg += sim
        avg_of_squares += sim ** 2
    avg = avg / n
    avg_of_squares = avg_of_squares / n
    std_dev = math.sqrt(avg_of_squares - avg ** 2)
    print(f"Average {avg}")
    print(f"Standard dev : {std_dev}")
    plt.xlim(0, 1.1 * max_value)
    values_to_plot = values
    plt.bar(bins, values_to_plot, width= max_value / bins_number, align='edge', edgecolor='black')
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize = 22)
    plt.xticks(np.arange(0, max_value, step=ticks_every * max_value / bins_number), fontsize = 18)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize = 24)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{title}.png'), format='png', bbox_inches='tight', dpi=300)
    plt.show()
    print('done')
    return avg, std_dev

if __name__ == "__main__":
    supersheet_file = os.path.join("Corpus", "Merged corpus",
                                   "Corpus Merging Dracor-BD-TD - Supersheet -empty plays.csv")
    supersheet = pd.read_csv(supersheet_file)
    plot_scene_distribution_binned(supersheet, os.path.join("Corpus", "Merged corpus files"))
