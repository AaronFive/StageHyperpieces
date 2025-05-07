import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords français si ce n'est pas déjà fait
nltk.download('stopwords')

# Liste des mots-outils français
mots_outils = set(stopwords.words('french'))

# Charger le texte
with open("texte_non_zipfien.txt", "r", encoding="utf-8") as f:
    texte = f.read()

# Nettoyer et découper le texte en mots
mots = re.findall(r'\b\w+\b', texte.lower())

# Supprimer les mots-outils
mots_filtres = [mot for mot in mots if mot not in mots_outils]

# Compter les mots restants
frequences = Counter(mots_filtres)

# Top 50
mots_tries = frequences.most_common(1500)
mots_labels, valeurs = zip(*mots_tries)
if __name__ == "__main__":
    # Affichage
    plt.figure(figsize=(14, 6))
    plt.bar(mots_labels, valeurs)
    plt.xticks(rotation=45, ha='right')
    plt.title("Fréquence des 50 mots les plus fréquents (sans mots-outils)")
    plt.xlabel("Mots")
    plt.ylabel("Occurrences")
    plt.tight_layout()
    plt.show()
