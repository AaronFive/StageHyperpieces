# README EXTRACTION DIFFÉRENTS AUTEURS

## Dracor :

### Déjà disponible dans le XML (récupéré dans Bibliothèque Dramatique). Dans ce cas de figure :

```xml
<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="fre">
    <teiHeader>
        <fileDesc>
            ...
            <sourceDesc>
                <bibl type="digitalSource">
                    ...
                    <bibl type="originalSource">
                        <date type="written" when=AAAA/>
                        <date type="print" when=AAAA/>
                        <date type="premiere" when="AAAA-MM-JJ">...</date>
                        <idno type="URL">...</idno>
                    </bibl>
                </bibl>
            </sourceDesc>
        </fileDesc>
```

On peut donc obtenir dans la balise <date> une année qui diffère en fonction de la valeur de son attribut "type". 

On peut obtenir l'année où l'oeuvre a été écrite (type="written")
Son année d'impression. (type="print")
Et l'année de sa première représentation de la ppièce (type="premiere").

Afin de dater une oeuvre avec une date unique, on lui associe donc une année normalisée en fonction de certains critères :

Par défaut, on prend l'année la plus chronologiquement ancienne entre celle d'édition et de première représentation.

Cependant, si la pièce a été écrite il y a plus de 10 avant l'année en question, c'est l'année d'écriture de la pièce qui sera gardée, afin de garder une certaine cohérence par rapport au contexte de l'époque où la pièce a été rédigée.

En cas d'absence de certaines données, on ne fait la comparaison qu'avec les années restantes.

## Théâtre Documentation

Ici, l'affaire est moins évidente. Les pièces ne sont gardées que sous format html, il faut donc les convertir en xml en essayant de retrouver les données qui nous intéressent. L'année n'est typiquement pas conservée en tant que méta-donnée, il faut donc la trouver selon certains patterns.

La plupart du temps, l'année de première représentation est écrite sous la forme :

- "Représentée ..., le JJ [mois] AAAA."


(Dans le cas d'un Vaudeville ou autre type de pièce dont le genre est masculin, "Représenté" est écrit à la place).

-"Représenté(e) en AAAA. 

(variante avec que l'année).

- Parfois, nous avons simplement une ligne avec écrit :

AAAA.

On part alors du principe que c'est l'année d'impression.Pour le moment aucun pattern d'affichage pour les années d'impression et d'écriture.

- Imprimée en AAAA.

Année d'impression.

- Plusieurs dates en une phrase :

Publié(e) en AAAA et représenté(e) ... le JJ [mois] AAAA.
Publié(e) en AAAA et représenté(e) en AAAA.

- Non représenté.

- MM AAAA.



Cas particulier :

- Mêlée de musique et d’entrées de ballet, représentée pour le Roi Saint-Germain-en-Laye, au mois de févier 1670 sous le titre de divertissement royal. 
(Les Amants magnifiques, Molière.)

- Irrégularité &nbsp; (\xa0 dansle parsing de python) au lieu de ' ' pour la grand-mère de Victor Hugo. 

- Fautes de frappe : 

févier au lieu de février pour Les Amants magnifiques de Molière
lévrier au lieu de février pour : le peintre exigeant de Tristan BERNARD.
Oubli d'espace (25avril au lieu de 25 avril) pour "Adieux des officiers ou venus justiifée" de Charles Dufresny.
Un espace de trop

Des exemple des comme Thomas Sauvage utilisent parfois des années dans leur texte, il faut donc vérifier que l'on a bien des années qui correspondent à celles de la création au sens large de la pièce.

- Deux centenaires de Corneille qui propose plusieurs dates, attention de bien prendre la bonne (1629) :

"fut jouée en 1629 ; la Mort de Mustapha en 1630, et le Cid, comme chacun fait, ne parût qu’en 1636. S’il est donc vrai que la première Pièce régulière mérite à son Auteur le titre de Créateur du Théâtre Français ; c’est à Mairet que ce titre appartient, et c’est sans fondement qu’on le donne au Grand-Corneille."

### Problème standardisation TD :

2 standard émergent dans TD. Si certains fichiers mettent leurs corps de texte avec des lignes <p>

