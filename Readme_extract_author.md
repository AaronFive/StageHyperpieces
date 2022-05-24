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

## Théâtre Classique

Dracor puisant quasiment toutes ses sources via Théâtre Classique, il est donc très probable que ces données soient assez facilement trouvables.

En effet, il suffit de récupérer les fichiers xml dans le document, voire le csv. Cependant, on trouve les données un peu différemment. Le header ne nous propose que la décennie durant laquelle l'oeuvre a été imprimée :

Exemple (LES ÉTRENNES DE L'AMITIÉ, DE L'AMOUR ET DE LA NATURE, Archambaud) :

```xml
<teiHeader>
    <fileDesc>
        <titleStmt>
            <title>
            LES ÉTRENNES DE L'AMITIÉ, DE L'AMOUR ET DE LA NATURE
            </title>
            <author born="1742" born_location="Paris" death="1812" death_location="Paris" academie="">ARCHAMBAULT, Louis-François</author>
        </titleStmt>
        <publicationStmt>
            <p>
            publié par Paul FIEVRE juillet 2014, revu octobre 2016
            </p>
            <idno>ARCHAMBAULT_ETRENNES</idno>
            <rights>Creative Commons BY-NC-SA</rights>
        </publicationStmt>
        <SourceDesc>
            <genre>Comédie</genre>
            <inspiration>fantaisie</inspiration>
            <structure>Un acte</structure>
            <type>vers</type>
            <periode>1781-1790</periode>
            <taille>750-1000</taille>
            <permalien/>
        </SourceDesc>
    </fileDesc>
</teiHeader>
```

On trouvera dans la balise periode de SourceDesc la décennie qui nous intéresse, ici 1781-1790

Les années exactes se trouvent un peu après :

```xml
<text>
<front>
<docTitle>
...
</docTitle>
<docDate value="1783">M. DCC. LXXXIII.</docDate>
<docAuthor id="ARCHAMBAULT, Louis-François" bio="achambault">Par M. DORVIGNY.</docAuthor>
<docImprint>
    <approbation id="">
    <head/>
    <p>Lu et approuvé. A Paris, ce 13 Juin 1783. SUARD.</p>
    <p>
    Vu l'Approbation, permis d'imprimer, À Paris, ce 14 Juin 1783. LE NOIR.
    </p>
    </approbation>
    <acheveImprime id=""/>
    <printer id="CAILLEAU">
    A PARIS, Chez CAILLEAU, Imprimeur-Libraire, rue Galande, vis-à-vis de la rue du Fouarre.
    </printer>
</docImprint>
<performance>
<premiere date="1780-01-01" location="Comédie française">
Représentée, pour la première fois, à Paris, sur le Théâtre de la Comédie Française, le premier Janvier 1780. Et à la Cour, devant LEURS MAJESTÉS, le 4 du même mois.
</premiere>
</performance>
...
</front>
</text>
```






## Théâtre Documentation

Ici, l'affaire est moins évidente. Les pièces ne sont gardées que sous format html, il faut donc les convertir en xml en essayant de retrouver les données qui nous intéressent. L'année n'est typiquement pas conservée en tant que méta-donnée, il faut donc la trouver selon certains patterns.

Pour ce qui est des balises, il faudra rechercher les balises <p> (balise dans lesquelles on met un paragraphe en HTML). Ensuite, il faut chercher en fonction des syntaxes.

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

2 standard émergent dans TD. Si certains fichiers mettent leurs corps de texte avec des lignes <p>, d'autres rajoutent en plus de celà ce genre de balisage :

'<p align="center" style="text-align:center"><b><i><span style="letter-spacing:-.3pt">"[Texte]"</span></i></b></p>' 

Qui centre et espace le texte à même le code HTML.

En conclusion, l'année n'est pas toujours facilement récupérable dans Théâtre Documentation. Parfois, il nous manque l'information du jour exact de la représentation d'une pièce. Parfois, nous n'avons carrément pas l'information du tout. Sans compter les quelques fautes que nous pouvons rencontrer. 

## 

