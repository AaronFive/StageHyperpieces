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
Son année d'édition (type="print")
Et l'année de sa première représentation de la ppièce (type="premiere").

Afin de dater une oeuvre avec une date unique, on lui associe donc une année normalisée en fonction de certains critères :

Par défaut, on prend l'année la plus chronologiquement ancienne entre celle d'édition et de première représentation.

Cependant, si la pièce a été écrite il y a plus de 10 avant l'année en question, c'est l'année d'écriture de la pièce qui sera gardée, afin de garder une certaine cohérence par rapport au contexte de l'époque où la pièce a été rédigée.

En cas d'absence de certaines données, on ne fait la comparaison qu'avec les années restantes.

## Théâtre Documentation

Ici, l'affaire est moins évidente. Les pièces ne sont gardées que sous format html, il faut donc les convertir en xml en essayant de retrouver les données qui nous intéressent. L'année n'est typiquement pas conservée en tant que méta-donnée, il faut donc la trouver selon certains patterns.

La plupart du temps, l'année de première représentation est écrite sous la forme :

"Représentée ..., le JJ [mois] AAAA."

Parfois, nous avons simplement une ligne avec écrit :

AAAA.

On part alors du principe que c'est l'année d'édition.