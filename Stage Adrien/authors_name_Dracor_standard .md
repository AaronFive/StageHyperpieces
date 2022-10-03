# Standard for author in Dracor XML_TEI

## Introduction :

Toutes les informations à propos du ou des auteurs d'une pièce de théâtre dans le fichier XML-TEI qui vous intéresse se trouvent dans cette suite de balises (dans les [...]):

```xml
<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="fre">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                ...
                <author>
      [...]
    </author>
```

Dans le cas où la pièce a été écrite à plusieurs, il y aura donc autant de balise author que d'auteurs.

Dans le cadre d'un auteur anonyme, vous trouverez simplement un texte à la balise <author> :

```xml
<author>[anonyme]</author>
```

Sinon, c'est que nous connaissons le nom de l'auteur, et c'est ce qui va nous intéresser.

Nous allons donc retrouver tous les noms (communs, propres, surnoms, titres etc) de l'auteur dans la balise <persName> :

```xml
<author>
...
      <persName>
        [...]
      </persName>
</author>
```


Dans certains cas, en général les différentes manières de le nommer, nous aurons plusieurs de ces balises.

A ce jour, il arrive que certains noms ne soient pas bien standardisés dans Dracor, nous verrons dans ce dossier les quelques exceptions possibles. 

A ce jour, l'une des pièces possède un nom d'auteur écrit sans balise <persName>, à même le texte de author :

```xml
<author key="isni:0000000078237682">Nicolas-Médard AUDINOT (1732-1801)</author>
```

dans la pièce audinot-dorothee.xml.


## persName :

Nous allons ici nous intéresser à ce que l'on peut trouver dans persName. La plupart du temps, vous y trouverez tous les types de noms de l'auteur. Pour quelqu'un possédant simplement un prénom et un nom de famille, nous obtiendrons par exemple :

```xml

<author>
      <idno type="isni">0000000122775219</idno>
      <idno type="wikidata">Q276675</idno>
      <persName>
        <forename>Jean-Joseph</forename>
        <surname>Vadé</surname>
      </persName>
    </author><!--VADÉ, Jean-Joseph-->
```

La balise <forename> contenant le prénom, la balise <surname> le nom de famille.

Certains ont des noms un peu plus compliqués... :

```xml
<author>
      <persName>
        <roleName>Abbé</roleName>
        <forename>Léonor</forename>
        <forename>Jean</forename>
        <forename>Christine</forename>
        <surname>Soulas d'Allainval</surname>
      </persName>
    </author><!--SOULAS d'ALLAINVAL, Abbé Léonor Jean Christine-->
```

- Un <rolename> comportant le titre de l'auteur (Abbé en l'occurence).
- Trois <forename> comportant les trois prénoms (!) de l'auteur 
- Et un <surname>, toujours pour le nom de famille.

Le très célèbre Jean de la Fontaine lui-même est un petit cas particulier :

```xml
<persName>
        <forename>Jean</forename>
        <nameLink>de</nameLink>
        <surname>La Fontaine</surname>
      </persName>
    </author><!--LA FONTAINE, Jean de-->
```

Les noms à particule seront le plus souvent représentés en utilisant la balise <nameLink> pour illustrer la particule "de" de son nom.

Enfin, certains cas ne comportent qu'un simple nom dans le texte de persName directement :

```xml
<author>
      <persName>Vallée</persName>
    </author><!--VALLÉE-->
``` 

## Les complications :

- Comme dit précédemment, il peut y avoir plusieurs persName. Celà peut notamment être la consquence du fait que l'auteur peut être appelé de différentes manières :

```xml
<author>
      ...
      <persName type="pen">Desfontaines</persName>
      <persName>
        <forename>François-Georges</forename>
        <surname>Fouques Deshayes</surname>
      </persName>
    </author>
```

Ici, on peut voir que l'auteur François-Georges Fouques Deshayes est plus connu sous le nom "Desfontaines". Celui-ci et représenté avec l'attribut "type" de persName prenant commme valeur "pen".
Voltaire (François-Marie Arouet) ou Molière (Jean-Baptiste Poquelin) sont deux exemples bien connus et respectent également cette règle.

- Il existe cependant d'autres cas particuliers, voici un exemple avec deux standards que l'on peut retrouver :

```xml
<author>
      ...
      <persName xml:space="preserve">
        <forename>Françoise</forename>
        <nameLink>d'</nameLink><surname>Aubigné</surname>,
        <roleName>marquise</roleName>
        <nameLink>de</nameLink>
        <surname sort="1">Maintenon</surname>
      </persName>
    </author><!--MAINTENON Françoise d'Aubigné, marquise de, (1635-1719)-->
```

Nous voyons ici que Mme de Maintenon, possédant un titre, est malgré tout appelée par son nom entier. Est utilisé l'attribut "xml:space" avec la valeur "preserve" dans la balise persName, ce qui garde les espaces potentiellement placés autour du texte, ici on ne remplace pas le nom initial de la marquise par son titre, elle est appelée par son nom entier.

Mais à quoi sert ce fameux "sort=1" dans la deuxième balise surname ? Et bien justement, cette balise est utilisée dans le cas où il y a deux surname. Aubigné étant son nom de famille et Maintenon le nom relié à son titre de marquise. On utilise alors cet attribut sur le deuxième surname pour lui donner les attributs du surname principal.

C'est de cette manière qu'on obtient ce nom à la fin :

```xml
<!--MAINTENON Françoise d'Aubigné, marquise de, (1635-1719)-->
```

Le MAINTENON est passé devant grâce au sort=1, et le "preserve" garde un espace entre MAINTENON et le reste du nom.

- Un cas particulier très particulier (une seule occurence) :

```xml
<persName>
        <forename>Joseph-Alexandre</forename>
        <forename>Pierre</forename>
        <roleName type="nobility">vicomte</roleName>
        <nameLink>de</nameLink>
        <surname>Ségur</surname>
      </persName>
```

Certains rôles possèdent comme particularité d'avoir le type "nobility", une pratique non standardisée (Mme de Maintenon était marquise) qui se retrouve sur ce roleName.

- Dernière singularité :

```xml
<author>
      <persName>
        <forename>Bernard</forename>
        <surname>Le Bouyer</surname>
        <nameLink>de</nameLink>
        <surname sort="1">Fontenelle</surname>
      </persName>
      <persName>
        <forename>Bernard</forename>
        <surname>Le Bouvier</surname>
        <nameLink>de</nameLink>
        <surname>Fontenelle</surname>
      </persName>
    </author><!--FONTENELLE, Bernard le Bouvier ou le Bouyer de-->
```

Bernard le Bouyer de Fontenelle, aussi appelé Bernard le Bouvier de Fontenelle. Le seul cas d'une personne ayant deux persName car il avait deux appelations. Celà peut donner des résultats alambiqués lors de l'extraction du nom de l'auteur...
