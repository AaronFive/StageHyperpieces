# Play Corpus

Code written for my M1 Internship with Philippe Gambette and Aaron Boussidan.

## What do you have ?

In the directory "Conversion and scraping", you have all the Python scripts you can use for download and conversion in XML-TEI.
You can also find a script to generate DOT files with XML files. It's useful to have a tree view of your plays.

## How to use it ?

### Download and Conversion

For each source, we developed differents scripts, one to download the datas from the source, the others to make the conversion in XML for DraCor or BibDramatique.

You can find all the scripts in the folder "Conversion and scraping".
The documentation is in the folder "Documentation.

For a file XXXXX.py, you have the documentation file XXXXX.html.

To generate a corpus of files with one source, start to go to the current repertory :

    ```bash
    cd "Conversion and scraping"
    ```

Then, you have to download the datas you want :

    ```bash
    python3 downloadXXXXX.py
    ```

Your files are at the folder corpusXXXXX, or "Corpus XXXXX" in the main menu (not in Conversion and scraping).

You just have to convert the downloaded files with :

    ```bash
    python3 convertXXXXXtoBibdramatique.py
    ```
    or

    ```bash
    python3 convertXXXXXtoDracor.py
    ```

to convert in "Bibliothèque Dramatique" or DraCor standard.

#### DraCor

NB : DraCor is already at the good format, we just have a file to download the new files from DraCor.

- Download : downloadDracor.py

#### Théâtre Documentation

- Download : downloadTheatreDocumentation.py
- Conversion for BibDramatique : convertTheatreDocToBibdramatique.py
- Conversion for DraCor : convertTheatreDocToDracor.py

### Théâtre Classique

NB : Already the good format for BibDramatique.

- Download : downloadTheatreClassique.py

### EMOTHE

- Conversion for BibDramatique : convertEmotheToBibdramatique.py

### Classique Garnier

- Conversion for BibDramatique : convertClassiquesgarnierToBibdramatique.py

### Tree View

We have also two scripts for tree representation of HTML or XML files.

For each source, we developed differents scripts, one to download the datas from the source, the others to make the conversion in XML for DraCor or BibDramatique.

You can find all the scripts in the folder "Conversion and scraping".
The documentation is in the folder "Documentation.

For a file XXXXX.py, you have the documentation file XXXXX.html.

#### displayXMLInTree.py

You can select the folder youwant (default DraCor Corpus) and generate in another folder all the trees. The folder must contains only XML files.

For corpusXXXXX -> treesXXXXX

Example : corpusDracor -> treesDracor

usage: Display Tree from HTML. [-h] [-a N N] [-c [CLEAN]] [-d [DIRECTORY]] [-i [N]] [-p] [-t [TREE]] [-v]

optional arguments:
-h, --help show this help message and exit
-a N N, --acts N N Selects only the files respecting a number minimum and maximum of an act.
-c [CLEAN], --clean [CLEAN]
Cleans up given folders
-d [DIRECTORY], --directory [DIRECTORY]
Selects a folder to run the program (by default, Dracor).
-i [N], --intersection [N]
Generates the intersection of the structure of each XML file as a dot file.
-p, --precision Explains act, scene, and line numbers, as well as characters.
-t [TREE], --tree [TREE]
Generates for each selected XML file its structure in a dot file, in the form of a tree.
-v, --verbose Generates for each selected XML file its structure in a text file

#### displayHTMLInTree.py

You can select the folder you want (default DraCor Corpus) and generate in another folder all the trees. The folder must contains only HTML files.

For now, only the corpus from TD are available in HTML.

For corpusXXXXX -> treesXXXXX

Example : corpusDracor -> treesDracor

usage: Display Tree from HTML. [-h] [-a N N] [-c [CLEAN]] [-d [DIRECTORY]] [-i [N]] [-p] [-t [TREE]] [-v]

optional arguments:
-h, --help show this help message and exit
-a N N, --acts N N Selects only the files respecting a number minimum and maximum of an act.
-c [CLEAN], --clean [CLEAN]
Cleans up given folders
-d [DIRECTORY], --directory [DIRECTORY]
Selects a folder to run the program (by default, Dracor).
-i [N], --intersection [N]
Generates the intersection of the structure of each XML file as a dot file.
-p, --precision Explains act, scene, and line numbers, as well as characters.
-t [TREE], --tree [TREE]
Generates for each selected XML file its structure in a dot file, in the form of a tree.
-v, --verbose Generates for each selected XML file its structure in a text file
riads@riads-HP-ZBook-15:~/StageHyperpieces/Conversion and scraping$ python3 displayHTMLInTree.py -h
usage: Display Tree from HTML. [-h] [-a N N] [-c [CLEAN]] [-d [DIRECTORY]] [-i [N]] [-p] [-t [TREE]] [-v]

optional arguments:
-h, --help show this help message and exit
-a N N, --acts N N Selects only the files respecting a number minimum and maximum of an act.
-c [CLEAN], --clean [CLEAN]
Cleans up given folders
-d [DIRECTORY], --directory [DIRECTORY]
Selects a folder to run the program (by default, TD).
-i [N], --intersection [N]
Generates the intersection of the structure of each XML file as a dot file.
-p, --precision Explains act, scene, and line numbers, as well as characters.
-t [TREE], --tree [TREE]
Generates for each selected html file its structure in a dot file, in the form of a tree.
-v, --verbose Generates for each selected html file its structure in a text file
