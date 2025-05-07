#!/usr/sfw/bin/python
# -*- coding: utf-8 -*-
import csv
import datetime
import os
import pickle
import re
import sys
from datetime import date
from os import walk, pardir
from os.path import abspath, dirname, join, basename, exists

import TheatredocToDracorWriting as writing
import editdistance as editdistance

"""
    theatredocToBibdramatique, a script to automatically convert 
    HTML theater plays from théâtre-documentation.com
    to XML-TEI as on http://bibdramatique.huma-num.fr/
    Copyright (C) 2021 Philippe Gambette

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU Lesser Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
# The folder containing this script must contain a subfolder named corpusTD
# containing plays downloaded from théâtre-documentation.com

# TODO : detection of privilege, acheve_imprime, printer_text, performance, signed. For now these are never done
# TODO: Add writing of "actStageIndication"
# Get the current folder
folder = abspath(dirname(sys.argv[0]))
root_folder = abspath(join(folder, pardir))
html_folder = abspath(join(root_folder, "cleanHTML_TD_normalized"))
Dracor_Folder = abspath(join(root_folder, "corpusTD_v2"))
clean_Dracor_Folder = abspath(join(root_folder, "corpusTD_cast_ok"))

if not exists(Dracor_Folder):
    os.system("mkdir {0}".format(Dracor_Folder))

mois = {
    'janvier': '01',
    'fevrier': '02',
    'mars': '03',
    'avril': '04',
    'mai': '05',
    'juin': '06',
    'juillet': '07',
    'aout': '08',
    'septembre': '09',
    'octobre': '10',
    'novembre': '11',
    'decembre': '12',
}
genres = ["tragedie", "comedie", "tragicomedie", "tragi-comedie", "farce", "vaudeville", "proverbe", "pastorale",
          "comedie musicale", "dialogue", "monologue"]
good_genre = {"tragedie": "Tragédie", "comedie": "Comédie", "tragicomedie": "Tragi-Comédie", "farce": "Farce",
              "vaudeville": "Vaudeville", "proverbe": "Proverbe", "pastorale": "Pastorale", "dialogue": "Dialogue",
              "comedie musicale": "Comédie Musicale", "dialogue": "Dialogue", "monologue": "Monologue"}


# DEBUG
def log(name, value):
    print(f'{name} : {value} ')


def notify_file(file):
    """Notify the user with the conversion of the input file.

    Args:
        file (str): Name of the file to convert.
    """
    print("Converting file " + file)
    # date_file.writelines(basename(file).replace(".html", '') + "\t")


# UTILS
def format_date_AAAAMMJJ(res):
    day = res[0].replace('<sup>er</sup>', '')
    if len(day) == 1:
        day = '0' + day
    return '-'.join(
        [res[2].replace('l', '1').replace('|', '1'),
         mois[res[1].lower().replace('é', 'e').replace('août', 'aout').replace('levrier', 'fevrier').replace('fevier',
                                                                                                             'fevrier')],
         day.replace('l', '1').replace('|', '1').replace('premier', '01')
         ])


def format_date_AAAAMM(res):
    return '-'.join(
        [res[1],
         mois[res[0].lower().replace('é', 'e').replace('août', 'aout').replace('levrier', 'fevrier').replace('fevier',
                                                                                                             'fevrier')]
         ])


def format_date_to_comparable(groups):
    """Convert a tuple of (day, month, year) to a comparable format YYYYMMDD or just year."""
    months = {
        'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04', 'mai': '05', 'juin': '06',
        'juillet': '07', 'août': '08', 'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
    }

    # Handle full dates with day, month, year
    if len(groups) == 3:
        day = groups[0].replace("1<sup>er</sup>", "01").replace("premier", "01").zfill(
            2)  # Replace "1er"/"premier" with "01"
        month = months.get(groups[1].lower(), '00')  # Convert month name to number
        year = groups[2]
        return f"{year}{month}{day}"  # Format as YYYYMMDD

    # Handle year-only dates
    elif len(groups) == 1:
        return groups[0]  # Return the year as it is

    return None


def get_oldest_date(dates):
    """Return the oldest date from a list of datetime objects."""
    return min(dates) if dates else None


def format_date_from_matches(matches):
    """Formats a date from a tuple of (day, month, year) or just (year)."""
    if len(matches) == 3:
        return format_date_AAAAMMJJ(matches)
    elif len(matches) == 1:
        return matches[0]
    return "[vide]"


def remove_html_tags_and_content(s):
    s = s.replace(u'\xa0', u' ')
    return re.sub('<[^>]+>', '', s)


def remove_html_tags(s):
    s = s.replace(u'\xa0', u' ')
    s = re.sub('<[^>]+>|</.*>', '', s)
    return s


def min_dict(d):
    """Returns (key_min, val_min) such that val_min is minimal among values of the dictionnary"""
    val_min = 100
    key_min = None
    for x in d:
        if d[x] < val_min:
            val_min = d[x]
            key_min = x
    return key_min, val_min


def normalize_line(line):
    # l = re.sub(r'</?span[^>]*>', '', line)
    l = line.replace("\xa0", ' ').replace('<a href="#_ftn1" name="_ftnref1" title="" id="_ftnref1">[1]</a>', '')
    return l.strip('\n')


def standard_line(playText):
    return list(map(normalize_line, playText))


def clean_scene_name(s):
    s = s.replace('</span>', '')
    s = re.sub('<span.*>', '', s)
    s = re.sub('\[\d{1,4}]', '', s)
    s = re.sub('<strong>|</strong>', '', s)
    s = remove_html_tags(s)
    if not s:
        return s
    if s[0] == ' ':
        s = s[1:]
    if s[-1] == ",":
        s = s[:-1]
    if s[-1] == ' ':
        s = s[:-1]
    if s in ['Notes', 'Variantes', 'PDF'] or 'PDF' in s:
        return ''
    s = s.strip()
    return s


def is_list_of_scenes(lst):
    """Checks if lst is a list of scenes of the form [['Acte 1,[Scène première, Scène II,...], [Acte 2, [Scène première, Scène II,...],...]
    Typically used on counters["sceneList]"""
    res = len(lst) > 0
    for x in lst:
        res = res and len(x) == 2 and x[1] and all(['Scène' in s for s in x[1]])
    return res


def normalize_character_name(s):
    if s and s[-1] == ".":
        s = s[:-1]
    clean_character_name = s.lower().replace("*", "")
    clean_character_name = re.sub("[\[\]\)\(]", "", clean_character_name)
    clean_character_name = remove_html_tags_and_content(clean_character_name)
    clean_character_name = re.sub("\A | \Z", "", clean_character_name)
    clean_character_name = re.sub(" +", "-", clean_character_name)
    return clean_character_name.strip()


# METADATA COLLECTION
def extract_sources(allPlays, fileSources):
    """Extract sources from each play

    Args:
        allPlays (TextIOWrapper): File with all the plays.
        fileSources (set): Empty set to fill with all sources.

    Returns:
        TextIOWrapper: Return allPlays.
    """
    for playLine in allPlays:
        res = re.search("([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\n", playLine)
        if res:
            fileSources[res.group(1)] = res.group(2)
    return allPlays


def get_source(fileSources, fileName):
    """Get the source from a file.

    Args:
        fileSources (dict): Dictionnary with files' name in key and their sources in values.
        fileName (str): Name of the file we want the source.

    Returns:
        str : The name of the source of the input file.
    """
    if fileName in fileSources:
        return fileSources[fileName]
    return ""


def get_title_and_author(line):
    """Extract the title and the author from a play

    Args:
        line (str): Line of the play with the title and the author's name.

    Returns:
        tuple: Tuple of strings with the title, the forename and the surname of the author.
    """
    title = ""
    author = ""
    persNames = ""
    forename = ""
    surname = ""

    res = re.search("<title>(.*) | théâtre-documentation.com</title>", line.replace(")", ""))
    if res:
        title = res.group(1)

        res2 = re.search(r"^(.*) \((.*)$", title)
        if res2:
            title = res2.group(1)
            author = res2.group(2).strip('| ')

            persNames = author.split(' ')
            forename = list(filter(lambda s: not s.isupper(), persNames))
            surname = list(filter(lambda s: s.isupper(), persNames))
    return title, forename, surname


def get_genre_versification_acts_number(playText):
    """Get the type of the play, and if it's in prose or in verses.

    Args:
        playText (TextIOWrapper): Text Contents of a play.

    Returns:
        _tuple_: Tuple of strings. The genre of the play, and the type of writing (verses or prose). Return [indéfini] if it's undefined.
    """
    res_genre, vers_prose, act_number = '[indéfini]', '[indéfini]', -1
    for l in standard_line(playText):
        res = re.search('<p>(.*)</p>', l)
        if res:
            content = res.group(1).lower().replace('é', 'e')
            if content == ' ':
                break
            for genre in genres:
                if genre in content:
                    res_genre = good_genre[genre]
                    break
            if 'prose' in content:
                vers_prose = 'prose'
            elif 'vers' in content:
                vers_prose = 'vers'
            act_number_string = re.search(r'(un|deux|trois|quatre|cinq|six|sept) actes?', content)
            if act_number_string:
                act_number = act_number_string.group(1)
                numbers_dict = {'un': 1, 'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5, 'six': 6, 'sept': 7}
                act_number = numbers_dict[act_number]
    return res_genre, vers_prose, act_number


# Predefined regex patterns for reuse
WRITING_DATE_REGEX = re.compile("<p>.*[ÉéEe]crit en ([0-9]+).* et [op]ublié.* en ([0-9]+).*</p>")
WRITING_DATE_ALT_REGEX = re.compile("<p>.*[ÉéEe]crit[e]? (.*)</p>")
PRINTING_DATE_REGEX = re.compile("<p>([0-9]+).*</p>")
PRINTING_ALT_DATE_REGEX = re.compile("<p>Imprimée en ([0-9]+).*</p>")
PREMIERE_DATE_REGEX = re.compile(r"([0-9]{1,2}|1<sup>er</sup>|premier)\s+([^ ]+)\s+([0-9]{2,4})")
PREMIERE_ALT_DATE_REGEX = re.compile(r"en\s+([0-9]{4})")


def extract_written_date(line):
    """Extract writing date from a line."""
    match = WRITING_DATE_REGEX.search(line)
    if match:
        return match.groups(), line.replace("<p>", "").replace("</p>", "")

    match = WRITING_DATE_ALT_REGEX.search(line)
    if match:
        date_info = match.group(1)
        specific_date_match = re.search(".*le ([0-9]+) ([^ ]+) ([0-9]+).*", date_info)
        year_match = re.search(".*en ([0-9]+).*", date_info)
        if specific_date_match:
            return (format_date_AAAAMMJJ(specific_date_match.groups()),), line
        elif year_match:
            return (year_match.group(1),), line
    return None, None


def extract_printing_date(line):
    """Extract printing date from a line."""
    match = PRINTING_DATE_REGEX.search(line)
    if match:
        return match.group(1), line.replace("<p>", "").replace("</p>", "")

    match = PRINTING_ALT_DATE_REGEX.search(line)
    if match:
        return match.group(1), line.replace("<p>", "").replace("</p>", "")

    return None, None


def extract_premiere_date(line):
    """Extract premiere date from a line, returning the oldest date if more than one is found."""

    found_dates = []

    # Look for full premiere dates (day, month, year)
    for match in PREMIERE_DATE_REGEX.finditer(line):
        comparable_date = format_date_to_comparable(match.groups())
        if comparable_date:
            found_dates.append((comparable_date, line.replace("<p>", "").replace("</p>", "")))

    # Look for alternative year-only premiere dates
    for match in PREMIERE_ALT_DATE_REGEX.finditer(line):
        comparable_date = format_date_to_comparable((match.group(1),))
        if comparable_date:
            found_dates.append((comparable_date, line.replace("<p>", "").replace("</p>", "")))

    # Return the oldest date (if found) along with its context line
    if found_dates:
        # Sort dates and select the oldest (lexicographically, 'YYYYMMDD' will sort correctly)
        oldest_date, context_line = min(found_dates)
        # Format the date back to the correct output format (YYYY-MM-DD or just year)
        formatted_date = (
            f"{oldest_date[:4]}-{oldest_date[4:6]}-{oldest_date[6:]}"
            if len(oldest_date) == 8
            else oldest_date
        )
        return formatted_date, context_line

    return None, None


def get_dates(playText):
    """Extract the dates of writing, printing, and first performance from the play, along with their context lines."""

    # Initialize result variables with default values
    date_written, line_written = "[vide]", "[vide]"
    date_print, line_print = "[vide]", "[vide]"
    date_premiere, line_premiere = "[vide]", "[vide]"

    is_written, is_print, is_premiere = False, False, False

    # Iterate through each line of the play
    for line in standard_line(playText):

        # Stop searching if characters section is reached
        if re.search(".*<strong><em>Personnages.*</em></strong>.*", line) or re.search(
                '<p align="center" style="text-align:center"><b><i>Personnages.*</span></i></b></p>', line):
            break

        # Extract writing date
        if not is_written:
            date_result, line_result = extract_written_date(line)
            if date_result:
                date_written, line_written = date_result[0], line_result
                is_written = True

        # Extract printing date
        if not is_print:
            date_result, line_result = extract_printing_date(line)
            if date_result:
                date_print, line_print = date_result, line_result
                is_print = True

        # Extract premiere date
        if not is_premiere:
            date_result, line_result = extract_premiere_date(line)
            if date_result:
                date_premiere, line_premiere = date_result, line_result
                is_premiere = True

        # Stop if all dates are found
        if is_written and is_print and is_premiere:
            break

    # Clean the dates from HTML tags and content
    all_dates = [date_written, date_print, date_premiere, line_written, line_print, line_premiere]
    all_dates = [remove_html_tags_and_content(date) for date in all_dates]

    return tuple(all_dates)


def find_summary(line, ul):
    """Detect if a line of the file is the start of the summary, with the tag <ul> of a HTML file, if it exists.

    Args:
        line (str) : The line where we try to detect the start of the summary.
        ul (int) : The number of ul tags found in the entire file.

    Returns:
        bool: True only if the line is the tag "<ul>".
    """
    if line == "<ul>":
        ul += 1
        return True
    return False


def extract_from_summary(line, ul):
    """Extract the datas from summary to count the number of acts and find an eventual dedicace in the play.

    Args:
        line (str) : The line where we try to detect the start of the summary.
        ul (int) : The number of ul tags found in the entire file.

    Returns:
        Match[str]: Return the datas extracted by the regex search function, None if it found nothing.
    """
    if line == "<ul>":
        ul += 1
        return True
    if line == "</ul>":
        ul -= 1
        return ul
    res = re.search("<li class=\"toc-level-([0-9])\"><a href=\"(.*)\"><strong>(.*)</strong></a></li>", line)
    if res:
        level = res.group(1)
        text = res.group(3)
        if level == 1:
            if "ACTE" in text:
                counters["actsInPlay"] += 1
            elif text != "PRÉFACE" and text != "PDF":
                counters["dedicace"] = True
    return res


def find_dedicace(line):
    """Extract the content of a dedicace in a play from a line if it has it.

    Args:
        line (str): The line where we're looking for a dedicace.

    Returns:
        str: The content of the dedicace if it exists in the line, None then.
    """
    res = re.search('<h1 class="rtecenter" style="color:#cc0066;" id=".*"><strong>(.*)</strong></h1>', line)
    if res:
        return res.group(1)
    return None


# METADATA WRITING

## Collecting body of play
def try_saving_lines(outputFile, line):
    """Look if the read line is in <p></p> tags. If yes, authorize the copy of the lines contents of the HTML file in the XML output file.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        line (str): Line to read.

    Returns:
        bool: True if the line is in <p></p> tags.
    """
    res = re.search("<p>(.*)</p>", line)
    if res:
        outputFile.writelines("<p>" + res.group(1) + "</p>\n")
    return bool(res)


def start_character_block(line, characterBlock):
    """Check if we have to save the next lines as characters names of a play.

    Args:
        line (str): line to read
        characterBlock (bool): Actual situation of saving of characters.

    Returns:
        bool: True if a line with "Personnage" is written in the line, or written before.
    """
    return characterBlock or re.search("<strong><em>Personnages</em></strong>", line) or re.search(
        '<p align="center" style="text-align:center"><b><i>(<span style="letter-spacing:-\.3pt">)?Personnages(</span>)?</i></b></p>',
        line)


def end_character_block(characterBlock, line):
    """Detect if all the characters of a play were saved.

    Args:
        characterBlock (bool): Flag to know if lines are still composed by characters.
        line (str): line to read.

    Returns:
        tuple: the boolean of all the characters, but also the actual line.
    """
    if characterBlock:
        end_block = re.search("<h[1,2]", line)
        if end_block:
            characterBlock = False
        else:
            character_declaration = re.search("<p>(.*)</p>", line)
            if character_declaration:
                name = character_declaration.group(1)
                if len(name) == 1:
                    if counters["characterIDList"]:
                        characterBlock = False
                        # print("Character list: " + str(counters["characterIDList"]))
                    return characterBlock, None
                character = name
                name_and_role = re.search("([^,]+)(,.*)", character)
                if name_and_role:
                    character = remove_html_tags_and_content(name_and_role.group(1))
                    role = remove_html_tags_and_content(name_and_role.group(2))
                else:
                    character = remove_html_tags_and_content(character)
                    role = ""
                if len(character) > 2 and character != "\xa0":
                    counters["characterFullNameList"].append(character)
                    clean_character_name = normalize_character_name(character)
                    counters["characterIDList"].append(clean_character_name)
                    counters["roleList"].append(role)
    return characterBlock, line


def find_scene_list(line, sceneList, inSceneList):
    """Finds the declaration of scenes at the beginning of the play and returns the list of scenes
        Args:
        line (str): line to read.
        sceneList(list): The list of scenes of the play. Elements are of the form ['Act Name',['Scene 1',...'Scene N']]
        inSceneList(bool): flag to know if we are reading a declaration of scenes

    Returns:
        list: Updated sceneList
        bool: Updated inSceneList"""
    # When inSceneList is true, we are currently reading the list of scenes. It is set at True when the "<div
    # class='toc-list'>" is found, and then fed forward sceneList is the list of scenes currently constructed
    if line is None and not inSceneList:
        return [], False
    elif line.strip() in [None, ''] and inSceneList:
        return sceneList, True
    if not inSceneList and line == "<div class='toc-list'>":
        return sceneList, True
    elif inSceneList and line in ['<ul>', '</ul>', '</li>']:
        return sceneList, True
    regex_act = r"<li class=\"toc-level-1\"><a href=[^>]*>(?:<strong>)?(?P<actename>[^<]+)(?:</strong>)?</a>"
    regex_scene = "<li class=\"toc-level-2\"><a href=.*><strong>(.+)</strong></a></li>|<li class=\"toc-level-1\"><a " \
                  "href=.*>(.*Scène.*)</a>"
    regex_scene_type = "[Jj]ournée|[Ss]cène|[Tt]ableau|[Ee]ntrée"
    regex_preface = "Préface|PREFACE|PRÉFACE"
    regex_dedicace = r"\A *À|\A *AU|.* À M\. (.*)"
    act_line = re.search(regex_act, line)
    scene_line = re.search(regex_scene, line)
    if act_line and act_line.group('actename') and inSceneList and "Scène" not in act_line.group(1):
        act_line = act_line.group('actename')
        act_line = clean_scene_name(act_line)
        dedicace_found = re.search(regex_dedicace, act_line)
        if dedicace_found:
            counters["dedicaceFound"] = True
            metadata["dedicaceHeader"] = act_line
            metadata["dedicaceSalute"] = dedicace_found.group(1)
        elif re.search(regex_preface, act_line):
            counters["prefaceFound"] = True
            metadata["prefaceHeader"] = act_line
        elif act_line:
            if not counters["noActPlay"]:
                sceneList.append([act_line, []])
            else:
                print("WARNING : ACT FOUND IN A NO ACT PLAY ? Treating it as a scene")
                sceneList.append(act_line)
        return sceneList, True
        # TODO:  Privilege
        # <div type="preface">
        # <head>Préface</head>
        #		<div type="docImprint">
    # 	<div type="privilege">
    #             <head>EXTRAIT DU PRIVILÈGE DU ROI</head>
    # 		<p>Par Grâce et privilège du Roi, donné à Paris le 19 janvier 1660, signé par le Roi en son conseil, Mareschal, il est permis à Guillaume de Luynes, Marchand-Libraire de notre bonne ville de Paris de faire imprimer, vendre, et débiter les Précieuses ridicules fait par le sieur Molière, pendant cinq années et défenses sont faites à tous autres de l'imprimer, ni vendre d'autre édition de celle de l'exposant, à peine de deux mille livres d'amende, de touts dépens, dommage et intérêts, comme il est porté plus amplement par les dites lettres.</p>
    # 		<p>Et le dit Luynes a fait part du privilège ci-dessus à Charles de Cercy et Claude Barbin, marchands-libraires, pour en jouir suivant l'accord fait entre-eux.</p>
    # 	</div><!--@id="1660-01-19"-->
    # 	<div type="printer">
    #             <p>À PARIS, chez Guilaume de LUYNES, Libraire juré au Palais, dans la Salle des Merciers, à la Justice.</p>
    #         </div><!--@id="LUYNES"-->
    # 	<div type="acheveImprime">
    #             <p>Achevé d'imprimer pour la première fois le 29 janvier 1660. Les exemplaires ont été fournis.</p>
    #         </div><!--@id="1660-01-29"-->
    # </div>
    if scene_line and inSceneList:
        if scene_line.group(1):
            scene_line = clean_scene_name(scene_line.group(1))
        else:
            scene_line = clean_scene_name(scene_line.group(2))
        if re.search(regex_dedicace, scene_line):
            counters["dedicaceFound"] = True
            metadata["dedicaceHeader"] = scene_line
            return sceneList, True
        elif re.search(regex_preface, scene_line):
            counters["prefaceFound"] = True
            metadata["prefaceHeader"] = scene_line
            return sceneList, True
        if sceneList:
            if scene_line:
                sceneList[-1][1].append(scene_line)
            return sceneList, True
        elif re.search(regex_scene_type, scene_line):
            # This is handling the case of plays with no acts but some scenes. We create a virtual unique act containing all scenes.
            sceneList.append(["Acte unique", [scene_line]])
        else:
            pass  # This is probably also a dedicace, or a line we don't know how to categorize. Throwing it away for now
        return sceneList, True
    if sceneList and inSceneList:
        return sceneList, False
    return [], False


def find_begin_act(line, counters, playContent):
    """Try to find the beginning of an act in a play and convert it in the XML file associated if it find it.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        line (str): line to read.
        counters (dict): Dictionnary with all the counters of the script.

    Returns:
        tuple: the line (str) and the refreshed counter (dict).
    """
    # Checking to see if this is a h1 header
    act_header = re.search(".*<h1[^>]*>(.*)</h1>", line)
    if act_header:
        # Creating the list of potential act names : they can either be regular act names or more unusual ones,
        # If they are unusual, they should be declared in the list of scenes at the beginning of the play
        act_header_string = act_header.group(1)
        acts_type = 'ACTE|JOURNÉE|TABLEAU|PARTIE|PROLOGUE|Prologue|ÉPOQUE|Partie|Tableau'  # Regular act names
        if counters["sceneList"] and not counters["noActPlay"]:
            acts_type = '|'.join(
                [acts_type] + [clean_scene_name(x[0].replace('*', '')) for x in counters["sceneList"] if
                               x[0] not in acts_type])  # acts declared in the beginning
        act_header_type = re.search(acts_type, act_header_string)
        if act_header_type:
            # Found a new act!
            counters["actsInPlay"] += 1
            counters["scenesInAct"] = 0
            act = act_header.group(1).replace("<strong>", "").replace("</strong>", "")
            act_number = re.search("ACTE (.*)", act)
            if act_number:
                counters["actNb"] = act_number.group(1)
            else:
                counters["actNb"] = act.replace(" ", "-").lower()
            if counters["noActPlay"]:  # Treating it as if it were a scene
                playContent.append({"sceneNumber": None, "sceneName": act, "repliques": [], "speakers_text": None,
                                    "speakers_ids": None})
            else:
                playContent.append({"actNumber": None, "actName": act, "Scenes": [], "actStageIndications": None})
    return line, counters, playContent


def find_begin_scene(line, counters, playContent):
    """Try to find the beginning of a scene in a play and convert it in the XML file associated if it find it.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        line (str): line to read.
        counters (dict): Dictionnary with all the counters of the script.

    Returns:
        tuple: the line (str) and the refreshed counter (dict).
    """
    regex_scenes = "|".join(
        [".*<h2 .*<strong>(?P<h2strong>.*)</strong>.*</h2>", ".*<h2.*>(?P<h2normal>Scène.*)</h2>",
         "<h1.*>(?P<h1>.*Scène.*)</h1>"])
    res = re.search(regex_scenes, line)
    if res:
        # counters["characterLines"] = []
        # counters["repliquesinScene"] = 0
        # counters["charactersinScene"] = ""
        # if not (counters["scenelessPlayBeginningWritten"]):
        #     scene = "Scène 1"
        if res.group('h2strong') is not None:
            scene = res.group('h2strong')
        elif res.group('h2normal') is not None:
            scene = res.group('h2normal')
        else:
            scene = res.group('h1')
        scene_number = re.search("Scène (.*)", scene)
        if scene_number:
            scene_number = scene_number.group(1)
        else:
            scene_number = scene.replace(" ", "-").lower()
        counters["scenesInAct"] += 1
        new_scene = {"sceneName": scene, "sceneNumber": scene_number, "speakers_text": None, "speakers_ids": None,
                     "repliques": []}
        if playContent:
            if "actName" in playContent[-1]:  # We are inside an act
                playContent[-1]["Scenes"].append(new_scene)
            elif counters["noActPlay"]:  # This is a play with no act, only scenes
                playContent.append(new_scene)
            else:
                raise ValueError(f"Ill-formed playContent : {playContent}")
        else:
            # There are no acts or scenes yet : this is the first scene of a play with no scene
            if not counters["sceneList"] or type(counters["sceneList"][0] == str):
                counters["noActPlay"] = True
                playContent.append(new_scene)
            else:
                pass
                # TODO : this happens when we read text unexpectedly before the beginning of the act, but we know there should be an act
    return line, counters, playContent


def find_character(line, counters, playContent):
    """Find a character name in a line from a play text and stock it in the counters dict.

    Args:
        line (str): line to read in the play.
        counters (dict): Dictionnary with all the counters of the script.

    Returns:
        dict: The refreshed counter
    """
    # In theatre doc, character declarations can be of the form  :
    # <p align="center" style="text-align:center"><span style="font-size:10.0pt"><span style="letter-spacing:-.3pt">ROBERT.</span></span></p>
    # This regex captures this and simpler version (without spans)
    # res = re.search("<p align=.center[^>]+>(?:<[^>]*>)*([^<>]*)(?:<[^>]*>)*</p>", line)
    added_character = False
    res = re.search("<p align=.*center[^>]*>(.*)</p>", line)
    character_type = None
    character_name = None
    if res and res.group(1) != "\xa0" and "Personnages" not in res.group(1):
        character_name = res.group(1)
        special_characters = re.search("(TOUS|TOUTES|ENSEMBLE|CHOEUR|CHŒUR)", character_name)
        number_of_characters = len(character_name.split(","))  # Checking to see whether this is multiple characters
        # This is a declaration of characters occuring in the scene (TODO: or multiple characters speaking or special chars)
        # if special_characters:
        #     character_type = "Special"
        # elif number_of_characters >=2:
        #     character_type = "Multiple"
        # else:
        #     character_type = "Simple"
        if playContent:
            if "actName" in playContent[-1]:  # There's an act
                scenes = playContent[-1]["Scenes"]
            else:  # There's only scenes
                scenes = playContent
            if not scenes:  # We are not in a scene, so this is not a character name, but probably a stage indication, like a location
                playContent[-1]["actStageIndication"] = character_name
            else:
                current_scene = scenes[-1]
                if not (current_scene["repliques"]) or current_scene["repliques"][-1]["type"] != "Speaker":
                    current_scene["repliques"].append({"type": "Speaker", "content": character_name})
                    added_character = True
                else:
                    # When reading multiple succesives lists of characters, we have to decide why this happens
                    if len(current_scene["repliques"]) == 1 and current_scene["speakers_text"] is None:
                        # If there was already just one speaker in the scene, then it was the list of speakers.
                        # We update it accordingly
                        current_scene["speakers_text"] = current_scene["repliques"][-1]["content"]
                        current_scene["repliques"].pop()
                        current_scene["repliques"].append({"type": "Speaker", "content": character_name})
                        added_character = True
                    elif '<em>' in character_name:
                        # If there's an <em>, it's probably a stage indication
                        current_scene["repliques"].append(
                            {"type": "Stage", "content": remove_html_tags(character_name)})
                    else:
                        # Sometimes, there is a problem in Theatre Doc. In this case, the previous line was centered, but the line of the previous character and the name of the next speaker are on the sme line.
                        # This is a dirty workaround to fix it : check if the last part of the last replique is an uppercase word followed by a dot
                        last_replique = current_scene["repliques"][-1]["content"]
                        words = last_replique.split(" ")
                        potential_last_character = words[-1].strip()
                        potential_last_line = words[:-1]
                        if potential_last_line and not(all([x.isupper() for x in potential_last_line])) and potential_last_character.isupper() and potential_last_character[-1] == '.':
                            current_scene["repliques"].pop()
                            fixed_last_replique = " ".join(words[:-1])
                            current_scene["repliques"].append({"type": "Dialogue", "content":fixed_last_replique})
                            current_scene["repliques"].append({"type": "Speaker", "content": potential_last_character})
                            added_character = True
                            print(f'split {last_replique} into {fixed_last_replique} and {potential_last_character}')
                        else:
                        # If all else fails, we still write it down
                            print(
                                f'Warning : Two consecutive char names ? {character_name} and {current_scene["repliques"][-1]["content"]}')
                            current_scene["repliques"].append({"type": "Speaker", "content": character_name})
                            added_character = True
    return counters, playContent, added_character


def speaker_currently_detected(playContent, alreadyDetected):
    """Checks if the last thing detected in the text currently is a speaker
        Returns the list where to append the next replique"""
    if not playContent:
        return False, None
    else:
        curr_elem = playContent[-1]
        # Getting the last scene
        if "actName" in curr_elem and curr_elem["Scenes"]:  # There's an act with a scene
            curr_scene = curr_elem["Scenes"][-1]
        elif "sceneName" in curr_elem:  # There's no act
            curr_scene = curr_elem
        else:
            return False, None

        if curr_scene["repliques"]:
            if alreadyDetected or (curr_scene["repliques"][-1]["type"] == "Speaker"):
                return True, curr_scene["repliques"]
            else:
                return False, None
        else:
            return False, None


def find_text(line, counters, playContent, scene):
    res = re.search("<p[^>]*>(.*)</p>", line)
    if res:
        playLine = res.group(1).replace("\xa0", " ")
        if playLine != " ":
            res = re.search("<em>(.*)</em>", playLine)
            # playLine = remove_html_tags(playLine)
            new_line = {"content": playLine}
            if res:  # This is stage direction
                new_line["type"] = "Stage"
            else:  # this is dialogue
                new_line["type"] = "Dialogue"
            scene.append(new_line)


def correct_character_id(characterId, counters, characters_in_scene, max_distance=3):
    """Given a character identifier, tries to find the correct character id, by comparing it with know legal ids,
    using edit distance and the list of characters present in the scene."""
    # even when normalizing character names, we often find ids that are not declared
    # this part aims at correcting that by checking if the name is part of a known id,
    # or if a known id is part of it
    # If everything fails, (which happens often), we use edit distance to find the closest one
    old_characterId = characterId
    # if len(characterId) >= 15:
    #     print(characterId)
    if characterId not in counters['characterIDList']:
        if characterId not in counters["undeclaredCharacterIDs"]:
            # print(f"Warning : unknown character id {characterId}")
            edit_distances = dict()
            for true_id in counters["characterIDList"]:
                if re.search(true_id, characterId) or re.search(characterId, true_id):
                    counters["undeclaredCharacterIDs"][characterId] = true_id
                    characterId = true_id
                    # print(f"Guessed {true_id} for {old_characterId}")
                    break
                else:
                    distance = editdistance.eval(characterId, true_id)
                    edit_distances[true_id] = distance
            # characterID has not been guessed with subchains
            if old_characterId not in counters["undeclaredCharacterIDs"]:
                closest_id, closest_distance = min_dict(edit_distances)
                if (closest_id in characters_in_scene and closest_distance <= 5) or closest_distance <= max_distance:
                    # print(f"{old_characterId} : Guessed {closest_id}, distance {closest_distance} ")
                    counters["undeclaredCharacterIDs"][characterId] = closest_id
                else:
                    # print(f"Could not guess {characterId} (best guess {closest_id})")
                    counters["undeclaredCharacterIDs"][characterId] = characterId
                    counters["unguessed_id"] = True
        else:
            characterId = counters["undeclaredCharacterIDs"][characterId]
    return characterId


def identify_character_ids(scene, counters):
    """Take a scene and tries to guess the correct ID for speakers appearing in the scene.
    Also corrects the text of speakers"""
    # First establish id of characters in scene
    speakers_text = []
    if scene["speakers_text"]:
        speakers = scene["speakers_text"]
        speakers = re.sub('puis|et', ',', speakers)  # Trying to get rid of delimiters
        speakers = remove_html_tags_and_content(speakers)  # Getting rid of stage indications
        speakers_text = speakers.split(',')
    scene["speaker_ids"] = set()
    for speaker in speakers_text:
        character = remove_html_tags_and_content(speaker)
        character_id = normalize_character_name(character)
        corrected_id = correct_character_id(character_id, counters, [], 3)
        if corrected_id in counters['characterIDList']:
            scene["speaker_ids"].add(corrected_id)
    for replique in scene["repliques"]:
        if replique["type"] == "Speaker":
            character = replique["content"]

            # character is the actual character name, from which we strip html tags.
            # clean_character will be used to get the corresponding id
            # Checking if the character name is preceded by a comma, indicating an action on stage.
            # Dracor convention seems to be to include it as a content of the speaker tag and not in <stage>,
            # so we follow this rule
            has_stage_direction = re.search("([^,<]+)(?:,.*|<em>.*</em>.*)", character)
            if has_stage_direction:
                character = has_stage_direction.group(1)
            character = remove_html_tags_and_content(character)
            clean_character = character
            # Removing ending dot if it exists
            characterId = normalize_character_name(clean_character)
            guessed_charactedId = correct_character_id(characterId, counters, scene["speaker_ids"])
            replique["characterId"] = guessed_charactedId


# TMP
def update_csv(row, key, value):
    if key not in row or not row[key] or row[key].strip() == '':
        row[key] = value


def get_metadata(counters, csv_row, metadata, line):
    # get and write title
    title, forename, surname = get_title_and_author(line)
    metadata['main_title'] = title
    metadata['author_forename'] = forename
    metadata['author_surname'] = surname
    update_csv(csv_row, "Title", title)
    update_csv(csv_row, "Author", f'{" ".join(forename)} {" ".join(surname)}')
    if title:
        # get type of play:
        copy_playtext = open(file, "r", encoding="utf-8")
        genre, vers_prose, act_number = get_genre_versification_acts_number(copy_playtext)
        metadata['vers_prose'] = vers_prose
        metadata["genre"] = genre
        update_csv(csv_row, "Genre", genre)
        update_csv(csv_row, "Type", vers_prose)
        if act_number == 1:
            counters["oneActPlay"] = True
            counters["actsInPlay"] = 1
        if act_number != -1:
            counters["actsDeclaredNumber"] = act_number
        # get date
        copy_playtext.close()
        copy_playtext = open(file, "r", encoding="utf-8")
        date_written, date_print, date_premiere, line_written, line_print, line_premiere = get_dates(
            copy_playtext)
        update_csv(csv_row, "Date ecriture", date_written)
        update_csv(csv_row, "Date impression", date_print)
        update_csv(csv_row, "Date premiere", date_premiere)
        metadata["date_written"] = date_written
        metadata["date_print"] = date_print
        metadata["date_premiere"] = date_premiere
        metadata['doc_date_text'] = 'unknown'  # TODO: Find it
        metadata['premiere_text'] = line_premiere
        metadata['print_text'] = line_print
        metadata['written_text'] = line_written
        metadata['premiere_location'] = 'unknown'  # TODO : Find it (should be easy)
        copy_playtext.close()


def find_dedicace_or_preface_start(line, inBlock, headerType):
    """Returns True if and only if the given line is the start of the dedicace or preface"""
    if inBlock:
        return True
    dedicace_header = re.search(".*<h1[^>]*>(.*)</h1>", line)
    if dedicace_header:
        header_text = clean_scene_name(dedicace_header.group(1))
        if header_text == metadata[headerType]:
            inBlock = True
    return inBlock


def find_dedicace_or_preface_content(line, type):
    res = re.search("<p[^>]*>(.*)</p>", line)
    if res:
        contentLine = res.group(1).replace("\xa0", " ")
        if contentLine != " ":
            if type not in metadata:
                metadata[type] = [contentLine]
            else:
                metadata[type].append(contentLine)
    regex_act = ".*<h1[^>]*>(.*)</h1>"
    regex_scene = "|".join(
        [".*<h2 .*<strong>(?P<h2strong>.*)</strong>.*</h2>", ".*<h2.*>(?P<h2normal>Scène.*)</h2>",
         "<h1.*>(?P<h1>.*Scène.*)</h1>"])
    newActOrScene = re.search(f"{regex_scene}|{regex_act}", line)
    return not newActOrScene


def write_play(metadata, playContent, counters, outputFile):
    output = writing.write_full_tei_file(metadata, playContent, counters)
    writing.output_tei_file(output, outputFile)
    return output


if __name__ == "__main__":
    output_csv = open("metadata theatredoc.csv", 'w', encoding='utf8', newline='')
    fieldnames = ["Raw Name", "Title", "Author", "Date impression", "Date premiere", "Date ecriture", "Genre", "Type"]
    gwriter = csv.DictWriter(output_csv, fieldnames=fieldnames)
    gwriter.writeheader()

    # stats temporary
    castnotWritten = 0
    noact = 0
    undeclared_character = 0
    unguessed_character = 0
    totalplays = 0
    sceneList_ok = 0
    possible_secnes_and_acts_strings = set()
    number_of_acts_correctly_declared = 0
    stats = open('stats_characters.txt', 'w+')

    # Declaration of flags and counters.
    documentNb = 0
    findSummary = False
    characterBlock = False
    ul = 0
    characterBlockLastLine = None

    # prepare the list of file sources
    fileSources = {}
    allPlays = extract_sources(open("PlaysFromTheatreDocumentation.csv", "r", encoding="utf-8"), fileSources)
    # Generate an XML-TEI file for every HTML file of the corpus
    default = (None, None, [])
    for file in list(map(lambda f: join(html_folder, f), next(walk(html_folder), default)[2])):
        #notify_file(file)

        # Find source
        fileName = basename(file)
        playText = open(file, "r", encoding="utf-8")
        outputFile = os.path.join(Dracor_Folder, fileName.replace("html", "xml"))
        # reset parameters
        csv_row = dict()
        csv_row["Raw Name"] = fileName.replace(".html", " ")
        metadata = dict()
        metadata['source'] = get_source(fileSources, fileName)
        counters = {
            "charactersinScene": "",
            "repliquesinScene": 0,
            "linesInPlay": 0,
            "linesInScene": 0,
            "scenesInAct": 0,
            "actsInPlay": 0,
            "noActPlay": False,
            "oneActPlay": False,
            "scenelessPlay": False,
            "scenelessPlayBeginningWritten": False,
            "characterLines": [],
            "characterIDList": [],
            "characterFullNameList": [],
            "roleList": [],
            "actNb": "",
            "sceneNb": "",
            "dedicaceFound": False,
            "dedicaceFinished": False,
            "preface": [],
            "prefaceFound": False,
            "prefaceFinished": False,
            "undeclaredCharacterIDs": dict(),
            "sceneList": [],
            "actsDeclaredNumber": -1,  # temp
            "unguessed_id": False  # temporary, delete later
        }
        # Reading the file a first time to find the characters
        inSceneList = False
        sceneList = []
        for index, line in enumerate(standard_line(playText)):
            # starting character block
            characterBlock = start_character_block(line, characterBlock)

            # Ending character block
            # We remember the ending line of the character block for the future
            # We do so by checking if the list of characters grows
            old_nb_char = len(counters["characterFullNameList"])
            characterBlock, line = end_character_block(characterBlock, line)
            new_nb_char = len(counters["characterFullNameList"])
            if old_nb_char != 0 and new_nb_char == old_nb_char and characterBlockLastLine is None:
                characterBlockLastLine = index

            # Getting scene list
            if not counters["sceneList"]:
                sceneList, inSceneList = find_scene_list(line, sceneList, inSceneList)
                # We detect that we have finished reading the scene List when some scenes have already been collected
                # and the flag inSceneList goes to false
                if sceneList and not inSceneList:
                    counters["sceneList"] = sceneList
                    if len(sceneList) == 1:
                        if sceneList[0][0] == 'Acte unique':
                            counters["oneActPlay"] = True
        playText.close()

        # Reading the file a second time to get metadata, text, and write output
        playText = open(file, "r", encoding="utf-8")
        playContent = []
        characterBlockFinished = False
        speaker_already_detected = False
        inDedicace, inPreface = False, False
        for index, line in enumerate(playText):
            if index == characterBlockLastLine:
                characterBlockFinished = True
            # Getting all metadata :
            get_metadata(counters, csv_row, metadata, line)

            # Some text can be before the beginning of the play : a dedicace, or a preface.
            # Dedicace
            if counters["dedicaceFound"] and not counters["dedicaceFinished"]:
                if inDedicace:
                    inDedicace = find_dedicace_or_preface_content(line, "dedicace")
                    if not inDedicace:  # If we have finished reading the dedicace
                        counters["dedicaceFinished"] = True
                        last_dedicace_line = metadata["dedicace"][-1] # Checking if the last line is a signature
                        if last_dedicace_line.isupper():
                            metadata["dedicace"].pop()
                            metadata["signed_text"] = last_dedicace_line
                else:
                    inDedicace = find_dedicace_or_preface_start(line, inDedicace, "dedicaceHeader")

            # Preface
            if counters["prefaceFound"] and not counters["prefaceFinished"]:
                if inPreface:
                    inPreface = find_dedicace_or_preface_content(line, "preface")
                    if not inPreface:  # If we have finished reading the preface
                        counters["prefaceFinished"] = True
                inPreface = find_dedicace_or_preface_start(line, inPreface, "prefaceHeader")

            # Now we read the whole text to find the body of the play. We are constructing a list called playContent
            # containing the whole play. It is structured as follows: playContent is either a list of acts or a list
            # of scenes.
            # An act is a dictionnary with the following keys :
            # "actNumber", "actName", "actStageIndications" (stage indications that may be placed outside of scenes), and "Scenes"
            # Scenes is a list of scenes. A scene is a dict structured with the following keys :
            # "sceneName", "sceneNumber", "speakers_text" (the string with the declaration of characters),
            # "speakers_ids" (the id of said speakers), "repliques"
            # "repliques" is a list of repliques. A replique is a dict with the following keys :
            # "type", which can either be "Dialogue","Speaker", or "Stage"
            # "content", which contains the actual content of the replique
            # If the type is Speaker, there is an additional key "characterId", containing the Id of the character speaking
            if (not counters["dedicaceFound"] or counters["dedicaceFinished"]) and (
                    not counters["prefaceFound"] or counters["prefaceFinished"]):
                line, counters, playContent = find_begin_act(line, counters, playContent)
                line, counters, playContent = find_begin_scene(line, counters, playContent)

                # Also Handling case of plays with no scenes:
                # No list of scene is present at the beginning, but a list of character has been done
                if not counters["sceneList"] and counters["characterIDList"]:
                    character_names_string = '|'.join(counters["characterFullNameList"])
                    if re.match(f"<p align=.*center[^>]*>(<span style=.*>)?.*</p>", line):
                        counters["scenelessPlay"] = True
                # We start reading the text once at least an act or a scene has been found
                # Or if it has been established that there is no scenes in the play
                # And we are done reading the cast of characters (or, there are none)
                if (playContent or counters["scenelessPlay"]) and (
                        characterBlockFinished or not counters["characterFullNameList"]):
                    # Getting character declaration
                    # speaker_detected
                    counters, playContent, added_character = find_character(line, counters, playContent)
                    # Getting text
                    speaker_already_detected, current_scene = speaker_currently_detected(playContent,
                                                                                         speaker_already_detected)
                    if speaker_already_detected and not added_character:
                        find_text(line, counters, playContent, current_scene)

        # Since characters names often have typos or are not exactly as described, we now correct those names
        # We also establish the list of characters speaking per scene

        # If there's no act, treat the entire playContent as a list of scenes
        scenes = playContent if counters["noActPlay"] else [scene for act in playContent for scene in act["Scenes"]]

        # Process each scene and identify character ids
        for scene in scenes:
            identify_character_ids(scene, counters)

        # Writing play
        gwriter.writerow(csv_row)
        write_play(metadata, playContent, counters, outputFile)

        # Stats collection, temporary
        if counters["sceneList"]:
            sceneList_ok += 1
            for x in counters["sceneList"]:
                possible_secnes_and_acts_strings.add(x[0])
                for y in x[1]:
                    possible_secnes_and_acts_strings.add(y)
        if len(counters["characterIDList"]) == 0:
            castnotWritten += 1
        if counters["actsInPlay"] == 0:
            noact += 1
            # if counters["sceneList"]:
            #     print(f'Play with no act but scene list : {file}'
            #           f'{counters["sceneList"]}')
            # else:
            #     print(f'Play with no act but no scene list : {file}')
        if len(counters["undeclaredCharacterIDs"]) > 0:
            undeclared_character += 1
        if counters["unguessed_id"]:
            unguessed_character += 1
        if counters["actsInPlay"] == counters["actsDeclaredNumber"]:
            number_of_acts_correctly_declared += 1
        totalplays += 1

    stats.writelines(f"""Total number of plays : {totalplays}
    Plays with no acts found : {noact}
    Plays with no cast of character found : {castnotWritten}
    Plays with unknow character ids found : {undeclared_character}
    Among those, plays where at least one character could not be guessed : {unguessed_character}
    Act number declared corresponding to act number found : {number_of_acts_correctly_declared}
    Scene List found : {sceneList_ok}""")
    print(f"Casts not written: {castnotWritten} sur {totalplays}")

# Plan :
# Une fois que toutes les métadonnées sont collectées :
# Parcourir la pièce, et garder les actes, scènes, et dialogues en mémoire
# Une fois qu'on arrive au bout du fichier : tout écrire

# Comment garder le texte en mémoire ?
# Il faut garder l'ordre : liste
# [ ('Acte Name', [('Scene Name',[(speakername, speaker id),[réplique 1, réplique 2,...]),(),...],(),()]),()...]
# Pour écrire : juste parcourir la liste et écrire au fur et à mesure
# Pour la liste des speakers dans la scène :
# On peut comparer la liste des speakers déclarés avec la liste des speakers collectés et deviner parmis ceux là
# Il faut deviner les speakers id ultérieurement ?
# Quand on trouve un speaker : si l'id est connu on le garde, si il est inconnu on le met en None
# Phase de correction : pour chaque None, on regarde l'id obtenu en normalisant juste, et on le compare par rapport à la liste des ids correspondants aux persos de la scène

# REMINDER : Stuff to write between acts :
# if counters["actsInPlay"] == 0:
#     outputFile.writelines("""
# </front>
# <body>""")
# end the previous scene of the previous act
# outputFile.writelines("""
# </sp>
# </div>""")

# Don't forget to handle plays with no acts or no scenes
# if not counters["castWritten"]:
#     write_character(outputFile)
#     counters["castWritten"] = True
# outputFile.close()

# Reminder : Stuff to write scenes
# if counters["scenesInAct"] == 1:
#     if counters["oneActPlay"]:
#         counters["actsInPlay"] = 1
#         # TODO : Vérifier que cette écriture est correcte pour le début de l'acte
#         write_act("1", "ACTE 1", outputFile)
#     write_scene(counters["actNb"] + str(counters["scenesInAct"]), scene + counters["charactersinScene"],
#                 outputFile)
#     if counters["scenelessPlay"] and not (counters["scenelessPlayBeginningWritten"]):
#         counters["scenelessPlayBeginningWritten"] = True
# else:
#     outputFile.writelines("""
#    </sp>
# </div>""")
#     write_scene(str(counters["actNb"]) + str(counters["scenesInAct"]), scene, outputFile)

# METADATA ADRIEN (now in function, delete if it works)
# # get and write title
# title, forename, surname = get_title_and_author(line)
# if write_title(outputFile, title):
#     # get and write type of play:
#     copy_playtext = open(file, "r", encoding="utf-8")
#     genre, vers_prose, act_number = get_genre_versification_acts_number(copy_playtext)
#     if act_number == 1:
#         counters["oneActPlay"] = True
#         counters["actsInPlay"] = 1
#     if act_number != -1:
#         counters["actsDeclaredNumber"] = act_number
#     write_type(outputFile, genre)
#     # get and write author
#     author = forename, surname
#     if write_author(outputFile, author):
#         # get and write source
#         write_source(outputFile, source)
#
#     # get and write date
#     copy_playtext.close()
#     copy_playtext = open(file, "r", encoding="utf-8")
#     date_written, date_print, date_premiere, line_written, line_print, line_premiere = get_dates(
#         copy_playtext)
#
#     write_dates(outputFile, date_written, date_print, date_premiere, line_premiere)
#
#     write_end_header(outputFile, genre, vers_prose)
#     write_start_text(outputFile, title, genre, date_print)
#
#     write_performance(outputFile, line_premiere, date_premiere)
#
# # try find dedicace in play
# if not findSummary:
#     findSummary = find_summary(line, ul)
# else:
#     findSummary = extract_from_summary(line, ul)
#
# # starting saving lines
# if not saveBegin:
#     saveBegin = try_saving_lines(outputFile, line)
# else:
#     # find and print dedicace
#     if counters['dedicace']:
#         if find_dedicace(line):
#             copy_playtext.close()
#             copy_playtext = open(file, "r", encoding="utf-8")
#             write_dedicace(outputFile, copy_playtext, author)

# def write_text(outputFile, line, counters):
#     """Write the text from a HTML file's line in the XML associated file.
#
#     Args:
#         outputFile (TextIOWrapper): Output file to generate in XML.
#         line (str): line to read in the play.
#         counters (dict): Dictionnary with all the counters of the script.
#
#     Returns:
#         dict: The refreshed counter
#     """
#     res = re.search("<p>(.*)</p>", line)
#     if res and not characterBlock:
#         playLine = res.group(1).replace("\xa0", " ")
#         if playLine != " ":
#             # log('sceneless',counters["scenelessPlay"])
#             # log("pbg",counters["scenelessPlayBeginningWritten"])
#             if counters["scenelessPlay"] and not counters["scenelessPlayBeginningWritten"]:
#                 print('scenelessstart')
#                 find_begin_scene(outputFile, line, counters)
#             if len(counters["characterLines"]) > 1:
#                 character = counters["characterLines"].pop(0)
#                 outputFile.writelines("""
#         <stage>""" + character + """</stage>""")
#             if len(counters["characterLines"]) > 0:
#                 if counters["repliquesinScene"] > 0:
#                     outputFile.writelines("""
#       </sp>""")
#                 character = counters["characterLines"].pop(0)
#                 counters["repliquesinScene"] += 1
#
#                 # character is the actual character name, from which we strip html tags.
#                 # clean_character will be used to get the corresponding id
#                 character = remove_html_tags_and_content(character)
#                 clean_character = character
#                 # Checking if the character name is preceded by a comma, indicating an action on stage.
#                 # Dracor convention seems to be to include it as a content of the speaker tag and not in <stage>,
#                 # so we follow this rule
#                 has_stage_direction = re.search("([^,]+),.*", clean_character)
#                 if has_stage_direction:
#                     clean_character = has_stage_direction.group(1)
#                 # Removing ending dot if it exists
#                 if clean_character[-1] == ".":
#                     clean_character = clean_character[:-1]
#                 characterId = normalize_character_name(clean_character)
#
#                 # even when normalizing character names, we often find ids that are not declared
#                 # this part aims at correcting that by checking if the name is part of a known id,
#                 # or if a known id is part of it
#                 # If everything fails, (which happens often), we use edit distance to find the closest one
#                 old_characterId = characterId
#                 if characterId not in counters['characterIDList']:
#                     if characterId not in counters["undeclaredCharacterIDs"]:
#                         # print(f"Warning : unknown character id {characterId}")
#                         edit_distances = dict()
#                         for true_id in counters["characterIDList"]:
#                             if re.search(true_id, characterId) or re.search(characterId, true_id):
#                                 counters["undeclaredCharacterIDs"][characterId] = true_id
#                                 characterId = true_id
#                                 print(f"Guessed {true_id} for {old_characterId}")
#                                 break
#                             else:
#                                 distance = editdistance.eval(characterId, true_id)
#                                 edit_distances[true_id] = distance
#                         # characterID has not been guessed with subchains
#                         if old_characterId not in counters["undeclaredCharacterIDs"]:
#                             closest_id, closest_distance = min_dict(edit_distances)
#                             if closest_distance <= 5:
#                                 print(f"{old_characterId} : Guessed {closest_id}, distance {closest_distance} ")
#                                 counters["undeclaredCharacterIDs"][characterId] = closest_id
#                             else:
#                                 # print(f"Could not guess {characterId} (best guess {closest_id})")
#                                 counters["undeclaredCharacterIDs"][characterId] = characterId
#                                 counters["unguessed_id"] = True
#                     else:
#                         characterId = counters["undeclaredCharacterIDs"][characterId]
#
#                 # if characterId == "":
#                 #     print(line)
#                 #     print("entering characterId if")
#                 #     # print("Character not found: " + character)
#                 #     res = re.search("([^,.<]+)([.,<].*)", character)
#                 #     if res:
#                 #         characterId = res.group(1).lower()
#                 #         # remove spaces in last position
#                 #         res = re.search("^(.*[^ ])[ ]+$", characterId)
#                 #         if res:
#                 #             characterId = res.group(1)
#                 #         characterId = characterId.replace(" ", "-")
#                 #         # print("Chose characterId " + characterId)
#                 outputFile.writelines(f"""
#             <sp who=\"#{characterId}\" xml:id=\"{counters["actNb"] + str(counters["scenesInAct"]) + "-" + str(
#                     counters["repliquesinScene"])}\">
#                 <speaker> {character} </speaker>""")
#
#             # Checking whether this line is dialogue or stage directions
#             res = re.search("<em>(.*)</em>", playLine)
#             if res:
#                 outputFile.writelines(f"""
#             <stage>{remove_html_tags_and_content(playLine)} </stage>""")
#             else:
#                 outputFile.writelines("""
#             <l n=\"""" + str(counters["linesInPlay"]) + """\" xml:id=\"l""" + str(
#                     counters["linesInPlay"]) + """\">""" + remove_html_tags_and_content(playLine) + """</l>""")
#                 counters["linesInPlay"] += 1
#                 counters["linesInScene"] += 1
#
#     return counters
