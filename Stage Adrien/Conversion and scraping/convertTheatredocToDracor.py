#!/usr/sfw/bin/python
# -*- coding: utf-8 -*-

import os, re, sys
from os import walk, pardir
from os.path import abspath, dirname, join, basename, exists
from datetime import date

import enchant.utils
from enchant import utils
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

# Get the current folder
folder = abspath(dirname(sys.argv[0]))
root_folder = abspath(join(folder, pardir))
html_folder = abspath(join(root_folder, "cleanHTML_TD"))
Dracor_Folder = abspath(join(root_folder, "corpusTD_v2"))
clean_Dracor_Folder = abspath(join(root_folder, "corpusTD_cast_ok"))

if not exists(Dracor_Folder):
    os.system("mkdir {0}".format(Dracor_Folder))

### temporaire
# date_file = open(join(root_folder, 'datesTD.txt'), 'w')
# count_date = 0
###temporaire

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

genres = ["tragedie", "comedie", "tragicomedie", "tragi-comedie"]
clean_genre = list(map(lambda g: g[0].upper() + g.replace('e', 'é')[1:-1] + g[-1], genres))


def good_genre(genre):
    return clean_genre[genres.index(genre)]


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


def remove_html_tags_and_content(s):
    s = s.replace(u'\xa0', u' ')
    return re.sub('<[^>]+>', '', s)

def min_dict(d):
    """Returns (key_min, val_min) such that val_min is minimal among values of the dictionnary"""
    val_min = 100
    key_min = None
    for x in d:
        if d[x] < val_min:
            val_min = d[x]
            key_min = x
    return key_min, val_min

def normalize_character_name(s):
    clean_character_name = s.lower().replace("*", "")
    clean_character_name = re.sub("[\[\]\)\(]", "", clean_character_name)
    return re.sub(" +", "-", clean_character_name)


def standard_line(playText):
    return list(map(lambda l: l.replace("<span style=\"letter-spacing:-.3pt\">", "").replace("\xa0", ' ').replace(
        '<a href="#_ftn1" name="_ftnref1" title="" id="_ftnref1">[1]</a>', '').strip('\n'), playText))


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


def notify_file(file):
    """Notify the user with the conversion of the input file.

    Args:
        file (str): Name of the file to convert.
    """
    print("Converting file " + file)
    # date_file.writelines(basename(file).replace(".html", '') + "\t")


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


def write_title(outputFile, title):
    """Write the extracted title in the output file in XML.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        title (str): Title of a file.

    Returns:
        str: The same title.
    """
    if title:
        outputFile.writelines("""<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="fre">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title type="main">""" + title + """</title>""")
    return title


def get_type(playText):
    """Get the type of the play, and if it's in prose or in verses.

    Args:
        playText (TextIOWrapper): Text Contents of a play.

    Returns:
        _tuple_: Tuple of strings. The genre of the play, and the type of writing (verses or prose). Return [indéfini] if it's undefined.
    """
    res_genre, vers_prose = '[indéfini]', '[indéfini]'
    for l in standard_line(playText):
        res = re.search('<p>(.*)</p>', l)
        if res:
            content = res.group(1).lower().replace('é', 'e')
            if content == ' ':
                break
            for genre in genres:
                if genre in content:
                    res_genre = good_genre(genre)
                    break
            if 'prose' in content:
                vers_prose = 'prose'
            elif 'vers' in content:
                vers_prose = 'vers'
    return res_genre, vers_prose


def write_type(outputFile, genre):
    """Write the extracted genre in the output file in XML.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        genre (str): Genre of a play.
    """
    if genre != '[indéfini]':
        outputFile.writelines("""
                <title type="sub">""" + genre + """</title>""")


def write_author(outputFile, author):
    """Write the author's name in the output file in XML.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        author (str): Author of the play.
    
    Returns:
        bool: True if the author's name have at least a forename or a surname, False then.
    """
    forename, surname = author
    if forename or surname:
        outputFile.writelines("""
            <author>
                <persName>""")
        if forename:
            for name in forename:
                if name in ['de', "d'"]:
                    outputFile.writelines("""
                    <linkname>""" + name + """</linkname>""")
                elif name in ['Abbé']:  # TODO identifier d'autres rolename
                    outputFile.writelines(f"""
                    <rolename>{name}</rolename>""")
                else:
                    outputFile.writelines(f"""
                    <forename>{name}</forename>""")
        if surname:
            for name in surname:
                if name in ['DE', "D'"]:
                    outputFile.writelines("""
                    <linkname>""" + name.lower() + """</linkname>""")
                else:
                    outputFile.writelines("""
                    <surname>""" + ''.join([name[0], name[1:].lower()]) + """</surname>""")

        outputFile.writelines("""
                </persName>
            </author>
                        <editor>Adrien Roumégous, dans le cadre d'un stage de M1 Informatique encadré par Aaron Boussidan et Philippe Gambette.</editor>
            </titleStmt>""")
        return True
    return False


def write_source(outputFile, source):
    """Write the source of the play in the output file in XML.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        source (str): Source of the play.
    """
    outputFile.writelines(f"""
            <publicationStmt>
                <publisher xml:id="dracor">DraCor</publisher>
                <idno type="URL">https://dracor.org</idno>
                <idno type="dracor" xml:base="https://dracor.org/id/">fre[6 chiffres]</idno>
                <idno type="wikidata" xml:base="http://www.wikidata.org/entity/">Q[id]</idno>
                <availability>
                    <licence>
                        <ab>CC BY-NC-SA 4.0</ab>
                        <ref target="https://creativecommons.org/licenses/by-nc-sa/4.0/">Licence</ref>
                    </licence>
                </availability> 
            </publicationStmt>
            <sourceDesc>
                <bibl type="digitalSource">
                    <name>Théâtre Documentation</name>
                    <idno type="URL"> {source} </idno>
                    <availability>
                        <licence>
                            <ab>loi française n° 92-597 du 1er juillet 1992 et loi n°78-753 du 17 juillet 1978</ab>
                            <ref target="http://théâtre-documentation.com/content/mentions-l%C3%A9gales#Mentions_legales">Mentions légales</ref>
                        </licence>
                    </availability>
                    <bibl type="originalSource">""")


def get_dates(playText):
    """Get the date of writing, the date of printing and the date of first performance of the play, and the line of context for each of them.

    Args:
        playText (TextIOWrapper): Text Contents of a play.

    Returns:
        tuple: Return a tuple of 6 strings :
            - Date of writing
            - Date of printing
            - Date of first performance
            - Line of date of writing
            - Line of date of printing
            - Line of date of first performance
    """
    # global count_date
    line_written = "[vide]"
    line_print = "[vide]"
    line_premiere = "[vide]"
    date_written = "[vide]"
    date_print = "[vide]"
    date_premiere = "[vide]"
    is_written = False
    is_print = False
    is_premiere = False

    for l in standard_line(playText):

        if re.search(".*<strong><em>Personnages.*</em></strong>.*", l) or re.search(
                '<p align="center" style="text-align:center"><b><i>Personnages.*</span></i></b></p>', l) or (
                True in (is_written, is_print, is_premiere) and l == '<p> </p>'):
            break

        if re.search("<p>Non représenté[^0-9]*</p>", l):
            line_premiere = l.replace("<p>", "").replace("</p>", "")
            break

        if not is_written and not is_print:
            res = re.search("<p>.*[ÉéEe]crit en ([0-9]+).* et [op]ublié.* en ([0-9]+).*</p>", l)
            if res:
                line_written = l.replace('<p>', '').replace('</p>', '')
                line_print = l.replace('<p>', '').replace('</p>', '')
                date_written, date_print = res.groups()
                is_written, is_print = True, True

        if not is_written:
            res = re.search("<p>.*[ÉéEe]crit[e]? (.*)</p>", l)
            if res:
                line_written = l.replace('<p>', '').replace('</p>', '')
                res2 = re.search(".*le ([0-9]+) ([^ ]+) ([0-9]+).*", res.group(1))
                if res2:
                    date_written = format_date_AAAAMMJJ(res2.groups())
                    is_written = True
                else:
                    res2 = re.search(".*en ([0-9]+).*", res.group(1))
                    res3 = re.search(".*en ([^0-9 ]+) ([0-9]+).*", res.group(1))
                    if res2:
                        date_written = res2.group(1)
                        is_written = True
                    elif res3:
                        date_written = format_date_AAAAMM(res3.groups())
                        is_written = True

        if not is_premiere and not is_print:
            res = re.search(
                "<p>Publié.* ([0-9]+) et représenté.* ([0-9]+|1<sup>er</sup>|premier) ([^ ]+) ([0-9]+).*</p>", l)
            res2 = re.search("<p>Publié.* ([0-9]+) et représenté.* ([0-9]+).*</p>", l)
            if res or res2:
                is_print, is_premiere = True, True
                if res:
                    date_print, date_premiere = res.group(1), format_date_AAAAMMJJ(res.groups()[1:])

                elif res2:
                    date_print, date_premiere = res2.group(1), res2.group(2)
                is_print, is_premiere = True, True
                line_print, line_premiere = l.replace("<p>", "").replace("</p>", ""), l.replace("<p>", "").replace(
                    "</p>", "")

        date_line = re.search("<p>.*([Rr]eprésenté.*)</p>", l)
        date_line2 = re.search("<p>.*(fut joué.*)</p>", l)
        if (date_line or date_line2) and not is_premiere:
            if date_line2:
                date_line = date_line2
            date_line = date_line.group(1)
            res = re.search(".* ([l\|]?[0-9]+|1<sup>er</sup>|premier)[ ]+([^ ]+) ([l\|]?[0-9]+).*", date_line)
            res2 = re.search(".* ([0-9]+|1<sup>er</sup>|premier)[ ]+([^ ]+) ([0-9]+).*" * 2, date_line)
            double_words_res = re.search(
                ".* ([l\|]?[0-9]+|1<sup>er</sup>|premier)[ ]+([^ ]+)[ ]+([^ ]+) ([l\|]?[0-9]+).*", date_line)
            between_years_res = re.search(".* ([0-9]+)-([0-9]+).*", date_line)
            line_premiere = date_line
            if res:
                if res2:
                    date_premiere = format_date_AAAAMMJJ(res2.groups())
                else:
                    date_premiere = format_date_AAAAMMJJ(res.groups())
                is_premiere = True
            elif double_words_res:
                if double_words_res.group(2).replace('é', 'e') in mois:
                    groups = (double_words_res.group(1), double_words_res.group(2), double_words_res.group(4))
                else:
                    groups = (double_words_res.group(1), double_words_res.group(3), double_words_res.group(4))
                date_premiere = format_date_AAAAMMJJ(groups)
                is_premiere = True
            elif between_years_res:
                date_premiere = between_years_res.groups()
                is_premiere = True
            else:
                res = re.search(".* en ([0-9]+).*", date_line)
                res2 = re.search(".* en ([0-9]+).*" * 2, date_line)
                res3 = re.search(".* en ([0-9]+).*" * 3, date_line)
                if res:
                    if res2 is not None:
                        res = res2
                        if res3 is not None:
                            res = res3
                    date_premiere = res.group(1)
                    is_premiere = True
                else:
                    res = re.search(".* (en|le|de) ([^ ]+) ([0-9]+).*", date_line)
                    weird_res = re.search(".* (en|le|de)([0-9]+) ([^ ]+) ([0-9]+).*", date_line)
                    if res:
                        res2 = re.search("([0-9]+)(.*)", res.group(2))
                        if res2:
                            date_premiere = format_date_AAAAMMJJ(res2.groups() + res.groups()[2:])
                        elif res:
                            date_premiere = format_date_AAAAMM(res.groups()[1:])
                        is_premiere = True
                    elif weird_res:
                        date_premiere = format_date_AAAAMMJJ(weird_res.groups()[1:])
                        is_premiere = True

        if not is_print:
            res = re.search("<p>([0-9]+).*</p>", l)
            res2 = re.search("<p>Imprimée en ([0-9]+).*</p>", l)
            res3 = re.search("<p>Non représentée[,\.] ([0-9]+).*</p>",
                             l.replace('<a href="#_ftn1" name="_ftnref1" title="" id="_ftnref1">[1]</a>', ''))

            if res or res2 or res3:
                if res is None:
                    res = res2
                    if res2 is None:
                        res = res3
                if len(res.group(1)) == 4:
                    date_print = res.group(1)
                    line_print = l.replace("<p>", "").replace("<p>", "")
                    is_print = True

        if date_line is None:
            date_line = ""

    # if not (is_print or is_premiere or is_written):
    #     count_date += 1

    if not is_written:
        line_written = "[vide]"

    # date_file.writelines(line_written + '\t' + line_print + '\t' + line_premiere + '\t')

    # date_file.writelines(date_written + '\t')

    # date_file.writelines(date_print + '\t')

    # date_file.writelines(str(date_premiere) + "\n")

    return date_written, date_print, date_premiere, line_written, line_print, line_premiere


def write_dates(outputFile, date_written, date_print, date_premiere, line_premiere):
    """Write the date of writing, the date of printing and the date of first performance of the play, and the line of context for each of them in an output file in XML.

    Args: 
        outputFile (TextIOWrapper): Output file to generate in XML.
        date_written (str): Date of writing of the play.
        date_print (str): Date of printing of the play.
        date_premiere (str): Date of first performance of the play.
        line_premiere (str): Line where the date of the first performance is written.
    """
    if date_written != "[vide]":
        outputFile.writelines("""
                        <date type="written" when=\"""" + date_written + """\">""")

    if date_print != "[vide]":
        outputFile.writelines("""
                        <date type="print" when=\"""" + date_print + """\">""")

    if date_premiere != "[vide]":
        if type(date_premiere) is str:
            outputFile.writelines("""
                        <date type="premiere" when=\"""" + date_premiere + """\">""" + line_premiere + """</date>""")
        else:
            outputFile.writelines("""
                        <date type="premiere" notBefore=\"""" + date_premiere[0] + """\" notAfter=\"""" + date_premiere[
                1] + """\" >""" + line_premiere + """</date>""")

    outputFile.writelines("""
                        <idno type="URL"/>
                    </bibl>
                </bibl>""")


def write_end_header(outputFile, genre, vers_prose):
    # TODO : Generate a better listPerson
    """Write the end of the header of a XML file

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        genre (str) : The genre of the converted play.
        vers_prose (str) : The type of the converted play, in verses or in prose.
    """
    outputFile.writelines(f"""
            </sourceDesc>
        </fileDesc>
        <profileDesc>
            <particDesc>
                <listPerson>""")

    for charaid, charaname in zip(counters["characterIDList"], counters["characterFullNameList"]):
        outputFile.writelines(f"""
                        <person xml:id="{charaid}" sex="SEX">
                            <persName>{charaname}</persName>
                        </person>""")

    outputFile.writelines(f"""
                </listPerson>
            </particDesc>
            <textClass>
            <keywords scheme="http://theatre-documentation.fr"> <!--extracted from "genre" and "type" elements-->
                    <term> {genre}</term>
                    <term> {vers_prose} </term>
                </keywords>
                <classCode scheme="http://www.wikidata.org/entity/">[QNumbers]</classCode>
            </textClass>
        </profileDesc>
        <revisionDesc>
            <listChange>
                <change when="{date.today()}">(mg) file conversion from source</change>
            </listChange>
        </revisionDesc>
   </teiHeader>""")


def write_start_text(outputFile, title, genre, date_print):
    """Write the start of the text body of a XML file.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        title (str) : The title of the converted play.
        genre (str) : The genre of the converted play.
        date_print (str) : The date of printing of the converted play.
    """
    outputFile.writelines("""
    <text>
    <front>
        <docTitle>
            <titlePart type="main">""" + title.upper() + """</titlePart>""")
    if genre:
        outputFile.writelines("""
            <titlePart type="sub">""" + genre.upper() + """</titlePart>
        </docTitle>""")
    if date_print:
        outputFile.writelines("""
        <docDate when=\"""" + date_print + """\">[Date Print Line]</docDate>
        """)


def write_performance(outputFile, line_premiere, date_premiere):
    """Write the performance tag of the chosen XML file.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        line_premiere (str) : line of the play where the date of the first performance is written.
        date_premiere (str) : The date of the first performance of the converted play.
    """
    if date_premiere != '[vide]':
        if type(date_premiere) is tuple:
            date_premiere = '-'.join(date_premiere)
        outputFile.writelines("""
        <performance>
            <ab type="premiere">""" + line_premiere + """</ab><!--@date=\"""" + date_premiere + """\"-->
        </performance>""")


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


def write_dedicace(outputFile, copy_playtext, author):
    """Write the dedicace sentence of a play in its XML version.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        copy_playtext (TextIOWrapper) : Content of the input file in HTML.
        author (tuple) : The author's name (tuple of string).
    """
    d = False
    header = True
    authorList = [i for i in author[0].extend(author[1]) if i not in ["de", "d'"]]
    for line in copy_playtext:
        dedicace = find_dedicace(line)
        if dedicace:
            outputFile.writelines("""
        <div type="dedicace">
                <opener>
                    <salute>""" + dedicace + """</salute>
                </opener>""")
            d = True
        if d:
            res = re.search('<p>(.*)</p>')
            if res:
                l = res.group(1)
                if l != ' ':
                    if header:
                        outputFile.writelines("""
        <head>""" + l + """</head>""")
                        header = False
                    elif any([i in l for i in authorList]):
                        outputFile.writelines("""
        <signed>""" + l + """</signed>
	</div>""")
                        return
                    else:
                        outputFile.writelines("""
        <p>""" + l + """</p>""")


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
        bool: True if a line with "Personnage" is write in the line, or written before.
    """
    return characterBlock or re.search("<strong><em>Personnages</em></strong>", line) or re.search(
        '<p align="center" style="text-align:center"><b><i><span style="letter-spacing:-.3pt">Personnages</span></i></b></p>',
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
        res = re.search("<h[1,2]", line)
        if res:
            characterBlock = False
            print("Character list: " + str(counters["characterIDList"]))
        else:
            res = re.search("<p>(.*)</p>", line)
            if res:
                name = res.group(1)
                if len(name) == 1:
                    if counters["characterIDList"]:
                        characterBlock = False
                        print("Character list: " + str(counters["characterIDList"]))
                    return characterBlock, None
                character = name
                res = re.search("([^,]+)(,.*)", character)
                if res:
                    character = remove_html_tags_and_content(res.group(1))
                    role = remove_html_tags_and_content(res.group(2))
                else:
                    character = remove_html_tags_and_content(character)
                    role = ""
                if len(character) > 2 and character != "\xa0":
                    counters["characterFullNameList"].append(character)
                    clean_character_name = normalize_character_name(character)
                    counters["characterIDList"].append(clean_character_name)
                    counters["roleList"].append(role)
    return characterBlock, line


def write_character(outputFile):
    """Write the saved characters of a play in the associated XML file.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
    """
    outputFile.writelines("""
        <castList>
                    <head> ACTEURS </head>""")
    for i, character in enumerate(counters["characterIDList"]):
        outputFile.writelines(f"""
            <castItem>
                    <role corresp="#{character}">{counters["characterFullNameList"][i]} </role>{counters["roleList"][i]}</castItem>"""
                              )
    outputFile.writelines("""
        </castList>""")


def find_begin_act(outputFile, line, counters):
    """Try to find the begin of an act in a play and convert it in the XML file associated if it find it.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        line (str): line to read.
        counters (dict): Dictionnary with all the counters of the script.

    Returns:
        tuple: the line (str) and the refreshed counter (dict).
    """
    res = re.search("<h1[^>]*>(.*ACTE.*)</h1>", line)
    if res:
        # Found a new act!
        if counters["actsInPlay"] == 0:
            write_character(outputFile)
            counters["castWritten"] = True
            outputFile.writelines("""
    </front>
    <body>""")
        else:
            print(str(counters["actsInPlay"]) + " acts so far")
            # end the previous scene of the previous act
            outputFile.writelines("""
   </sp>
   </div>""")
        counters["actsInPlay"] += 1
        counters["scenesInAct"] = 0
        act = res.group(1).replace("<strong>", "").replace("</strong>", "")
        res = re.search("ACTE (.*)", act)
        if res:
            counters["actNb"] = res.group(1)
        else:
            counters["actNb"] = act.replace(" ", "-").lower()
        outputFile.writelines("""
   </div>
   <div type="act" xml:id=\"""" + counters["actNb"] + """\">
   <head>""" + act + """</head>""")

    return line, counters


def find_begin_scene(outputFile, line, counters):
    """Try to find the begin of a scene in a play and convert it in the XML file associated if it find it.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        line (str): line to read.
        counters (dict): Dictionnary with all the counters of the script.

    Returns:
        tuple: the line (str) and the refreshed counter (dict).
    """
    res = re.search("<h2 .*<strong>(.*)</strong></h2>", line)
    if res:
        if not counters["castWritten"]:
            write_character(outputFile)
            counters["castWritten"] = True
        counters["characterLines"] = []
        counters["charactersInScene"] = 0
        scene = res.group(1)

        res = re.search("Scène (.*)", scene)
        if res:
            counters["sceneNb"] = res.group(1)
        else:
            counters["sceneNb"] = scene.replace(" ", "-").lower()
        counters["scenesInAct"] += 1
        if counters["scenesInAct"] == 1:
            outputFile.writelines("""
    <div type="scene" xml:id=\"""" + counters["actNb"] + str(counters["scenesInAct"]) + """\">
        <head>""" + scene + """</head>""")
        else:
            outputFile.writelines("""
      </sp>
   </div>
   <div type="scene" xml:id=\"""" + counters["actNb"] + str(counters["scenesInAct"]) + """\">
      <head>""" + scene + """</head>""")
    return line, counters


def find_character(line, counters):
    """Find a character name in a line from a play text and stock it in the counters dict.

    Args:
        line (str): line to read in the play.
        counters (dict): Dictionnary with all the counters of the script.

    Returns:
        dict: The refreshed counter
    """
    res = re.search("<p align=.center.>(.*)</p>", line)
    if res and res.group(1) != "\xa0":
        counters["characterLines"].append(res.group(1))
    return counters


def write_text(outputFile, line, counters):
    """Write the text from a HTML file's line in the XML associated file.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        line (str): line to read in the play.
        counters (dict): Dictionnary with all the counters of the script.

    Returns:
        dict: The refreshed counter
    """
    res = re.search("<p>(.*)</p>", line)
    if res and not characterBlock:
        playLine = res.group(1).replace("\xa0", " ")
        if playLine != " ":
            if len(counters["characterLines"]) > 1:
                character = counters["characterLines"].pop(0)
                outputFile.writelines("""
        <stage>""" + character + """</stage>""")
            if len(counters["characterLines"]) > 0:
                if counters["charactersInScene"] > 0:
                    outputFile.writelines("""
      </sp>""")
                character = counters["characterLines"].pop(0)
                counters["charactersInScene"] += 1

                # character is the actual character name, from which we strip html tags.
                # clean_character will be used to get the corresponding id
                character = remove_html_tags_and_content(character)
                clean_character = character
                # Checking if the character name is preceded by a comma, indicating an action on stage Dracor
                # convention seems to be to include it as a content of the speaker tag and not in <stage>,
                # so we follow this rule
                has_stage_direction = re.search("([^,]+),.*", clean_character)
                if has_stage_direction:
                    clean_character = has_stage_direction.group(1)
                # Removing ending dot if it exists
                if clean_character[-1] == ".":
                    clean_character = clean_character[:-1]
                characterId = normalize_character_name(clean_character)

                # even when normalizing character names, we often find ids that are not declared
                # this part aims at correcting that by checking if the name is part of a known id,
                # or if a known id is part of it
                # If everything fails, (which happen often), we use edit distance to find the closest one
                old_characterId = characterId
                if characterId not in counters['characterIDList']:
                    if characterId not in counters["undeclaredCharacterIDs"]:
                        print(f"Warning : unknown character id {characterId}")
                        edit_distances = dict()
                        for true_id in counters["characterIDList"]:
                            if re.search(true_id, characterId) or re.search(characterId, true_id):
                                counters["undeclaredCharacterIDs"][characterId] = true_id
                                characterId = true_id
                                print(f"Guessed {true_id}")
                                break
                            else:
                                distance = enchant.utils.levenshtein(characterId, true_id)
                                edit_distances[true_id] = distance
                        # characterID has not been guessed with subchains
                        if old_characterId not in counters["undeclaredCharacterIDs"]:
                            closest_id, closest_distance = min_dict(edit_distances)
                            if closest_distance <= 15:
                                print(f"Guessed {closest_id}, distance {closest_distance} ")
                                counters["undeclaredCharacterIDs"][characterId] = closest_id
                            else:
                                print(f"Could not guess {characterId} (best guess {closest_id}")
                                counters["undeclaredCharacterIDs"][characterId] = characterId
                                counters["unguessed_id"] = True
                    else:
                        characterId = counters["undeclaredCharacterIDs"][characterId]
                # if clean_character in [counters["characterFullNameList"]]:
                #     characterIdindex = [counters["characterFullNameList"].index(clean_character)]
                #     characterId = counters["characterIDList"][characterIdindex]
                # for c in counters["characterFullNameList"]:
                #     if re.search(c, clean_character):
                #         characterId = c
                # if characterId == "":
                #     print(line)
                #     print("entering characterId if")
                #     # print("Character not found: " + character)
                #     res = re.search("([^,.<]+)([.,<].*)", character)
                #     if res:
                #         characterId = res.group(1).lower()
                #         # remove spaces in last position
                #         res = re.search("^(.*[^ ])[ ]+$", characterId)
                #         if res:
                #             characterId = res.group(1)
                #         characterId = characterId.replace(" ", "-")
                #         # print("Chose characterId " + characterId)
                outputFile.writelines(f"""
            <sp who=\"#{characterId}\" xml:id=\"{counters["actNb"] + str(counters["scenesInAct"]) + "-" + str(
                    counters["charactersInScene"])}\">
                <speaker> {character} </speaker>""")

            # Checking whether this line is dialogue or stage directions (Aaron)
            res = re.search("<em>(.*)</em>", playLine)
            if res:
                outputFile.writelines(f"""
            <stage>{remove_html_tags_and_content(playLine)} </stage>""")
            else:
                outputFile.writelines("""
            <l n=\"""" + str(counters["linesInPlay"]) + """\" xml:id=\"l""" + str(
                    counters["linesInPlay"]) + """\">""" + remove_html_tags_and_content(playLine) + """</l>""")
                counters["linesInPlay"] += 1
                counters["linesInScene"] += 1

    return counters


def write_end(outputFile):
    """Write the end of the XML output file.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
    """
    outputFile.writelines("""
         </sp>
         <p>FIN</p>
      </div>
      </div>
   </body>
</text>
</TEI>""")


if __name__ == "__main__":

    #stats temporary
    castnotWritten = 0
    noact = 0
    undeclared_character = 0
    unguessed_character = 0
    totalplays = 0
    stats = open('stats_characters.txt', 'w+')
    # Declaration of flags and counters.
    documentNb = 0
    findSummary = False
    saveBegin = False
    characterBlock = False
    ul = 0

    # prepare the list of file sources
    fileSources = {}
    allPlays = extract_sources(open("PlaysFromTheatreDocumentation.csv", "r", encoding="utf-8"), fileSources)
    # Generate an XML-TEI file for every HTML file of the corpus
    for file in list(map(lambda f: join(html_folder, f), next(walk(html_folder), (None, None, []))[2])):
        notify_file(file)

        # Find source
        fileName = basename(file)
        source = get_source(fileSources, fileName)

        playText = open(file, "r", encoding="utf-8")
        outputFile = open(join(Dracor_Folder, fileName.replace("html", "xml")), "w", encoding="utf-8")

        # reset parameters

        counters = {
            "charactersInScene": 0,
            "linesInPlay": 0,
            "linesInScene": 0,
            "scenesInAct": 0,
            "actsInPlay": 0,
            "characterLines": [],
            "characterIDList": [],
            "characterFullNameList": [],
            "roleList": [],
            "actNb": "",
            "sceneNb": "",
            "dedicace": False,
            "castWritten": False,
            "undeclaredCharacterIDs": dict(),
            "unguessed_id" : False #temporary, delete later
        }
        # Reading the file a first time to find the characters

        for line in playText:  # TODO: add a break when all characters are found
            # starting character block
            characterBlock = start_character_block(line, characterBlock)

            # ending character block
            characterBlock, line = end_character_block(characterBlock, line)
        playText.close()
        playText = open(file, "r", encoding="utf-8")
        for line in playText:

            # get and write title
            title, forename, surname = get_title_and_author(line)
            if write_title(outputFile, title):
                # get and write type of play:
                copy_playtext = open(file, "r", encoding="utf-8")
                genre, vers_prose = get_type(copy_playtext)
                write_type(outputFile, genre)
                # get and write author
                author = forename, surname
                if write_author(outputFile, author):
                    # get and write source
                    write_source(outputFile, source)

                # get and write date
                copy_playtext.close()
                copy_playtext = open(file, "r", encoding="utf-8")
                date_written, date_print, date_premiere, line_written, line_print, line_premiere = get_dates(
                    copy_playtext)

                write_dates(outputFile, date_written, date_print, date_premiere, line_premiere)

                write_end_header(outputFile, genre, vers_prose)
                write_start_text(outputFile, title, genre, date_print)

                write_performance(outputFile, line_premiere, date_premiere)

            # try find dedicace in play
            if not findSummary:
                findSummary = find_summary(line, ul)
            else:
                findSummary = extract_from_summary(line, ul)

            # starting saving lines
            if not saveBegin:
                saveBegin = try_saving_lines(outputFile, line)
            else:
                # find and print dedicace
                if counters['dedicace']:
                    if find_dedicace(line):
                        copy_playtext.close()
                        copy_playtext = open(file, "r", encoding="utf-8")
                        write_dedicace(outputFile, copy_playtext, author)

            # Find the beginning of an act and write cast list (characters) TODO: split in 2 functions
            line, counters = find_begin_act(outputFile, line, counters)

            # Find the beginning of a scene
            line, counters = find_begin_scene(outputFile, line, counters)

            # Find the list of characters on stage
            counters = find_character(line, counters)

            # Write the list of characters on stage
            counters = write_text(outputFile, line, counters)

        write_end(outputFile)

        outputFile.close()
        copy_playtext.close()

        if len(counters["undeclaredCharacterIDs"]) == 0:
            os.replace(join(Dracor_Folder, fileName.replace("html", "xml")), join(clean_Dracor_Folder, fileName.replace("html", "xml")))

        if len(counters["characterIDList"]) == 0:
            castnotWritten += 1
        if counters["actsInPlay"] == 0:
            noact += 1
        if len(counters["undeclaredCharacterIDs"]) > 0:
            undeclared_character += 1
        if counters["unguessed_id"]:
            unguessed_character += 1
        totalplays += 1


    # date_file.close()

    #  print("Number of plays without date :", count_date)
    stats.writelines(f"""Total number of plays : {totalplays}
    Plays with no acts found : {noact}
    Plays with no cast of character found : {castnotWritten}
    Plays with unknow character ids found : {undeclared_character}
    Among those, plays where at least one character could not be guessed : {unguessed_character}""")
    print(f"Casts not written: {castnotWritten} sur {totalplays}")