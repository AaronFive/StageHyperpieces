#!/usr/sfw/bin/python
# -*- coding: utf-8 -*-

import os, re, sys
from os import walk, pardir
from os.path import abspath, dirname, join, basename, exists


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

if not exists(Dracor_Folder):
   os.system("mkdir {0}".format(Dracor_Folder))

### temporaire
date_file = open(join(root_folder, 'datesTD.txt'), 'w')
count_date = 0
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
clean_genre = list(map(lambda g: g[0].upper() + g.replace('e', 'é')[1:-1] + g[-1] , genres))

def good_genre(genre):
    return clean_genre[genres.index(genre)]

def format_date_AAAAMMJJ(res):
    day = res[0].replace('<sup>er</sup>', '')
    if len(day) == 1:
        day = '0' + day
    return '-'.join(
        [res[2].replace('l', '1').replace('|', '1'),
        mois[res[1].lower().replace('é', 'e').replace('août', 'aout').replace('levrier', 'fevrier').replace('fevier', 'fevrier')], 
        day.replace('l', '1').replace('|', '1').replace('premier', '01')
    ])

def format_date_AAAAMM(res):
    return '-'.join(
        [res[1],
        mois[res[0].lower().replace('é', 'e').replace('août', 'aout').replace('levrier', 'fevrier').replace('fevier', 'fevrier')]
    ])

def standard_line(playText):
    return list(map(lambda l: l.replace("<span style=\"letter-spacing:-.3pt\">", "").replace("\xa0", ' ').replace('<a href="#_ftn1" name="_ftnref1" title="" id="_ftnref1">[1]</a>', '').strip('\n'), playText))

def extract_sources(allPlays):
    for playLine in allPlays:
        res = re.search("([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\n", playLine)
        if res:
            fileSources[res.group(1)] = res.group(2)
    return allPlays

def notify_file(file):
    print("Converting file " + file)
    date_file.writelines(basename(file).replace(".html", '') + "\t")

def get_source(fileSources, fileName):
    if fileName in fileSources:
        return fileSources[fileName]
    return ""

def get_title_and_author(line):
    title = ""
    author = ""
    persNames = ""
    forename = ""
    surname = ""
   
    res = re.search("<title>(.*) | théâtre-documentation.com</title>", line.replace(")",""))
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
    if title:   
        outputFile.writelines("""<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="fre">
    <teiHeader>
        <fileDesc>
            <titleStmt>
            <title type="main">""" + title + """</title>""")
    return title

def get_type(playText):
    for l in standard_line(playText):
        res = re.search('<p>(.*)</p>', l)
        if res:
            content = res.group(1).lower().replace('é', 'e')
            if content == ' ':
                break
            for genre in genres:
                if genre in content:
                    return good_genre(genre) 
    return '[indéfini]'

def write_type(outputFile, genre):
    if genre != '[indéfini]':
        outputFile.writelines("""
            <title type="sub">""" + genre + """</title>""")
            

def write_author(outputFile, author):
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
                elif name in ['Abbé']: # TODO identifier les rolename 
                    outputFile.writelines("""
                    <rolename>""" + name + """</rolename>""")
                else:
                    outputFile.writelines("""
                    <forename>""" + name + """</forename>""")
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
    outputFile.writelines("""
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
                    <idno type="URL">""" + source + """</idno>
                    <availability>
                        <licence>
                            <ab>loi française n° 92-597 du 1er juillet 1992 et loi n°78-753 du 17 juillet 1978</ab>
                            <ref target="http://théâtre-documentation.com/content/mentions-l%C3%A9gales#Mentions_legales">Mentions légales</ref>
                        </licence>
                    </availability>
                    <bibl type="originalSource">
            """)

def get_dates(playText):
    global count_date
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

        if re.search(".*<strong><em>Personnages.*</em></strong>.*", l) or re.search('<p align="center" style="text-align:center"><b><i>Personnages.*</span></i></b></p>', l) or (True in (is_written, is_print, is_premiere) and l == '<p> </p>'):
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
            res = re.search("<p>Publié.* ([0-9]+) et représenté.* ([0-9]+|1<sup>er</sup>|premier) ([^ ]+) ([0-9]+).*</p>", l)
            res2 = re.search("<p>Publié.* ([0-9]+) et représenté.* ([0-9]+).*</p>", l)
            if res or res2:
                is_print, is_premiere = True, True
                if res:
                    date_print, date_premiere = res.group(1), format_date_AAAAMMJJ(res.groups()[1:])

                elif res2:
                    date_print, date_premiere = res2.group(1), res2.group(2)
                is_print, is_premiere = True, True
                line_print, line_premiere = l.replace("<p>", "").replace("</p>", ""), l.replace("<p>", "").replace("</p>", "")

        date_line = re.search("<p>.*([Rr]eprésenté.*)</p>", l)
        date_line2 = re.search("<p>.*(fut joué.*)</p>", l)
        if (date_line or date_line2) and not is_premiere:
            if date_line2:
                date_line = date_line2
            date_line = date_line.group(1)
            res = re.search(".* ([l\|]?[0-9]+|1<sup>er</sup>|premier)[ ]+([^ ]+) ([l\|]?[0-9]+).*", date_line)
            res2 = re.search(".* ([0-9]+|1<sup>er</sup>|premier)[ ]+([^ ]+) ([0-9]+).*" * 2, date_line)
            double_words_res = re.search(".* ([l\|]?[0-9]+|1<sup>er</sup>|premier)[ ]+([^ ]+)[ ]+([^ ]+) ([l\|]?[0-9]+).*", date_line)
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
            res3 = re.search("<p>Non représentée[,\.] ([0-9]+).*</p>", l.replace('<a href="#_ftn1" name="_ftnref1" title="" id="_ftnref1">[1]</a>', ''))

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

    if not (is_print or is_premiere or is_written):
        count_date += 1

    if not is_written:
        line_written = "[vide]"

    date_file.writelines(line_written + '\t' + line_print + '\t' + line_premiere + '\t')

    date_file.writelines(date_written + '\t')
   
    date_file.writelines(date_print + '\t')
   
    date_file.writelines(str(date_premiere) + "\n")

    return date_written, date_print, date_premiere, line_written, line_print, line_premiere

def write_dates(outputFile, date_written, date_print, date_premiere, line_premiere):
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
                        <date type="premiere" notBefore=\"""" + date_premiere[0] + """\" notAfter=\"""" + date_premiere[1] + """\" >""" + line_premiere + """</date>""")
   
    outputFile.writelines("""
                        <idno type="URL"/>
                    </bibl>
                </bibl>
   """)

def write_end_header(outputFile, genre):
    outputFile.writelines("""
            </sourceDesc>
        </fileDesc>
        <profileDesc>
            <particDesc>
                <listPerson>
                    <person xml:id="id" sex="SEX">
                        <persName>Name</persName>
                    </person>
                    <person xml:id="..." sex="...">
                        <persName>...</persName>
                    </person>
                 </listPerson>
            </particDesc>
            <textClass>
            <keywords scheme="http://theatre-documentation.fr"> <!--extracted from "genre" and "type" elements-->
                    <term>""" + genre + """</term>
                    <term>[vers, prose ?]</term>
                </keywords>
                <classCode scheme="http://www.wikidata.org/entity/">[QNumbers]</classCode>
            </textClass>
        </profileDesc>
        <revisionDesc>
            <listChange>
                <change when="2022-07-12">(mg) file conversion from source</change>
            </listChange>
        </revisionDesc>
   </teiHeader>""")

def write_start_text(outputFile):
    outputFile.writelines("""
    <text>
        <body>
            <head>""" + title + """</head>
            <div type="set">
            <div>
                <head>PERSONNAGES</head>
                <castList>
   """)

def try_saving_lines(outputFile):
    res = re.search("<p>(.*)</p>", line)
    if res:
        outputFile.writelines("<p>" + res.group(1) + "</p>\n")
    return bool(res)

def start_character_block(line, characterBlock):
    return characterBlock or re.search("<strong><em>Personnages</em></strong>", line) or re.search('<p align="center" style="text-align:center"><b><i><span style="letter-spacing:-.3pt">Personnages</span></i></b></p>', line)

def end_character_block(characterBlock, line, outputFile):
    if characterBlock:
        res = re.search("<h[1,2]", line)
        if res:
            characterBlock = False
            print("Character list: " + str(counters["characterList"]))
        else:
            res = re.search("<p>(.*)</p>", line)
            if res:
                name = res.group(1)
                if len(name) == 1:
                    if counters["characterList"]:
                        characterBlock = False
                        print("Character list: " + str(counters["characterList"]))
                    return characterBlock, None   
                character = name
                role = ""
                res = re.search("([^,]+)(,.*)", character)
                if res:
                    character = name
                    role = res.group(2)
                if len(character) > 2 and character != "\xa0":
                    counters["characterList"].append(character.lower().replace("*","").replace(" ","-"))
                    outputFile.writelines("""
        <castItem>
        <role rend="male/female" xml:id=\"""" + character.lower().replace(" ","-") + """\">""" + character + """</role>
        <roleDesc>""" + role + """</roleDesc>
        </castItem>
""") 
    return characterBlock, line

def find_begin_act(outputFile, line, counters):
    res = re.search("<h1[^>]*>(.*ACTE.*)</h1>", line)
    if res:
        # Found a new act!
        if counters["actsInPlay"] == 0:
            outputFile.writelines("""
      </castList>
   </div>""")
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
            counters["actNb"] = act.replace(" ","-").lower()
        outputFile.writelines("""
   </div>
   <div type="act" xml:id=\"""" + counters["actNb"] + """\">
   <head>""" + act + """</head>""")
    return line, counters

def find_begin_scene(outputFile, line, counters):
    res = re.search("<h2 .*<strong>(.*)</strong></h2>", line)
    if res:
        counters["characterLines"] = []
        counters["charactersInScene"] = 0
        scene = res.group(1)

        res = re.search("Scène (.*)", scene)
        if res:
            counters["sceneNb"] = res.group(1)
        else:
            counters["sceneNb"] = scene.replace(" ","-").lower()
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
    res = re.search("<p align=.center.>(.*)</p>", line)
    if res and res.group(1) != "\xa0":
        counters["characterLines"].append(res.group(1))
    return counters

def write_text(outputFile, line, counters):
    res = re.search("<p>(.*)</p>", line)
    if res and not characterBlock:
        playLine = res.group(1).replace("\xa0"," ")
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
                # find the character name among all characters
                characterId = ""
                for c in counters["characterList"]:
                    if re.search(c, character.lower()):
                        characterId = c
                if characterId == "":
                    #print("Character not found: " + character)                  
                    res = re.search("([^,.<]+)([.,<].*)", character)
                    if res:
                        characterId = res.group(1).lower()
                        # remove spaces in last position
                        res = re.search("^(.*[^ ])[ ]+$", characterId)
                        if res:
                            characterId = res.group(1)
                        characterId = characterId.replace(" ","-")
                        #print("Chose characterId " + characterId)
                    outputFile.writelines("""
        <sp who=\"""" + characterId + """\" xml:id=\"""" + counters["actNb"] + str(counters["scenesInAct"]) + "-" + str(counters["charactersInScene"]) + """\">
            <speaker>""" + character + """</speaker>""")
            counters["linesInPlay"] += 1
            res = re.search("<em>(.*)</em>", playLine)
            if res:
                outputFile.writelines("""
            <stage><hi rend=\"italic\">""" + playLine + """</hi></stage>""")            
            else:
                outputFile.writelines("""
            <l n=\"""" + str(counters["linesInPlay"]) + """\" xml:id=\"l""" + str(counters["linesInPlay"]) + """\">""" + playLine + """</l>""")
            counters["linesInScene"] += 1
    return counters

def write_end(outputFile):
    outputFile.writelines("""
         </sp>
         <p>FIN</p>
      </div>
      </div>
   </body>
</text>
</TEI>""")

if __name__ == "__main__":  
    # Declaration of flags and counters. 
    documentNb = 0
    saveBegin = False
    characterBlock = False
    # prepare the list of file sources
    fileSources = {}
    allPlays = extract_sources(open("PlaysFromTheatreDocumentation.csv", "r", encoding="utf-8"))  

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
            "charactersInScene" : 0,
            "linesInPlay" : 0,
            "linesInScene" : 0,
            "scenesInAct" : 0,
            "actsInPlay" : 0,
            "characterLines" : [],
            "characterList" : [],
            "actNb" : "",
            "sceneNb" : "",
        } 

        for line in playText:
            # get and write title
            title, forename, surname = get_title_and_author(line)
            if write_title(outputFile, title):
                # get and write type of play:
                copy_playtext = open(file, "r", encoding="utf-8")
                genre = get_type(copy_playtext)
                write_type(outputFile, genre)
                # get and write author
                author = forename, surname
                if write_author(outputFile, author):
                    # get and write source
                    write_source(outputFile, source)

                # get and write date
                copy_playtext.close()
                copy_playtext = open(file, "r", encoding="utf-8")
                date_written, date_print, date_premiere, line_written, line_print, line_premiere = get_dates(copy_playtext)

                write_dates(outputFile, date_written, date_print, date_premiere, line_premiere)

                write_end_header(outputFile, genre)
                write_start_text(outputFile)

            # starting saving lines
            if not saveBegin:
                saveBegin = try_saving_lines(outputFile)
            else:
                # starting character block
                characterBlock = start_character_block(line, characterBlock)
                
                # ending character block
                characterBlock, line = end_character_block(characterBlock, line, outputFile)
                if not line:
                    continue
                
            # Find the beginning of an act
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


    date_file.close()

    print("Number of plays without date :", count_date)
