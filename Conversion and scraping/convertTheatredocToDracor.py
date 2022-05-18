#!/usr/sfw/bin/python
# -*- coding: utf-8 -*-

from genericpath import exists
import glob, os, re, sys, time, requests, subprocess
from operator import is_
from os import walk, pardir
from os.path import abspath, dirname, join, basename


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
html_folder = abspath(join(root_folder, "notConvertTD"))
Dracor_Folder = abspath(join(root_folder, "corpusDracor"))

### temporaire
date_file = open('datesTD.txt', 'w')
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

documentNb = 0
saveBegin = False
characterBlock = False

# prepare the list of file sources
fileSources = {}
allPlays = open("PlaysFromTheatreDocumentation.csv", "r", encoding="utf-8")
maxFileNameLength = 0
for playLine in allPlays:
   res = re.search("([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\n", playLine)
   if res:
      fileSources[res.group(1)] = res.group(2)
      

# Generate an XML-TEI file for every HTML file of the corpus
for file in list(filter(lambda f: ".html" in f, map(lambda f: join(html_folder, f), next(walk(html_folder), (None, None, []))[2]))):
   # if exists(join(Dracor_Folder, file.replace("html", "xml"))):
   #    continue
   print("Converting file " + file)
   date_file.writelines(file + "\n")
   
   # Find source
   source = ""
   fileName = basename(file)
   
   if fileName in fileSources:
      source = fileSources[fileName]
      
   
   playText = open(file, "r", encoding="utf-8")
   outputFile = open(join(Dracor_Folder, file.replace("html", "xml")), "w", encoding="utf-8")

   # reset parameters
   charactersInScene = 0
   linesInPlay = 0
   linesInScene = 0
   scenesInAct = 0
   actsInPlay = 0
   characterLines = []
   characterList = []
   actNb = ""
   sceneNb = ""

   for line in playText:
      title = ""
      author = ""
      
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

         
         outputFile.writelines("""<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="fre">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title type="main">""" + title + """</title>""")

         outputFile.writelines("""
        <author>
         <persName>""")

         for name in forename:
            if name in ['de', "d'"]:
               outputFile.writelines("""
               <linkname>""" + name + """</linkname>""")
            elif name in ['Abbé']: # identifier les rolename
               outputFile.writelines("""
               <rolename>""" + name + """</rolename>""")
            else:
               outputFile.writelines("""
               <forename>""" + name + """</forename>""")
         
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

         outputFile.writelines("""
      <publicationStmt>
        <publisher xml:id="dracor">DraCor</publisher>
        <idno type="URL">https://dracor.org</idno>
                <idno type="dracor" xml:base="https://dracor.org/id/">fre000383</idno>
        </availability>
        <idno>""" + source + """</idno>
      </publicationStmt>
      <sourceDesc>""")

         outputFile.writelines("""
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

         date_written = "[date]"
         date_print = "[date]"
         date_premiere = "[date]"
         is_written = False
         is_print = False
         is_premiere = False

         for l in playText:
            date_line = re.search("<p>(Représentée.*)</p>", l)
            if date_line:
               date_line = date_line.group(1)
               res = re.search(".*le ([0-9]+|1<sup>er</sup>) ([^ ]+) ([0-9]+).*", date_line)
               print(date_line)
               date_file.writelines(date_line + "\n")
               if res and res.group(1) and res.group(2) and res.group(3):
                  day = res.group(1).strip('<sup>er</sup>')
                  if len(day) == 1:
                     day = '0' + day
                  date_premiere = '-'.join(
                     [res.group(3),
                     mois[res.group(2).lower().replace('é', 'e').replace('août', 'aout').replace('levrier', 'fevrier')], 
                     day
                  ])
                  is_premiere = True
               break
            else:
               date_line = ""
               if not is_print:
                  res = re.search("<p>([0-9]+).</p>", l)
                  if res:
                     date_print = res.group(1)
                     date_file.writelines(date_print + ".\n")
                     is_print = True
                     break



         print(date_premiere)
         if not (is_written or is_print or is_premiere):
            count_date+=1
            outputFile.writelines("[date]...\n\n")
         

         if is_print:
            date_file.writelines(date_print + "\n\n")
            outputFile.writelines("""
            <date type="print" when=\"""" + date_print + """\">""")

         if is_premiere:
            date_file.writelines(date_premiere + "\n\n")
            outputFile.writelines("""
                        <date type="premiere" when=\"""" + date_premiere + """\">""" + date_line + """</date>""")
         
         outputFile.writelines("""
                        <idno type="URL"/>
                    </bibl>
                </bibl>
         """)





         outputFile.writelines("""
         </sourceDesc>
    </fileDesc>
    <profileDesc>
      <creation>
        <date when="[date]">[date]</date>
      </creation>
      <langUsage>
        <language ident="fre"/>
      </langUsage>
      <textClass>
        <keywords>
          <term subtype="tragedy" type="genre">Tragédie</term>
        </keywords>
      </textClass>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <head>""" + title + """</head>
      <div type="set">
        <div>
          <head>PERSONNAGES</head>
          <castList>
""")

      # starting saving lines
      if not saveBegin:
         res = re.search("<p>(.*)</p>", line)
         if res:
            saveBegin = True
            outputFile.writelines("<p>" + res.group(1) + "</p>\n")
      else:
         # starting character block
         res = re.search("<strong><em>Personnages</em></strong>", line)
         res2 = re.search('"<p align="center" style="text-align:center"><b><i><span style="letter-spacing:-.3pt">Personnages</span></i></b></p>"', line)
         if res or res2: 
            characterBlock = True
         
         # ending character block
         if characterBlock:
            res = re.search("<h[1,2]", line)
            if res:
               characterBlock = False
               print("Character list: " + str(characterList))
            else:
               res = re.search("<p>(.*)</p>", line)
               if res:
                  name = res.group(1)
                  if len(name) == 1:
                     if characterList:
                        characterBlock = False
                        print("Character list: " + str(characterList))
                     continue   
                  character = res.group(1)
                  role = ""
                  res = re.search("([^,]+)(,.*)", character)
                  if res:
                     character = res.group(1)
                     role = res.group(2)
                  if len(character) > 2 and character != "&nbsp;":
                     characterList.append(character.lower().replace("*","").replace(" ","-"))
                     outputFile.writelines("""
            <castItem>
              <role rend="male/female" xml:id=\"""" + character.lower().replace(" ","-") + """\">""" + character + """</role>
              <roleDesc>""" + role + """</roleDesc>
            </castItem>
""")  
      
      # Find the beginning of an act
      res = re.search("<h1[^>]*>(.*ACTE.*)</h1>", line)
      if res:
         # Found a new act!
         if actsInPlay == 0:
            outputFile.writelines("""
          </castList>
        </div>""")
         else: 
            print(str(actsInPlay) + " acts so far")
            # end the previous scene of the previous act
            outputFile.writelines("""
        </sp>
      </div>""")
         actsInPlay += 1
         scenesInAct = 0
         act = res.group(1).replace("<strong>", "").replace("</strong>", "")
         res = re.search("ACTE (.*)", act)
         if res:
            actNb = res.group(1)
         else:
            actNb = act.replace(" ","-").lower()
         outputFile.writelines("""
      </div>
      <div type="act" xml:id=\"""" + actNb + """\">
        <head>""" + act + """</head>""")
         
      # Find the beginning of a scene
      res = re.search("<h2 .*<strong>(.*)</strong></h2>", line)
      if res:
         characterLines = []
         charactersInScene = 0
         scene = res.group(1)
         res = re.search("Scène (.*)", act)
         if res:
            sceneNb = res.group(1)
         else:
            sceneNb = scene.replace(" ","-").lower()
         scenesInAct += 1
         if scenesInAct == 1:
            outputFile.writelines("""
        <div type="scene" xml:id=\"""" + actNb + str(scenesInAct) + """\">
          <head>""" + scene + """</head>""")
         else:
            outputFile.writelines("""
          </sp>
        </div>
        <div type="scene" xml:id=\"""" + actNb + str(scenesInAct) + """\">
          <head>""" + scene + """</head>""")
         #sceneNb += 1

      # Find the list of characters on stage
      res = re.search("<p align=.center.>(.*)</p>", line)
      if res and res.group(1) != "&nbsp;":
         characterLines.append(res.group(1))

      # Find the list of characters on stage
      res = re.search("<p>(.*)</p>", line)
      r = res
      if res and not characterBlock:
         playLine = res.group(1).replace("&nbsp;"," ")
         if playLine != " ":
            if len(characterLines) > 1:
               character = characterLines.pop(0)
               outputFile.writelines("""
          <stage>""" + character + """</stage>""")
            if len(characterLines) > 0:
               if charactersInScene > 0:
                  outputFile.writelines("""
          </sp>""")
               character = characterLines.pop(0)
               charactersInScene += 1
               # find the character name among all characters
               characterId = ""
               for c in characterList:
                  try:
                     res = re.search(c, character.lower())
                  except re.error:
                     raise ValueError(f"Character : {c}\nList : {characterList}\nLigne courante : {r.group(1)}")
                  if res:
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
          <sp who=\"""" + characterId + """\" xml:id=\"""" + actNb + str(scenesInAct) + "-" + str(charactersInScene) + """\">
            <speaker>""" + character + """</speaker>""")
            linesInPlay += 1
            res = re.search("<em>(.*)</em>", playLine)
            if res:
               outputFile.writelines("""
            <stage><hi rend=\"italic\">""" + playLine + """</hi></stage>""")            
            else:
               outputFile.writelines("""
            <l n=\"""" + str(linesInPlay) + """\" xml:id=\"l""" + str(linesInPlay) + """\">""" + playLine + """</l>""")
            linesInScene += 1
          
   outputFile.writelines("""
          </sp>
          <p>FIN</p>
        </div>
      </div>
    </body>
  </text>
</TEI>""")
   outputFile.close()


date_file.close()

print(count_date)