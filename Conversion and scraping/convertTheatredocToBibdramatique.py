#!/usr/sfw/bin/python
# -*- coding: utf-8 -*-

import glob, os, re, sys, time, requests, subprocess
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
TD_folder = abspath(join(root_folder, "corpusTD"))

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
for file in list(filter(lambda f: ".html" in f, map(lambda f: join(TD_folder, f), next(walk(TD_folder), (None, None, []))[2]))):
   print("Converting file " + file)
   
   # Find source
   source = ""
   fileName = basename(file)
   
   if fileName in fileSources:
      source = fileSources[fileName]
      
   
   playText = open(file, "r", encoding="utf-8")
   outputFile = open(file.replace("html", "xml"), "w", encoding="utf-8")

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
            author = res2.group(2)
         
         outputFile.writelines("""<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="../Teinte/tei2html.xsl"?>
<?xml-model href="http://oeuvres.github.io/Teinte/teinte.rng" type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>
<?xml-model href="bibdramatique.sch" type="application/xml" schematypens="http://purl.oclc.org/dsdl/schematron"?>
<TEI xml:lang="fr" xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>""" + title + """</title>
        <author key=\"""" + author +""" (naissance-mort)\">""" + author +"""</author>
      </titleStmt>
      <editionStmt>
        <edition>Edition initialement mise à disposition sur théâtre-documentation.com au format HTML, convertie en XML-TEI dans le cadre de cadre du projet Hyperpièces avec Céline Fournial et du stage de master 2 d'Aaron Boussidan au LIGM.</edition>
        <respStmt>
          <name>Michel Capus</name>
          <resp>Edition de la pièce au format HTML sur théâtre-documentation.com</resp>
        </respStmt>
        <respStmt>
          <name>Philippe Gambette</name>
          <resp>Conversion du code HTML vers XML/TEI</resp>
        </respStmt>
      </editionStmt>
      <publicationStmt>
        <publisher>LIGM</publisher>
        <date when="2021"/>
        <availability status="free">
          <p>In the public domain</p>
        </availability>
        <idno>""" + source + """</idno>
      </publicationStmt>
      <sourceDesc>
        <bibl><author>""" + author + """</author>. <title>""" + title + """</title>. </bibl>
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
         if res: 
            characterBlock = True
         
         # ending character block
         if characterBlock:
            res = re.search("<h[1,2]", line)
            if res:
               characterBlock = False
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
            print("Character list: " + str(characterList))
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
   