#!/usr/sfw/bin/python
# -*- coding: utf-8 -*-

import glob, os, re, sys, time, requests, subprocess

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
folder = os.path.abspath(os.path.dirname(sys.argv[0]))

documentNb = 0
saveBegin = False
characterBlock = False

# prepare the list of file sources
fileSources = {}
titles = {}
authors = {}
allPlays = open("PlaysFromEmothe.csv", "r", encoding="utf-8")
maxFileNameLength = 0
for playLine in allPlays:
   res = re.search("([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\n", playLine)
   if res:
      fileSources[res.group(1)] = res.group(2)
      titles[res.group(1)] = res.group(3)
      authors[res.group(1)] = res.group(4)
print(fileSources)
      

# Generate an XML-TEI file for every HTML file of the corpus
for file in glob.glob(os.path.join(os.path.join(folder, "corpusE"),"*.php")):
   print("Converting file " + file)
   
   # Find source
   source = ""
   title = ""
   author = ""
   fileName = os.path.basename(file)
   
   if fileName in fileSources:
      source = fileSources[fileName]
      title = titles[fileName]
      author = authors[fileName]
   
   # Put together lines with blockings and verses
   playText = open(file, "r", encoding="utf-8")
   groupMode = False
   playTextLines = []
   for line in playText:
      if groupMode:
         res = re.search("^[ ]*</div>[ ]*$", line)
         if res:
            playTextLines.append(lineGroup + "</div>")
            groupMode = False
         else:
            res = re.search("^[ ]*([^\r\n]*)[\r\n]*$", line)
            if res:
               lineGroup += res.group(1)
      else:
         res = re.search("(<div class=.verso. name=.verso.>[^<\r\n]*)[\r\n]*$", line)
         if res:
            lineGroup = res.group(1)
            groupMode = True
         else:
            playTextLines.append(line)
   playText.close()
   
   outputFile = open(file+".xml", "w", encoding="utf-8")

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

   for line in playTextLines:
      res = re.search("<title>(.*)</title>", line.replace(")",""))
      if res:
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
        <edition>Edition initialement mise à disposition sur emothe.uv.es au format HTML, convertie en XML-TEI dans le cadre de cadre du projet Hyperpièces avec Céline Fournial et du stage de master 2 d'Aaron Boussidan au LIGM.</edition>
        <respStmt>
          <name>Michel Capus</name>
          <resp>Edition de la pièce au format HTML sur emothe.uv.es</resp>
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
      if not(saveBegin):
         res = re.search("<div id=\"elenco\">", line)
         if res:
            saveBegin = True
      else:
         # starting character block
         res = re.search("<table>", line)
         if res: 
            characterBlock = True
         
         # ending character block
         if characterBlock:
            res = re.search("</table>", line)
            if res:
               characterBlock = False
            else:
               res = re.search("<td>(.*)</td>", line)
               if res:
                  character = res.group(1)
                  role = ""
                  res = re.search("([^,]+)(,.*)", character)
                  if res:
                     character = res.group(1)
                     role = res.group(2)
                  if len(character)>2 and character != "&nbsp;":
                     characterList.append(character.lower().replace("*","").replace("[","").replace("]","").replace(" ","-"))
                     outputFile.writelines("""
            <castItem>
              <role rend="male/female" xml:id=\"""" + character.lower().replace(" ","-") + """\">""" + character + """</role>
              <roleDesc>""" + role + """</roleDesc>
            </castItem>
""")
      
      # Find the beginning of an act
      res = re.search("<h2 class=.tituloActo.>(.*ACTE.*)</h2>", line)
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
      res = re.search("<h3 class=.tituloEscena.>(.*)</h3>", line)
      if res:
         characterLines = []
         charactersInScene = 0
         scene = res.group(1)
         res = re.search("SCÈNE (.*)", scene)
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
      res = re.search("<div class=.speaker.>(.*)</div>", line)
      if res:
         character = res.group(1).replace("&nbsp;"," ")
         if charactersInScene > 0:
            outputFile.writelines("""
          </sp>""")
         charactersInScene += 1
         # find the character name among all characters
         characterId = ""
         for c in characterList:
            res = re.search(c, character.lower())
            if res:
               characterId = c
         if characterId == "":
            #print("Character not found: " + character)                  
            res = re.search("([^,.<]+)([.,<].*)", character)
            if res:
               characterId = res.group(1).lower()
            else:
               characterId = character.lower()   
            # remove spaces in last position
            res = re.search("^(.*[^ ])[ ]+$", characterId)
            if res:
               characterId = res.group(1)
            characterId = characterId.replace(" ","-")
         outputFile.writelines("""
          <sp who=\"""" + characterId + """\" xml:id=\"""" + actNb + str(scenesInAct) + "-" + str(charactersInScene) + """\">
            <speaker>""" + character + """</speaker>""")
         
      res = re.search("^[ ]*<div class=\"acotacionInterna\" name=\"acotacionInterna\">(.*)</div>[ ]*$", line)
      if res:
         outputFile.writelines("""
            <stage><hi rend=\"italic\">""" + res.group(1) + """</hi></stage>""")            
         
      res = re.search("^[ ]*<div class=.verso. name=.verso.>(.*)</div>[ ]*$", line)
      if res:
         currentLine = res.group(1)
         res = re.search("^(.*)<div class=\"acotacionInterna\" name=\"acotacionInterna\">(.*)</div>(.*)$", currentLine)
         if res:
            outputFile.writelines("""
            <l n=\"""" + str(linesInPlay) + """\" xml:id=\"l""" + str(linesInPlay) + """\">""" + res.group(1)
            + """<stage><hi rend=\"italic\">""" + res.group(2) + """</hi></stage>"""
            + res.group(3) + """</l>""")
         else:
            outputFile.writelines("""
            <l n=\"""" + str(linesInPlay) + """\" xml:id=\"l""" + str(linesInPlay) + """\">""" + currentLine + """</l>""")
         linesInScene += 1
         linesInPlay += 1
         
          
   outputFile.writelines("""
          </sp>
          <p>FIN</p>
        </div>
      </div>
    </body>
  </text>
</TEI>""")
   outputFile.close()
   