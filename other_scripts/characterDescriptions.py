#!/usr/sfw/bin/python
# -*- coding: utf-8 -*-

import glob, os, re, sys, time, requests, subprocess
from xml.dom import minidom

"""
    characterDescriptions, a script to automatically extract character descriptions 
    from XML-TEI plays from theatre-classique.fr or bibdramatique.huma-num.fr
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

documentNb = 0

# Get the current folder
folder = os.path.abspath(os.path.dirname(sys.argv[0]))

corpusFolder = os.path.join(folder,"corpusTC")

# get the text inside the XML node
def displayNodeText(node, avoidedTags):
   if node.childNodes.length == 0:
      text = ""
      try:
         text = node.data
      except:
         pass
      return text
   else:
      text = ""
      for child in node.childNodes:
         if child.nodeName.lower() not in avoidedTags:
            text += displayNodeText(child, avoidedTags)
      return text

#outputFile = open(os.path.join(folder, "characterDescriptions.txt"), "w", encoding="utf-8")
outputFile = open(os.path.join(folder, "characterRelationships.txt"), "w", encoding="utf-8")


def extractName(roleText):
   roleWords = roleText.replace("l'","").replace("l’","").replace("[","").replace("]","").lower().split(" ")
   stopList = ["d.", "le", "la", "les", "un", "une", "des", "deux", "trois", "st", "ste", "dom", "don", "dona", "et"]
   if roleWords[0] in stopList:
      roleText = roleWords[1]
   else:
      roleText = roleWords[0]
   return roleText.replace(",","").replace(".","").lower()

# Consider all Word files in the corpus folder
for file in glob.glob(os.path.join(corpusFolder, "*.xml")):
      # Open the XML file and extract information
      print("Extracting information from " + file)
      mydoc = minidom.parse(file)
      
      roles = []
      # Extract character names
      roleNodes = mydoc.getElementsByTagName("role")
      if len(roleNodes) > 0:
         for role in roleNodes:
            roleText = extractName(displayNodeText(role, []))
            if len(roleText) > 0:
               roles.append(roleText)
      else:
         print("No role found!")
      #print(roles)
      #print(str(len(roles)) + " roles found!")
      
      
      # Extract character information without character names
      if len(mydoc.getElementsByTagName("castList")) > 0:
         castList = displayNodeText(mydoc.getElementsByTagName("castList")[0], ["role"])
         
         castListLines = castList.split("\n")
         for castItem in castListLines:
            res = re.search("^[,. ]*(sa|son) ([^ ,.]+)", castItem.lower())
            if res:
               outputFile.writelines(res.group(2) + "\n")
            for role in roles:
               res = re.search("([^ ]*[ ]*)([^ ]+) d(e|'|’|es) .*(" + role + ")", castItem.lower())
               if res:
                  outputFile.writelines(res.group(1) + "\t" + res.group(2) + "\t" + res.group(4) + "\n")
                  print(res.group(1) + "\t" + res.group(2) + "\t" + res.group(4))
         
      outputFile.writelines(" a a a a a a a a a a a a a a a a a a a \n")
         #for line in castList.split("\n"):
            #print(line)

outputFile.close()