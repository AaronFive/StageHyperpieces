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

outputFile = open(os.path.join(folder, "characterDescriptions.txt"), "w", encoding="utf-8")

# Consider all Word files in the corpus folder
for file in glob.glob(os.path.join(corpusFolder, "*.xml")):
      # Open the XML file and extract information
      print("Extracting information from " + file)
      mydoc = minidom.parse(file)

      # Extract character information without character names
      if len(mydoc.getElementsByTagName("castList")) > 0:
         castList = displayNodeText(mydoc.getElementsByTagName("castList")[0], ["role"])
         outputFile.writelines(castList + " a a a a a a a a a a a a a a a a a a a \n")

outputFile.close()