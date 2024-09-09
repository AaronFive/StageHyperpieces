#!/usr/sfw/bin/python
# -*- coding: utf-8 -*-

import glob, os, re, sys, time, requests
from os.path import abspath, dirname, join, exists

"""
    downloadTheatreDocumentation, a script to automatically 
    download HTML plays from théâtre-documentation.com
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

# Get the current folder
folder = abspath(dirname(sys.argv[0]))
root_folder = abspath(join(folder, os.pardir))
html_folder = abspath(join(root_folder, "notConvertTD"))

if not exists(html_folder):
    os.makedirs(html_folder)


# Get the list of pages listing plays from http://xn--thtre-documentation-cvb0m.com/content/oeuvres
playList = requests.get("http://xn--thtre-documentation-cvb0m.com/content/oeuvres")
playListPages = []
for line in playList.text.split("\n"):
    res = re.search("<li .*leaf.*><a href=\"/content/([^\"]+)\">[^<]+</a></li>", line)
    if res:
        playListPages.append("http://xn--thtre-documentation-cvb0m.com/content/" + res.group(1))

# Clean the title or the author name
def cleanString(str):
    return str.replace('"','').replace('\t',' ').replace('  ',' ')

# Transform a title into a file name without special characters
def cleanTitle(str):
    # lowercase + delete le la etc.
    str = str.lower().replace("(l’)","").replace("(l')","").replace("(le)","").replace("(la)","").replace("(les)","").replace("(un)","").replace("(une)","").replace("(des)","")
    # remove final space or newline character
    regex = "^(.*)[ \r\n]$"
    res = re.search(regex, str)
    while res: 
        str = res.group(1)
        res = re.search(regex, str)
    replacements = {
    "æ":"ae",
    "à":"a",
    "â":"a",
    "ä":"a",
    "ç":"c",
    "é":"e",
    "è":"e",
    "ê":"e",
    "ë":"e",
    "î":"i",
    "ï":"i",
    "œ":"oe",
    "ô":"o",
    "ö":"o",
    "ù":"u",
    "û":"u",
    "ü":"u",
    "'":" ",
    "’":" ",
    ",":" ",
    ".":" ",
    "?":" ",
    "!":" ",
    "/":" ",
    "(":" ",
    ")":" ",
    "-":" ",
    ":":" ",
    "«":" ",
    "»":" ",
    ";":" ",
    "_":" ",
    "…":" ",
    "*":" ",
    chr(8203):" ",
    }
    for key in replacements:
        str = str.replace(key, replacements[key])

    # remove double spaces
    regex = "^[ ]*(.*)  (.*)$"
    res = re.search(regex, str)
    while res: 
        str = res.group(1) + " " + res.group(2)
        res = re.search(regex, str)
      
    # remove final space
    regex = "^(.*) $"
    res = re.search(regex, str)
    if res: 
        str = res.group(1)

    # remove initial space
    regex = "^ (.*)$"
    res = re.search(regex, str)
    if res: 
        str = res.group(1)
   
    return str.replace(' ','_')

# Get the list of all plays by visiting those pages
"""
outputFile = open("PlaysFromTheatreDocumentation.csv", "w", encoding="utf-8")
for url in playListPages:
    print("Extracting the plays from page " + url)
    playList = requests.get(url)
    for line in playList.text.split("\n"):
        # Check if the plays title is not written in blue or green
        # (otherwise the text is not available)
        res = re.search("</span>", line)
        if not(res):
            # remove internal link interruptions in the title
            regex = "^(.*)</a><a[^>]*>(.*)$"
            res = re.search(regex, line)
            while res: 
                line = res.group(1) + res.group(2)
                res = re.search(regex, line)
            line = line.replace('&nbsp;',' ')
            # Extract information from line
            res = re.search("<li[^>]*><a href=\"([^\"]+)\">([^<>]+) \(([^\)]+)\)</a></li>", line)
            if res:
                # Save the found url of the play
                outputFile.writelines(cleanTitle(res.group(3)) + "-" + cleanTitle(res.group(2)) + ".html\t"
                + "http://théâtre-documentation.com/content/" + res.group(1) + "\t" 
                + cleanString(res.group(3)) + "\t" 
                + cleanString(res.group(2)) + "\n")
    time.sleep(1)
outputFile.close()
"""

characters = []
# Save each play
allPlays = open("PlaysFromTheatreDocumentation_without27first.csv", "r", encoding="utf-8")

for playLine in allPlays:
    res = re.search("([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\n", playLine)
    if res:
        fileName = res.group(1)
        if not exists(join(html_folder, fileName)):
            # Add title characters to the list of observed characters
            for char in fileName:
                if char not in characters:
                    characters.append(char)
            print("Downloading " + res.group(4) + " by " + res.group(3))

            response = requests.get(res.group(2))
            open(join(html_folder, fileName), 'wb').write(response.content)
            print("File " + fileName + " written!")
            time.sleep(2)

"""
# Check if titles contain no special characters
for char in characters:
   print(char + " : " + str(ord(char)))
"""