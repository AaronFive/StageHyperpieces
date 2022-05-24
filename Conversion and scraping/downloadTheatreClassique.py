#!/usr/sfw/bin/python
# -*- coding: utf-8 -*-

import glob, os, re, sys, time, requests

from os.path import abspath, dirname, join, exists

"""
    downloadTheatreClassique, a script to automatically 
    download XML-TEI plays from theatre-classique.fr
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

# Get the current folder
folder = abspath(dirname(sys.argv[0]))
root_folder = abspath(join(folder, os.pardir))
TC_folder = abspath(join(root_folder, "corpusTC"))
link = "http://www.theatre-classique.fr/pages/documents/"

if not exists(TC_folder):
    os.makedirs(TC_folder)

plays = []

for line in requests.get("http://www.theatre-classique.fr/pages/programmes/PageEdition.php").text.split('\n'):
   res = re.search("[^<]*<td width='30' align='center'><a href='./edition.php\?t=../documents/([^']*)'>HTML</a></td>.*", line)
   if res:
      plays.append(link + res.group(1))

for url in plays:
   # Extract the file name
   res = re.search("\/([^\/]+.xml)", url)
   if res:
      fileName = res.group(1)
      if not exists(join(TC_folder, fileName)):
         print("Downloading file " + fileName)
         # Download and save the file
         response = requests.get(url)

         open(join(TC_folder, fileName), 'wb').write(response.content)
         time.sleep(5)

print(len(plays))