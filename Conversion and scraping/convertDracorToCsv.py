from json.tool import main
import os, sys, requests, json, xmltodict
from os.path import abspath, dirname, join

from xml.etree import ElementTree as ET
import csv

folder = abspath(dirname(sys.argv[0]))
dracor_folder = abspath(join(join(folder, os.pardir), "corpusDracor"))

def get_actual_meta_datas(path):
    from os import walk
    contents = []
    files = list(map(lambda f: join(path, f), next(walk(path), (None, None, []))[2]))
    for file in files:
            xml = ET.parse(file)
            with open("Dracor.csv", 'w', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(['title', 'author', 'year'])

                for 
    return contents

if __name__ == "__main__":
    get_actual_meta_datas(dracor_folder)