import glob, os, re, sys, time, requests, json

import xml.etree.ElementTree as ET

#compatible avec python 3.10

folder = os.path.abspath(os.path.dirname(sys.argv[0]))
dracor_folder = os.path.abspath(os.path.join(os.path.join(folder, os.pardir), "corpusDracor"))
tei_header = '{http://www.tei-c.org/ns/1.0}'

def load_datas(link):
    """load datas from the chosen link.
    Args:
        link (string): chosen
    Returns:
        dict: Dictionnary with database from the URL.
    """

    # verification réponse requête ?
    return json.loads(requests.get(link, 'metrics').content)

def get_header(xml):
    return ''.join([xml.partition('<text>')[0], '</TEI>'])

def get_actual_datas(path):
    from os import walk
    contents = []
    files = list(map(lambda f: os.path.join(dracor_folder, f), next(walk(path), (None, None, []))[2]))
    for file in files:
        with open(file) as f:
            contents.append(ET.parse(f))
    return contents

def get_title(content):
    return content[0][0][0][0].text

def contains_pen(d):
    return 'pen' in d.values()

def l_contains_pen(l):
    return any(contains_pen(d) for d in filter(lambda d: isinstance(d, dict), l))

def l_find_pen(l):
    for i in range(len(l)):
        if isinstance(l[i], dict) and 'pen' in l[i].values():
            return i
    return len(l)

def get_sort(persName):
    if isinstance(persName, dict):
        surnames = persName.get('surname')
        if type(surnames) is list:
            for surname in surnames:
                if(isinstance(surname, dict)) and surname.get('@sort') == '1':
                    return surname.get('#text')
    return None

def get_preserve(persNames):
    if type(persNames) is list and len(persNames) == 2:
        persName, d = persNames
        if(isinstance(d, dict)) and d.get('@xml:space') == 'preserve' and type(persName) is str:
            return persName
    elif(isinstance(persNames, dict)) and persNames.get('@xml:space') == 'preserve':
        return persNames.get('surname')
    return None

def get_pseudonym(persNames):
    if type(persNames) is list:
        for persName in persNames:
            if(isinstance(persName, dict)) and persName.get('@type') == 'pseudonym':
                pseudo = persName.get('#text')
                if pseudo is None:
                    return persName.get('surname')
                return pseudo
    elif(isinstance(persNames, dict)) and persNames.get('@type') == 'pseudonym':
        return persNames.get('surname')
    return None

def concat_authors_in_list(l):
    # pen
    return ' '.join(list(map(
        lambda d:
            d if type(d) is str      
            else d.text if d.text is not None and d.text.strip() != '' 
            else concat_authors_in_list(d) if len(d) > 1
            else concat_author_in_dico(d) if len(d) == 1
            else 'None',
            l)))

def concat_author_in_dico(persNames):
    if persNames is None or type(persNames) is str:
        return persNames
    #pseudo
    #preserve
    if len(persNames) != 1:
        return concat_authors_in_list(persNames)
    #sort
    return concat_authors_in_list([child.text for child in persNames[0]])

def tei_findall(root, name):
    return root.findall(''.join([tei_header, name]))

# {'{http://www.w3.org/XML/1998/namespace}space': 'preserve'}
# {'type': 'pen'}
# {'type': 'pseudonym'}
def get_authors(content):
    titleStmt = content[0][0][0]
    authors = tei_findall(titleStmt, 'author') 
    for author in authors:
        if author.text == '[anonyme]':
            return author.text

    #persNames = tei_findall(author, 'persName')

    if len(authors) == 1:
        author = authors[0]
        persNames = tei_findall(author, 'persName')
        if len(persNames) == 0:
            return author.text
        else:
            return concat_author_in_dico(persNames)
    
    else:
        return list(map(concat_author_in_dico, map(
            lambda author: 
                tei_findall(author, 'persName') if tei_findall(author, 'persName') != 0
                else author.text, 
            filter(lambda author: author is not None, authors))))

def get_year(content):
    bibl = content[0][0][2][0][4]
    for date in tei_findall(bibl, 'date'):
        attrib = date.attrib
        if attrib.get('type') == 'print':
            res = attrib.get('when')
            if res is None:
                return '-'.join([attrib.get('notBefore'), attrib.get('notAfter')])
            return res


def extract_important_datas(contents):
    return [{
        'title': get_title(root),
        'authors': get_authors(root), 
        'year': get_year(root)} 
        for root in map(lambda content: content.getroot(), contents)]


def display(datas):
    for data in filter(lambda d: d.get('authors') != '', datas):
        print(data)

if __name__ == "__main__":
    data_dic = load_datas("https://dracor.org/api/corpora/fre")
    plays = data_dic.get('dramas')
    display(extract_important_datas(get_actual_datas(dracor_folder)))