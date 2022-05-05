import glob, os, re, sys, time, requests, json, xmltodict

folder = os.path.abspath(os.path.dirname(sys.argv[0]))
dracor_folder = os.path.abspath(os.path.join(os.path.join(folder, os.pardir), "corpusDracor"))

def load_datas(link):
    """load datas from the chosen link.

    Args:
        link (string): chosen

    Returns:
        dict: Dictionnary with database from the URL.
    """
    return json.loads(requests.get(link, 'metrics').content)

def get_header(xml):
    return ''.join([xml.partition('<text>')[0], '</TEI>'])

def get_actual_datas(path):
    from os import walk
    contents = []
    files = list(map(lambda f: os.path.join(dracor_folder, f), next(walk(path), (None, None, []))[2]))
    for file in files:
        with open(file) as f:
            contents.append(xmltodict.parse(get_header(f.read())))
    return contents

def get_title(content):
    s = content.get('TEI').get('teiHeader').get('fileDesc').get('titleStmt').get('title')
    if type(s) is list:
        return list(s[0].values())[1]
    else:
        return list(s.values())[1]

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
    if (l_contains_pen(l)):
        pen_dico = l_find_pen(l)
        name = l[pen_dico].get('#text')
        if name is not None:
            return name
        else:
            return l[pen_dico].get('surname')
    return ' '.join(list(map(
        lambda d: 
            concat_authors_in_list(d) if type(d) is list 
            else concat_author_in_dico(d), 
            l)))

def concat_author_in_dico(persNames):
    if persNames is None or type(persNames) is str:
        return persNames
    pseudo = get_pseudonym(persNames)
    if pseudo is not None:
        return pseudo    
    preserve = get_preserve(persNames)
    if preserve is not None:
        return preserve
    if type(persNames) is list:
        return concat_authors_in_list(persNames)
    sort = get_sort(persNames)
    if not sort is None:
        return sort

    return concat_authors_in_list(list(filter(lambda value: value != 'nobility', persNames.values())))

def get_authors(content):
    s = content.get('TEI').get('teiHeader').get('fileDesc').get('titleStmt').get('author')
    if type(s) is str:
        return s
    if type(s) is list:
        res = list(filter(lambda author: author is not None, map(concat_author_in_dico, map(
            lambda d:
                d if d is None
                else d.get('persName') if type(d) is not str 
                else d, 
            s))))   
        if len(res) == 1:
            res = res[0]
        return res        
    persName = s.get('persName')
    if persName is None:
        return s.get('#text')
    return concat_author_in_dico(persName)

def choose_year(writtenYear, printYear, premiereYear):
    res = None
    if printYear is None:
        res = int(premiereYear)
    elif premiereYear is None:
        res = int(printYear)
    else:
        res = min(int(premiereYear), int(printYear))
    if writtenYear is not None and res - int(writtenYear) > 10:
        return writtenYear
    return str(res)


def get_year(content):
    dates = content.get('TEI').get('teiHeader').get('fileDesc').get('sourceDesc').get('bibl').get('bibl').get('date')
    all_dates = {'written': None, 'print': None, 'premiere': None}
    if type(dates) is list:
        for date in dates:
            typ = date.get('@type')
            if typ in all_dates.keys():
                year = date.get('@when')
                if year is None:
                    year = date.get('@notAfter')
            all_dates[typ] = year.split('-')[0]
        return choose_year(all_dates['written'], all_dates['print'], all_dates['premiere'])
    res = dates.get('@when')
    if res is None:
        res = dates.get('@notAfter')
    return res

def extract_important_datas(contents):
    return [{
        'title': get_title(content),
        'authors': get_authors(content), 
        'yearNormalized': get_year(content)} 
        for content in contents]  

def extract_datas_plays(plays):
    return [{
        'title': play.get('title'),
        'authors': play.get('authors'), 
        'yearNormalized': play.get('yearNormalized')} 
        for play in plays]

def display(datas):
    for data in datas:
        print(data, "\n")

def extend_nickname(t):
    if t[2] is not None:
        tmp = t[:2]
        tmp.extend(t[2])
        t = tmp
    else:
        t.pop(2)
    return t

def equals_authors(old, new):
    # print(new, "\n")
    if type(old) is str and type(new[0]) is str:
        new = extend_nickname(new)
        return old in new or new[0] in old
    if type(old) is list and type(new[0]) is list and len(old) == len(new):
        new = list(map(extend_nickname, new))
        return all(map(lambda name, names: name in names or any(
            names[0] in n or names[1] in n for n in old
            ), old, new))
    return False

def extract_duplicates(datas, new_datas):
    fd = 0
    for new_data in new_datas:
        tuple_datas = []
        for data in datas:
            if new_data.get('title') == data.get('title') and new_data.get('yearNormalized') == data.get('yearNormalized'):
                tuple_datas.append((data, new_data))
        for old, new in tuple_datas:
            old_authors = old['authors']
            new_authors = list(filter(lambda t: None not in t[:2], map(lambda author: [author['fullname'], author['shortname'], author.get('alsoKnownAs')], new['authors'])))
            if len(new_authors) == 1:
                new_authors = new_authors[0]
            if not equals_authors(old_authors, new_authors):
                print("old :", old, '\nnew :', new, '\n')
                fd += 1
    print("{fd} duplicates".format(fd = fd))

# alsoKnownAs (pen)

#probleme sort
"""
old : {'title': 'Conversations de Madame la Marquise de Maintenon', 'authors': ['Aubigné', OrderedDict([('@sort', '1'), ('#text', 'Maintenon')])], 'yearNormalized': '1637'} 
new : {'title': 'Conversations de Madame la Marquise de Maintenon', 'authors': [{'name': "Maintenon, Françoise d'Aubigné, marquise de", 'fullname': "Françoise d'Aubigné, marquise de Maintenon", 'shortname': 'Maintenon', 'refs': [{'ref': '0000000109273521', 'type': 'isni'}, {'ref': 'Q230670', 'type': 'wikidata'}]}], 'yearNormalized': '1637'} 
"""

#probleme de
"""
old : {'title': 'Endymion', 'authors': 'Fontenelle Bernard Le Bouvier Fontenelle de', 'yearNormalized': '1731'} 
new : {'title': 'Endymion', 'authors': [{'name': 'Fontenelle, Bernard Le Bouyer de', 'fullname': 'Bernard Le Bouyer de Fontenelle', 'shortname': 'Fontenelle', 'alsoKnownAs': ['Bernard Le Bouvier de Fontenelle']}], 'yearNormalized': '1731'} 
"""

#etc

if __name__ == "__main__":
    data_dic = load_datas("https://dracor.org/api/corpora/fre")
    plays = data_dic.get('dramas')
    datas = extract_important_datas(get_actual_datas(dracor_folder))
    new_datas = extract_datas_plays(plays)
    # display(datas)
    # display(new_datas)
    extract_duplicates(datas, new_datas)
    print("Actuellement : {datas} pièces\nTrouvé en ligne : {new_datas} pièces".format(datas = str(len(datas)), new_datas = str(len(new_datas))))
    

