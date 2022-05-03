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
        lambda d: d if d is None or type(d) is str
        else concat_authors_in_list(d) if type(d) is list 
        else concat_author_in_dico(d)
        , l)))

def concat_author_in_dico(s):
    if s is None or type(s) is str:
        return s
    pseudo = get_pseudonym(s)
    if pseudo is not None:
        return pseudo    
    preserve = get_preserve(s)
    if preserve is not None:
        return preserve
    if type(s) is list:
        return concat_authors_in_list(s)
    sort = get_sort(s)
    if not sort is None:
        return sort
    # return ' '.join(list(map(
    #     lambda d: 'None' if d is None 
    #     else concat_authors_in_list(d) if type(d) is list 
    #     else d if type(d) is str
    #     else concat_author_in_dico(d), 
    #     s.values())))
    return concat_authors_in_list(s.values())

def get_authors(content):
    s = content.get('TEI').get('teiHeader').get('fileDesc').get('titleStmt').get('author')
    if type(s) is str:
        return s
    if type(s) is list:
        return list(map(concat_author_in_dico, map(
            lambda d:
                d if d is None
                else d.get('persName') if type(d) is not str 
                else d, 
            s)))           
    else:
        persName = s.get('persName')
        if persName is None:
            return s.get('#text')
        return concat_author_in_dico(persName)

def get_year(content):
    dates = content.get('TEI').get('teiHeader').get('fileDesc').get('sourceDesc').get('bibl').get('bibl').get('date')
    if type(dates) is list:
        for date in dates:
            if date.get('@type') == 'print':
                res = date.get('@when')
                if res is None:
                    return '-'.join([date.get('@notBefore'), date.get('@notAfter')])
                return res
    return dates.get('@when')

def extract_important_datas(contents):
    return [{
        'title': get_title(content),
        'authors': get_authors(content), 
        'year': get_year(content)} 
        for content in contents]  

def display(datas):
    for data in datas:
        print(data)

if __name__ == "__main__":
    data_dic = load_datas("https://dracor.org/api/corpora/fre")
    plays = data_dic.get('dramas')
    datas = extract_important_datas(get_actual_datas(dracor_folder))
    display(datas)
