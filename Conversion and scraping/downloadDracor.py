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

def concat_authors_in_list(l):
    return ' '.join(list(map(
        lambda d: d if d is None or type(d) is str
        else concat_authors_in_list(d) if type(d) is list 
        else concat_author_in_dico(d)
        , l)))

def concat_author_in_dico(s):
    if s is None or type(s) is str:
        return s
    if type(s) is list:
        return concat_authors_in_list(s)
    return ' '.join(list(map(
        lambda d: 'None' if d is None 
        else concat_authors_in_list(d) if type(d) is list 
        else d if type(d) is str
        else concat_author_in_dico(d), 
        s.values())))

def get_authors(content):
    # pen à gérer
    # preserve à gérer
    # None : cas key isno dans balise, pas de persName
    # cas complètement None ([] normalement ?)
    # Antoine de la Motte (sort = 1)
    s = content.get('TEI').get('teiHeader').get('fileDesc').get('titleStmt').get('author')
    if type(s) is str:
        return s
    if type(s) is list:
        return list(map(concat_author_in_dico, map(
            lambda d: d.get('persName') if type(d) is not str 
            else d, 
            filter(lambda d: d is not None, s))))           
    else:
        return concat_author_in_dico(s.get('persName'))

def get_year(content):
    #print("Content :", content.get('TEI').get('teiHeader').get('fileDesc').get('titleStmt'))
    return None

def extract_important_datas(contents):
    # return [{
    #     'title': get_title(content),
    #     'author': get_authors(content), 
    #     'year': get_year(content)} 
    #     for content in contents]
    for content in contents:
        s = get_authors(content)
        if s == 'pen Voltaire François-Marie Arouet':
            print("########", content.get('TEI').get('teiHeader').get('fileDesc').get('titleStmt').get('author').get('persName'))
        print(get_authors(content))
        


if __name__ == "__main__":
    data_dic = load_datas("https://dracor.org/api/corpora/fre")
    plays = data_dic.get('dramas')
    print(extract_important_datas(get_actual_datas(dracor_folder)))
