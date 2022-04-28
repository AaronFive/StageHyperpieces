import glob, os, re, sys, time, requests, json

def load_datas(link):
    return json.loads(requests.get(link, 'metrics').content)

if __name__ == "__main__":
    data_dic = load_datas("https://dracor.org/api/corpora/fre")
    plays = data_dic.get('dramas')
