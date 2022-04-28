import glob, os, re, sys, time, requests, json

data_dic = json.loads(requests.get("https://dracor.org/api/corpora/fre", 'metrics').content)

plays = data_dic.get('dramas')

print(plays[0].keys())