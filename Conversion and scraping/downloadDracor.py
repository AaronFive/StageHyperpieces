import glob, os, re, sys, time, requests, json

# response = requests.get("https://dracor.org/api/corpora", 'metrics')
# print(response.status_code)
# print(response.json())

# def jprint(obj):
#     # create a formatted string of the Python JSON object
#     text = json.dumps(obj, sort_keys = True, indent = 4)
#     print(text)

# jprint(response.json())


response_fre = requests.get("https://dracor.org/api/corpora/fre", 'metrics')

def j_to_string(obj):
    # create a formatted string of the Python JSON object
    return json.dumps(obj, sort_keys = True, indent = 4)

print(j_to_string(response_fre.json()))

# # Get the current folder
# folder = os.path.abspath(os.path.dirname(sys.argv[0])) 

# # List of plays to generate

# # TODO : Créer classe qui stocke option d'ajouts de fichiers.

# # https://dracor.org/api/corpora/fre/play/