import sys, re
from os import walk, makedirs, pardir
from os.path import abspath, join, dirname, exists, basename


folder = abspath(dirname(sys.argv[0]))
root = join(folder, pardir)
input = join(root, 'notConvertTD')
output = join(root, 'cleanHTML_TD')

def parse_files(input):
    return list(filter(lambda file: 'xml' not in basename(file), map(lambda f: join(input, f), next(walk(input), (None, None, []))[2])))

def convertHTML(input, output):
    for file in parse_files(input):
        print(f"Convert {file}")
        with open(join(input, basename(file)), 'r') as f:
            with open(join(output, basename(file)), 'w') as out:
                for l in f:
                    res = re.search("<([^ ]*) .*/>", l)
                    if (res):
                        l = l.replace("/>", f"></{res.group(1)}>").replace(' >', '>')
                    out.write(l)

if __name__ == "__main__":
    if not exists(output):
        makedirs(output)
    convertHTML(input, output)
    