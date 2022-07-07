from bs4 import BeautifulSoup
from os.path import abspath, dirname, join, exists, basename
from random import shuffle

import os, sys
import argparse

from os import walk

folder = abspath(dirname(sys.argv[0]))
root = join(folder, os.pardir)
inputs_folder = abspath(join(root, "cleanHTML_TD"))
outputs_folder = abspath(join(root, "treesTD"))
common_trees_folder = abspath(join(root, "commonTrees"))

if not exists(outputs_folder):
    os.makedirs(outputs_folder)

def safe_children(node):
    return list(filter(lambda child: child != node.text, node.children))

def parse_files(path):
    return list(filter(lambda file: basename(file) not in ['sitemap.html', 'index.html'], map(lambda f: join(path, f), next(walk(path), (None, None, []))[2])))

def writeStart(output):
    output.write("digraph Tree {\n")

def writeNodes(output, node, height, n=0):
    name = node.name
    if name is not None:
        output.write("\t\"t{0}\" [label = \"{1}\"];\n".format(n, name))
        if height != 0:
            for child in safe_children(node):
                n = writeNodes(output, child, height - 1, n + 1)
    return n

def writeLinks(output, node, height, n=0):
    m = n
    if node.name is not None:
        if height != 0:
            for child in safe_children(node):
                m += 1
                output.write("\t\"t{0}\" -> \"t{1}\";\n".format(n, m))
                m = writeLinks(output, child, height - 1, m)
    return m

def writeEnd(output):
    output.write("}\n")

def parse_dot(output, node, height):
    writeStart(output)
    writeNodes(output, node, height)
    writeLinks(output, node, height)
    writeEnd(output)

def generate_graph(path, height=-1):
    for file in parse_files(path):
        name = basename(file).replace('html', 'dot')
        print(f"Generate graph : {name}")
        with open(join(outputs_folder, name), 'w') as output, open(file, 'r') as input:
            parse_dot(output, BeautifulSoup(input, 'html.parser'), height)

def parse_html(output, node, indent=0):
    name = node.name
    if name is not None:
        output.write(indent * '\t' + '<' + name + '>\n')
        for child in safe_children(node):
            parse_html(output, child, indent + 1)

def parse_plays(path):
    for file in parse_files(path):
        name = basename(file).replace('html', 'txt')
        print(f"Converting {file}")
        with open(join(outputs_folder, name), 'w') as output, open(file, 'r') as input:
            parse_html(output, BeautifulSoup(input, 'html.parser'))

def parse_same_nodes(node, path_buffer=''):
    res = []
    if node.name is not None:
        name = '/'.join([path_buffer, node.name])
        res.append(name)
        for child in safe_children(node):
            res.extend(parse_same_nodes(child, name))
    return res

def parse_same_links(node, path_buffer=''):
    res = []
    if node.name is not None:
        name = '/'.join([path_buffer, node.name])
        for child in safe_children(node):
            if child.name is not None:
                res.append((name, '/'.join([name, child.name])))
                res.extend(parse_same_links(child, name))
    return res

def find_same_nodes(path, limit=-1):
    nodes = []
    links = []
    files = parse_files(path)
    if limit != - 1:
        shuffle(files)
        files = files[:limit]
    for file in files:
        name = basename(file).replace('html', 'txt')
        print(f"Check nodes of {file}")
        with open(file, 'r') as output:
            root = BeautifulSoup(output, 'html.parser')
        res = parse_same_nodes(root)
        if not nodes:
            nodes = list(set(res.copy()))
        else:
            nodes = list(filter(lambda node: node in res, nodes))
        res2 = parse_same_links(root)
        if not links:
            links = list(set(res2.copy()))
        else:
            links = list(filter(lambda link: link in res2, links))
    return nodes, links

def create_common_tree(nodes, links, file):
    with open(join(common_trees_folder, file), 'w') as output:
        writeStart(output)
        for node in nodes:
            output.write("\t\"{0}\" [label = \"{1}\"];\n".format(node, node.split('/')[-1]))
        for father, son in links:
            output.write("\t\"{0}\" -> \"{1}\";\n".format(father, son))
        writeEnd(output)

def clean_outputs_directory(path):
    if 'trees' in path:
        os.system('rm ' + path + '/*')
        print('Delete files from', path)
    else:
        print("You can only clean an output folder (begin with tree).")   

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Display Tree from HTML.')  
    parser.add_argument(
        '-a', '--acts',
        help="Selects only the files respecting a number minimum and maximum of an act.",
        metavar='N', type=int,
        nargs=2)
    parser.add_argument(
        '-c', '--clean',
        help="Cleans up given folders",
        nargs='?')
    parser.add_argument(
        '-d', '--directory',
        help="Selects a folder to run the program (by default, TD).",
        nargs='?')
    parser.add_argument(
        '-i', '--intersection',
        help="Generates the intersection of the structure of each XML file as a dot file.",
        metavar='N', type=int, default=False,
        nargs='?')
    parser.add_argument(
        '-p', '--precision',
        help="Explains act, scene, and line numbers, as well as characters.",
        action="store_true")
    parser.add_argument(
        '-t', '--tree',
        help="Generates for each selected html file its structure in a dot file, in the form of a tree.",
        type=int, default=False,
        nargs='?')
    parser.add_argument(
        '-v', '--verbose', 
        help="Generates for each selected html file its structure in a text file", 
        action="store_true")
    
    args = parser.parse_args()
    print(args)

    clean = args.clean
    if clean:
        clean_outputs_directory(abspath(join(root, abspath(join(root, clean)).replace('Corpus', 'trees').replace('corpus', 'trees'))))

    directory = args.directory
    if directory:
        inputs_folder = abspath(join(root, directory))
        outputs_folder = abspath(join(root, directory.replace('Corpus', 'trees').replace('corpus', 'trees')))
    
    if not exists(outputs_folder):
        os.makedirs(outputs_folder)

    if args.verbose:
        parse_plays(inputs_folder)
    
    if args.tree is not False:
        if args.tree is not None:
            if args.tree < 0:
                raise ValueError("argument -t/--tree : int height value must be positive.")
            generate_graph(inputs_folder, args.tree)
        else:
            generate_graph(inputs_folder)
    
    if args.intersection is not False:
        if args.intersection is not None:
            if args.intersection <= 0:
                raise ValueError("argument -i/--intersection : int height value must be strictly positive.")
            nodes, links = find_same_nodes(inputs_folder, args.intersection)
        else:
            nodes, links = find_same_nodes(inputs_folder)
        common_file = join(common_trees_folder, basename(inputs_folder).replace('corpus', 'common').replace('Corpus', 'common') + '.dot')
        create_common_tree(nodes, links, common_file)


    


