from xml.dom import minidom
from os.path import abspath, dirname, join, exists
import os, sys
from os import walk

folder = abspath(dirname(sys.argv[0]))
root = join(folder, os.pardir)
dracor_folder = abspath(join(root, "corpusDracor"))
outputs_folder = abspath(join(root, "DracorTrees"))

if not exists(outputs_folder):
    os.makedirs(outputs_folder)

def writeStart(output):
    output.write("digraph Tree {\n")

def writeNodes(output, node, n=0):
    if not '#' in node.nodeName:
        output.write("\t\"t{0}\" [label = \"{1}\"];\n".format(n, node.nodeName))
        for child in node.childNodes:
            n = writeNodes(output, child, n + 1)
        return n
    return n - 1

def writeLinks(output, node, n=0):
    m = n
    for child in node.childNodes:
        if not '#' in child.nodeName:
            m += 1
            output.write("\t\"t{0}\" -> \"t{1}\";\n".format(n, m))
            m = writeLinks(output, child, m)
    return m

def writeEnd(output):
    output.write("}\n")

def parse_dot(output, node):
    writeStart(output)
    writeNodes(output, node)
    writeLinks(output, node)
    writeEnd(output)

def generate_graph(path):
    from os import walk
    files = list(map(lambda f: join(path, f), next(walk(path), (None, None, []))[2]))
    for file in files:
        name = file.split('/')[-1].replace('xml', 'dot')
        print(f"Generate graph : {name}")
        with open(join(outputs_folder, name), 'w') as f:
            parse_dot(f, minidom.parse(file).childNodes[0])

def parse_xml(output, node, indent=-1):
    name = node.nodeName
    if name[0] != '#':
        output.write(indent * '\t' + '<' + node.nodeName + '>\n')
    for child in node.childNodes:
        parse_xml(output, child, indent + 1)

def parse_plays(path):
    for file in list(map(lambda f: join(path, f), next(walk(path), (None, None, []))[2])):
        name = file.split('/')[-1].replace('xml', 'txt')
        print(f"Converting {file}")
        with open(join(outputs_folder, name), 'w') as f:
            parse_xml(f, minidom.parse(file))

def export_svg(path):
    for file in list(filter(lambda f: '.dot' in f, map(lambda f: join(path, f), next(walk(path), (None, None, []))[2]))):
        print(f"Exporting {file}")
        os.system(f"dot -Tsvg {file} -o {file.replace('.dot', '.svg')}")
        break

def parse_same_nodes(node, path_buffer=''):
    name = node.nodeName
    res = []
    if name[0] != '#':
        name = '/'.join([path_buffer, name])
        res.append(name)
        for child in node.childNodes:
            res.extend(parse_same_nodes(child, name))
    return res

def parse_same_links(node, path_buffer=''):
    name = node.nodeName
    res = []
    if name[0] != '#':
        name = '/'.join([path_buffer, name])
        for child in node.childNodes:
            if child.nodeName[0] != '#':
                res.append((name, '/'.join([name, child.nodeName])))
                res.extend(parse_same_links(child, name))
    return res

def find_same_nodes(path):
    nodes = []
    links = []
    for file in list(map(lambda f: join(path, f), next(walk(path), (None, None, []))[2])):
        name = file.split('/')[-1].replace('xml', 'txt')
        print(f"Check nodes of {file}")
        root = minidom.parse(file).childNodes[0]
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

def create_common_tree(nodes, links):
    with open(join(folder, 'common_tree.dot'), 'w') as output:
        writeStart(output)
        for node in nodes:
            output.write("\t\"{0}\" [label = \"{1}\"];\n".format(node, node.split('/')[-1]))
        for father, son in links:
            output.write("\t\"{0}\" -> \"{1}\";\n".format(father, son))
        writeEnd(output)

def parse_paths_from_file(node, path_buffer=''):
    nodes = []
    name = node.nodeName
    if name[0] != '#':
        name = '/'.join([path_buffer, name])
        print(name)
        for child in node.childNodes:
            parse_paths_from_file(child, name)   

def parse_paths(path):
    for file in list(map(lambda f: join(path, f), next(walk(path), (None, None, []))[2])):
        name = file.split('/')[-1].replace('xml', 'txt')
        print(f"Look path from {file}")
        parse_paths_from_file(minidom.parse(file).childNodes[0])  
        break       

if __name__ == "__main__":
    # parse_plays(dracor_folder)
    # generate_graph(dracor_folder)
    # export_svg(outputs_folder)
    nodes, links = find_same_nodes(dracor_folder)
    create_common_tree(nodes, links)
    # parse_paths(dracor_folder)

