from xml.dom import minidom
from os.path import abspath, dirname, join, exists
import os, sys

folder = abspath(dirname(sys.argv[0]))
root = join(folder, os.pardir)
dracor_folder = abspath(join(root, "corpusDracor"))
outputs_folder = abspath(join(root, "DracorTrees"))

if not exists(outputs_folder):
    os.makedirs(outputs_folder)

def writeStart(output, node):
    # output.write("graph Tree {node [shape=record,height=.1]\n")
    # output.write("edge [tailclip=false, arrowtail=dot, dir=both];\n")
    output.write("graph Tree {\n")

# def export_dot(node):
#     """Renvoie une chaîne encodant le graphe au format dot."""
#     liste_sommets = node.sommets()
#     liste_aretes = node.aretes()

#     chaine = "graph G {\n"

#     for sommet in liste_sommets:
#         chaine += "\t{0} [label = \"{1}\"];\n".format(sommet, node.nom_sommet(sommet))

#     for u, v, station in liste_aretes:
#         chaine += "\t{0} -> {1} [label = \"{2}\"];\n".format(u, v, station)

#     chaine += "}"

#     return chaine

# def writeNodes(output, node, n=0):
#     output.write("\t\"t{0}\" [label = \"{1}\"];\n".format(n, node.nodeName))
#     for child in node.childNodes:
#         n = writeNodes(output, child, n + 1)
#     return n

# def writeLinks(output, node, n=0):
#     m = n
#     for child in node.childNodes:
#         m += 1
#         output.write("\t\"t{0}\" -- \"t{1}\";\n".format(n, m))
#         m = writeLinks(output, child, m)
#     return m

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
            output.write("\t\"t{0}\" -- \"t{1}\";\n".format(n, m))
            m = writeLinks(output, child, m)
    return m

def writeEnd(output, node):
    output.write("}\n")

def parse_dot(output, node):
    writeStart(output, node)
    writeNodes(output, node)
    writeLinks(output, node)
    writeEnd(output, node)

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
    from os import walk
    files = list(map(lambda f: join(path, f), next(walk(path), (None, None, []))[2]))
    for file in files:
        name = file.split('/')[-1].replace('xml', 'txt')
        print(f"Converting {file}")
        with open(join(outputs_folder, name), 'w') as f:
            parse_xml(f, minidom.parse(file))

def export_svg(path):
    from os import walk
    files = list(filter(lambda f: '.dot' in f, map(lambda f: join(path, f), next(walk(path), (None, None, []))[2])))
    for file in files:
        print(f"Exporting {file}")
        os.system(f"dot -Tsvg {file} -o {file.replace('.dot', '.svg')}")
        break

if __name__ == "__main__":
    parse_plays(dracor_folder)
    generate_graph(dracor_folder)
    # export_svg(outputs_folder)