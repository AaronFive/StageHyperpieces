import xml.etree.ElementTree as ET

from convertTheatredocToDracor import *

counters = dict()


def initialize_xml():
    root = ET.Element('TEI', attrib={'xml:lang': "fre", 'xmlns': "http://www.tei-c.org/ns/1.0"})
    return root


# TEI HEADER

def write_header(root_node):
    header = ET.SubElement(root_node, 'teiHeader')
    # Creating the fileDescription

    return header


def write_file_profile_and_revisionDesc(header):
    fileDesc = ET.SubElement(header, "fileDesc")
    profileDesc = ET.SubElement(header, "profileDesc")
    revisionDesc = ET.SubElement(header, "revisionDesc")
    return fileDesc, profileDesc, revisionDesc


def write_fileDesc(fileDesc_node):
    """Create and organize the structure inside the fileDesc node."""

    # Create the titleStmt (already being handled in other functions, like write_title)
    titleStmt = ET.SubElement(fileDesc_node, 'titleStmt')

    # Create publicationStmt (this is where publication information goes, already handled in write_source)
    publicationStmt = ET.SubElement(fileDesc_node, 'publicationStmt')

    # Create sourceDesc (to hold information about the source, like in write_source and write_dates)
    sourceDesc = ET.SubElement(fileDesc_node, 'sourceDesc')

    return titleStmt, publicationStmt, sourceDesc


def write_title_and_type(titleStmt, title, genre):
    """Write the genre (type) of the play as a sub-title in the TEI header.

    Args:
        fileDesc (Element): The fileDesc node to add the genre element.
        genre (str): The genre of the play.
    """
    type_node = None
    if titleStmt is not None:
        title_node = ET.SubElement(titleStmt, 'title', attrib={"type": "main"})
        title_node.text = title
        if genre and genre != '[indéfini]':
            type_node = ET.SubElement(titleStmt, 'title', attrib={"type": "sub"})
            type_node.text = genre

    return type_node


def write_author(titleStmt, author):
    """Write the author's name as part of the TEI header.

    Args:
        fileDesc (Element): The fileDesc node to add the author element.
        author (tuple): A tuple containing two lists: forename(s) and surname(s).

    Returns:
        bool: True if the author's name is provided, False otherwise.
    """
    forename, surname = author
    if forename or surname:
        author_el = ET.SubElement(titleStmt, 'author')
        pers_name_el = ET.SubElement(author_el, 'persName')

        if forename:
            for name in forename:
                if name in ['de', "d'"]:
                    ET.SubElement(pers_name_el, 'linkname').text = name
                elif name == 'Abbé':
                    ET.SubElement(pers_name_el, 'rolename').text = name
                else:
                    ET.SubElement(pers_name_el, 'forename').text = name

        if surname:
            for name in surname:
                if name in ['DE', "D'"]:
                    ET.SubElement(pers_name_el, 'linkname').text = name.lower()
                else:
                    ET.SubElement(pers_name_el, 'surname').text = name.capitalize()

        # Adding editor information
        ET.SubElement(titleStmt,
                      'editor').text = "Adrien Roumégous, dans le cadre d'un stage de M1 Informatique encadré par Aaron Boussidan et Philippe Gambette."
    return author is not None


def write_source(publicationStmt, sourceDesc, source):
    """Write the source information as part of the TEI header.

    Args:
        fileDesc (Element): The fileDesc node to add the source elements.
        source (str): The source URL of the play.
    """
    ET.SubElement(publicationStmt, 'publisher', attrib={"xml:id": "dracor"}).text = "DraCor"
    ET.SubElement(publicationStmt, 'idno', attrib={"type": "URL"}).text = "https://dracor.org"
    ET.SubElement(publicationStmt, 'idno',
                  attrib={"type": "dracor", "xml:base": "https://dracor.org/id/"}).text = "fre[6 chiffres]"
    ET.SubElement(publicationStmt, 'idno',
                  attrib={"type": "wikidata", "xml:base": "http://www.wikidata.org/entity/"}).text = "Q[id]"

    availability = ET.SubElement(publicationStmt, 'availability')
    licence = ET.SubElement(availability, 'licence')
    ET.SubElement(licence, 'ab').text = "CC BY-NC-SA 4.0"
    ET.SubElement(licence, 'ref',
                  attrib={"target": "https://creativecommons.org/licenses/by-nc-sa/4.0/"}).text = "Licence"

    bibl = ET.SubElement(sourceDesc, 'bibl', attrib={"type": "digitalSource"})
    ET.SubElement(bibl, 'name').text = "Théâtre Documentation"
    ET.SubElement(bibl, 'idno', attrib={"type": "URL"}).text = source

    availability_bibl = ET.SubElement(bibl, 'availability')
    licence_bibl = ET.SubElement(availability_bibl, 'licence')
    ET.SubElement(licence_bibl,
                  'ab').text = "loi française n° 92-597 du 1er juillet 1992 et loi n°78-753 du 17 juillet 1978"
    ET.SubElement(licence_bibl, 'ref', attrib={
        "target": "http://théâtre-documentation.com/content/mentions-l%C3%A9gales#Mentions_legales"}).text = "Mentions légales"

    return bibl


def write_dates(bibl, date_written, date_print, date_premiere, line_premiere):
    """Write the dates for writing, printing, and premiere of the play.

    Args:
        fileDesc (Element): The fileDesc node to add the date elements.
        date_written (str): The date the play was written.
        date_print (str): The date the play was printed.
        date_premiere (str or tuple): The date of the first performance, or a range of dates.
        line_premiere (str): The line describing the first performance.
    """

    if date_written != "[vide]":
        ET.SubElement(bibl, 'date', attrib={"type": "written", "when": date_written})

    if date_print != "[vide]":
        ET.SubElement(bibl, 'date', attrib={"type": "print", "when": date_print})

    if date_premiere != "[vide]":
        if isinstance(date_premiere, str):
            ET.SubElement(bibl, 'date', attrib={"type": "premiere", "when": date_premiere}).text = line_premiere
        else:
            ET.SubElement(bibl, 'date', attrib={"type": "premiere", "notBefore": date_premiere[0],
                                                "notAfter": date_premiere[1]}).text = line_premiere


def write_profile_and_revision(profileDesc, revisionDesc, genre, vers_prose, counters):
    """Write the end of the TEI header, including character information and class code.

    Args:
        header (Element): The root node to add the final header elements.
        genre (str): The genre of the play.
        vers_prose (str): The form of the play (in verse or prose).
    """
    particDesc = ET.SubElement(profileDesc, 'particDesc')
    listPerson = ET.SubElement(particDesc, 'listPerson')

    for charaid, charaname in zip(counters["characterIDList"], counters["characterFullNameList"]):
        person = ET.SubElement(listPerson, 'person', attrib={"xml:id": charaid, "sex": "SEX"})
        ET.SubElement(person, 'persName').text = charaname

    wikidata_codes = {'Tragi-Comédie': 192881, 'Farce': 193979, 'Tragédie': 80930, 'Comédie': 40831,
                      'Vaudeville': 186286, 'Proverbe': 2406762, 'Pastorale': 0, 'Dialogue': 0}
    wikicode = wikidata_codes.get(genre, None)

    textClass = ET.SubElement(profileDesc, 'textClass')
    keywords = ET.SubElement(textClass, 'keywords', attrib={"scheme": "http://theatre-documentation.fr"})
    ET.SubElement(keywords, 'term').text = genre
    ET.SubElement(keywords, 'term').text = vers_prose

    if wikicode is not None:
        ET.SubElement(textClass, 'classCode',
                      attrib={"scheme": "http://www.wikidata.org/entity/"}).text = f"Q{wikicode}"

    listChange = ET.SubElement(revisionDesc, 'listChange')
    ET.SubElement(listChange, 'change', attrib={"when": f"{date.today()}"}).text = "(mg) file conversion from source"


def write_tei_header(root, metadata, counters):
    # Call the first function to write the teiHeader
    header = write_header(root)
    fileDesc, profileDesc, revisionDesc = write_file_profile_and_revisionDesc(header)
    titleStmt, publicationStmt, sourceDesc = write_fileDesc(fileDesc)

    title = metadata['main_title']
    genre = metadata['genre']
    author = (metadata['author_forename'], metadata['author_surname'])
    source = metadata['source']
    date_written = metadata['date_written']
    date_print = metadata['date_print']
    date_premiere = metadata['date_premiere']
    line_premiere = metadata['premiere_text']
    vers_prose = metadata['vers_prose']

    # Call each function to build up the header
    write_title_and_type(titleStmt, title, genre)
    write_author(titleStmt, author)
    bibl = write_source(publicationStmt, sourceDesc, source)  # TODO : handle different bibl
    write_dates(bibl, date_written, date_print, date_premiere, line_premiere)
    write_profile_and_revision(profileDesc, revisionDesc, genre, vers_prose, counters)
    return root


# WRITE text structure

def write_text(root):
    text = ET.SubElement(root, 'text')
    front = ET.SubElement(text, 'front')
    body = ET.SubElement(text, 'body')
    return front, body


# WRITE FRONT PART

def write_title_parts(doc_title_node, main_title, sub_title=None):
    """Adds title parts to the given <docTitle> node."""
    title_part_main = ET.SubElement(doc_title_node, 'titlePart', attrib={'type': 'main'})
    title_part_main.text = main_title
    if sub_title is not None:
        title_part_sub = ET.SubElement(doc_title_node, 'titlePart', attrib={'type': 'sub'})
        title_part_sub.text = sub_title

#TODO: Handle Date print line, acheve imprime, privilege, printer
# For now I don't find date_text
def write_doc_date(front_node, date_text, date_print):
    """Adds the <docDate> node."""
    doc_date = ET.SubElement(front_node, 'docDate', attrib={'when': date_print})
    doc_date.text = date_text


def write_privilege(doc_imprint_node, head_text, paragraphs):
    """Adds the privilege section to the <docImprint> node."""
    privilege = ET.SubElement(doc_imprint_node, 'div', attrib={'type': 'privilege'})
    head = ET.SubElement(privilege, 'head')
    head.text = head_text

    for paragraph in paragraphs:
        p = ET.SubElement(privilege, 'p')
        p.text = paragraph


def write_acheve_imprime(doc_imprint_node, acheve_text):
    """Adds the achevé d'imprimer section."""
    acheve_imprime = ET.SubElement(doc_imprint_node, 'div', attrib={'type': 'acheveImprime'})
    acheve_p = ET.SubElement(acheve_imprime, 'p')
    acheve_p.text = acheve_text


def write_printer(doc_imprint_node, printer_text):
    """Adds the printer section."""
    printer = ET.SubElement(doc_imprint_node, 'div', attrib={'type': 'printer'})
    printer_p = ET.SubElement(printer, 'p')
    printer_p.text = printer_text


def write_performance(front_node, premiere_text, premiere_date, premiere_location):
    """Adds the <performance> node."""
    performance = ET.SubElement(front_node, 'performance')
    ab = ET.SubElement(performance, 'ab', attrib={'type': 'premiere'})
    ab.text = premiere_text
    ab.set("date", premiere_date)
    ab.set("location", premiere_location)

#Found in counters[dedicace] and counters[dedicace_head]
def write_dedicace(front_node, salute_text, head_text, paragraphs, signed_text):
    """Adds the dedication section (<div type="dedicace">)."""
    dedicace = ET.SubElement(front_node, 'div', attrib={'type': 'dedicace'})
    opener = ET.SubElement(dedicace, 'opener')
    salute = ET.SubElement(opener, 'salute')
    salute.text = salute_text

    head = ET.SubElement(dedicace, 'head')
    head.text = head_text

    for paragraph in paragraphs:
        p = ET.SubElement(dedicace, 'p')
        p.text = paragraph

    signed = ET.SubElement(dedicace, 'signed')
    signed.text = signed_text


def write_preface(front_node, head_text, paragraphs):
    """Adds the preface section (<div type="preface">)."""
    preface = ET.SubElement(front_node, 'div', attrib={'type': 'preface'})
    head = ET.SubElement(preface, 'head')
    head.text = head_text

    for paragraph in paragraphs:
        p = ET.SubElement(preface, 'p')
        p.text = paragraph


def write_cast_list(front_node):
    """Adds the <castList> node."""
    cast_list = ET.SubElement(front_node, 'castList')


# TODO: add if X is not None pour tous les élements
# TODO : Restructure the content part into referring counters ?
def write_front_content(front_node, metadata, counters):
    """
    Adds the required structure as children nodes of the given <front> node.

    Args:
        front_node (Element): The <front> node where the content will be added.
        content (dict): Dictionary containing the content to populate the nodes.
    """
    # Add docTitle
    doc_title = ET.SubElement(front_node, 'docTitle')
    write_title_parts(doc_title, metadata['main_title'], metadata.get('sub_title',None))

    # Add docDate
    write_doc_date(front_node, metadata['doc_date_text'], metadata['date_print'])

    # Add docImprint and its children
    doc_imprint = ET.SubElement(front_node, 'div', attrib={'type': 'docImprint'})

    if 'privilege_head' in metadata and 'privilege_text' in metadata:
        write_privilege(doc_imprint, metadata['privilege_head'], metadata['privilege_text'])
    if 'acheve_imprime_text' in metadata:
        write_acheve_imprime(doc_imprint, metadata['acheve_imprime_text'])
    if 'printer_text' in metadata:
        write_printer(doc_imprint, metadata['printer_text'])

    # Add performance
    write_performance(front_node, metadata['premiere_text'], metadata['date_premiere'], metadata['premiere_location'])

    # Add dedication (dedicace)
    if counters['dedicaceFound']:
        metadata['signed_text'] = metadata['author_forename'] + ' ' + metadata['author_surname'] #TODO: CHANGE THIS ONCE SIGNED DETECTION WORKS PROPERLY
        write_dedicace(front_node, metadata['dedicaceSalute'], metadata['dedicaceHeader'], metadata['dedicace'],
                     metadata['signed_text'])

    # Add preface
    if counters['prefaceFound']:
        write_preface(front_node, metadata['prefaceHeader'], metadata['preface'])

    # Add castList
    write_cast_list(front_node)


# WRITE CONTENT OF THE PLAY

def write_scene_beginning(act_node, scene_number, scene_header):
    scene_node = ET.SubElement(act_node, 'div', attrib={'type': "scene", "xml:id": str(scene_number)})
    head = ET.SubElement(scene_node, 'head')
    head.text = scene_header
    return scene_node


def write_scene(scene_node, scene, replique_number):
    current_node = scene_node
    for replique in scene["repliques"]:
        if replique["type"] == "Speaker":
            # Checking for first replique
            character = remove_html_tags(replique["content"])
            characterId = replique["characterId"]
            sp = ET.SubElement(scene_node, 'sp', attrib={'who': f'"#{characterId}"'})
            speaker = ET.SubElement(sp, 'speaker')
            speaker.text = character
            current_node = sp
        # TODO : Add xml id ? xml:id=\"{counters["actNb"] + str(counters["scenesInAct"]) + "-" + str(counters["repliquesinScene"])}
        if replique["type"] == "Dialogue":
            replique_number += 1
            replique_node = ET.SubElement(current_node, 'l', attrib={'n': str(replique_number)})
            replique_node.text = remove_html_tags(replique["content"])
        # TODO : add xml id ? xml:id=\"l""" + str(counters["linesInPlay"])
        if replique["type"] == "Stage":
            direction = replique["content"]
            stage_node = ET.SubElement(current_node, 'stage')
            stage_node.text = direction
    return replique_number


def write_act_beginning(body_node, act_number, act_header):
    act_node = ET.SubElement(body_node, 'div', attrib={'type': 'act', 'n': act_number})
    head_node = ET.SubElement(act_node, 'head')
    head_node.text = act_header
    return act_node


def write_scene_content(parent_node, scenes, replique_number):
    """Handles writing of scenes, whether they are within acts or standalone."""
    for scene in scenes:
        scene_number = scene["sceneNumber"]
        scene_header = scene["sceneName"]
        scene_node = write_scene_beginning(parent_node, scene_number, scene_header)
        replique_number = write_scene(scene_node, scene, replique_number)
    return replique_number


def write_act_content(body_node, act, act_number, replique_number):
    """Handles writing of acts and their associated scenes."""
    act_number_string = str(act_number) if not act["actNumber"] else act["actNumber"]
    act_name = f"ACTE {act_number_string}" if not act["actName"] else remove_html_tags_and_content(act["actName"])
    act_node = write_act_beginning(body_node, act_number_string, act_name)

    # Write scenes within the act
    replique_number = write_scene_content(act_node, act["Scenes"], replique_number)
    return replique_number


def write_body(body_node, playContent, counters):
    """Main function to handle the writing of play body."""
    replique_number = 0
    act_number = 0

    if counters["noActPlay"]:
        # No acts, only scenes directly under body
        write_scene_content(body_node, playContent, replique_number)
    else:
        # Play with acts
        for act in playContent:
            act_number += 1
            replique_number = write_act_content(body_node, act, act_number, replique_number)


# MAIN FUNCTIONS
def write_full_tei_file(content, playContent, counters):
    root = initialize_xml()
    write_tei_header(root, content, counters)
    front, body = write_text(root)
    write_front_content(front, content, counters)
    write_body(body, playContent, counters)
    return root


def output_tei_file(root, output_file):
    # Generate and print the XML for inspection
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")  # Pretty print (requires Python 3.9+)
    try:
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
    except:
        ET.dump(root)

    # Print the XML to console for quick view



if __name__ == "__main__":
    pass
