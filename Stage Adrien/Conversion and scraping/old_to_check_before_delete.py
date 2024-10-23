def write_title(outputFile, title):
    """Write the extracted title in the output file in XML.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        title (str): Title of a file.

    Returns:
        str: The same title.
    """
    if title:
        outputFile.writelines("""<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:lang="fre">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title type="main">""" + title + """</title>""")
    return title

def write_type(outputFile, genre):
    """Write the extracted genre in the output file in XML.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        genre (str): Genre of a play.
    """
    if genre != '[indéfini]':
        outputFile.writelines("""
                <title type="sub">""" + genre + """</title>""")

        def write_author(outputFile, author):
            """Write the author's name in the output file in XML.

            Args:
                outputFile (TextIOWrapper): Output file to generate in XML.
                author (str): Author of the play.

            Returns:
                bool: True if the author's name have at least a forename or a surname, False then.
            """
            forename, surname = author
            if forename or surname:
                outputFile.writelines("""
                    <author>
                        <persName>""")
                if forename:
                    for name in forename:
                        if name in ['de', "d'"]:
                            outputFile.writelines("""
                            <linkname>""" + name + """</linkname>""")
                        elif name in ['Abbé']:  # TODO identifier d'autres rolename
                            outputFile.writelines(f"""
                            <rolename>{name}</rolename>""")
                        else:
                            outputFile.writelines(f"""
                            <forename>{name}</forename>""")
                if surname:
                    for name in surname:
                        if name in ['DE', "D'"]:
                            outputFile.writelines("""
                            <linkname>""" + name.lower() + """</linkname>""")
                        else:
                            outputFile.writelines("""
                            <surname>""" + ''.join([name[0], name[1:].lower()]) + """</surname>""")

                outputFile.writelines("""
                        </persName>
                    </author>
                                <editor>Adrien Roumégous, dans le cadre d'un stage de M1 Informatique encadré par Aaron Boussidan et Philippe Gambette.</editor>
                    </titleStmt>""")
                return True
            return False

def write_source(outputFile, source):
    """Write the source of the play in the output file in XML.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        source (str): Source of the play.
    """
    outputFile.writelines(f"""
            <publicationStmt>
                <publisher xml:id="dracor">DraCor</publisher>
                <idno type="URL">https://dracor.org</idno>
                <idno type="dracor" xml:base="https://dracor.org/id/">fre[6 chiffres]</idno>
                <idno type="wikidata" xml:base="http://www.wikidata.org/entity/">Q[id]</idno>
                <availability>
                    <licence>
                        <ab>CC BY-NC-SA 4.0</ab>
                        <ref target="https://creativecommons.org/licenses/by-nc-sa/4.0/">Licence</ref>
                    </licence>
                </availability> 
            </publicationStmt>
            <sourceDesc>
                <bibl type="digitalSource">
                    <name>Théâtre Documentation</name>
                    <idno type="URL"> {source} </idno>
                    <availability>
                        <licence>
                            <ab>loi française n° 92-597 du 1er juillet 1992 et loi n°78-753 du 17 juillet 1978</ab>
                            <ref target="http://théâtre-documentation.com/content/mentions-l%C3%A9gales#Mentions_legales">Mentions légales</ref>
                        </licence>
                    </availability>
                    <bibl type="originalSource">""")

def write_dates(outputFile, date_written, date_print, date_premiere, line_premiere):
    """Write the date of writing, the date of printing and the date of first performance of the play, and the line of context for each of them in an output file in XML.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        date_written (str): Date of writing of the play.
        date_print (str): Date of printing of the play.
        date_premiere (str): Date of first performance of the play.
        line_premiere (str): Line where the date of the first performance is written.
    """
    if date_written != "[vide]":
        outputFile.writelines("""
                        <date type="written" when=\"""" + date_written + """\">""")

    if date_print != "[vide]":
        outputFile.writelines("""
                        <date type="print" when=\"""" + date_print + """\">""")

    if date_premiere != "[vide]":
        if type(date_premiere) is str:
            outputFile.writelines("""
                        <date type="premiere" when=\"""" + date_premiere + """\">""" + line_premiere + """</date>""")
        else:
            outputFile.writelines("""
                        <date type="premiere" notBefore=\"""" + date_premiere[0] + """\" notAfter=\"""" + date_premiere[
                1] + """\" >""" + line_premiere + """</date>""")

    outputFile.writelines("""
                        <idno type="URL"/>
                    </bibl>
                </bibl>""")

def write_end_header(outputFile, genre, vers_prose):
    # TODO : Generate a better listPerson
    """Write the end of the header of a XML file

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        genre (str) : The genre of the converted play.
        vers_prose (str) : The type of the converted play, in verses or in prose.
    """
    outputFile.writelines(f"""
            </sourceDesc>
        </fileDesc>
        <profileDesc>
            <particDesc>
                <listPerson>""")

    for charaid, charaname in zip(counters["characterIDList"], counters["characterFullNameList"]):
        outputFile.writelines(f"""
                        <person xml:id="{charaid}" sex="SEX">
                            <persName>{charaname}</persName>
                        </person>""")
    # TODO : Get character sex from character name
    # TODO : Add Pastorale,Dialogue code and more genres ?
    wikidata_codes = {'Tragi-Comédie': 192881,
                      'Farce': 193979, 'Tragédie': 80930, 'Comédie': 40831,
                      'Vaudeville': 186286, 'Farce': 193979, "Proverbe": 2406762, 'Pastorale': 0000, "Dialogue": 00000}
    if genre in wikidata_codes:
        wikicode = wikidata_codes[genre]
    else:
        if genre != '[indéfini]':
            print(f"UNKNOW GENRE : {genre}")
        wikicode = None
    wikicode_part = ["", f"""
                <classCode scheme="http://www.wikidata.org/entity/">[Q{wikicode}]</classCode>"""]
    outputFile.writelines(f"""
                </listPerson>
            </particDesc>
            <textClass>
            <keywords scheme="http://theatre-documentation.fr"> <!--extracted from "genre" and "type" elements-->
                    <term> {genre}</term>
                    <term> {vers_prose} </term>
                </keywords>{wikicode_part[wikicode is not None]}
            </textClass>
        </profileDesc>
        <revisionDesc>
            <listChange>
                <change when="{date.today()}">(mg) file conversion from source</change>
            </listChange>
        </revisionDesc>
   </teiHeader>""")

def write_start_text(outputFile, title, genre, date_print):
    """Write the start of the text body of a XML file.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        title (str) : The title of the converted play.
        genre (str) : The genre of the converted play.
        date_print (str) : The date of printing of the converted play.
    """
    outputFile.writelines("""
    <text>
    <front>
        <docTitle>
            <titlePart type="main">""" + title.upper() + """</titlePart>""")
    if genre:
        outputFile.writelines("""
            <titlePart type="sub">""" + genre.upper() + """</titlePart>
        </docTitle>""")
    if date_print:
        outputFile.writelines("""
        <docDate when=\"""" + date_print + """\">[Date Print Line]</docDate>
        """)
def write_performance(outputFile, line_premiere, date_premiere):
    """Write the performance tag of the chosen XML file.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
        line_premiere (str) : line of the play where the date of the first performance is written.
        date_premiere (str) : The date of the first performance of the converted play.
    """
    if date_premiere != '[vide]':
        if type(date_premiere) is tuple:
            date_premiere = '-'.join(date_premiere)
        outputFile.writelines("""
        <performance>
            <ab type="premiere">""" + line_premiere + """</ab><!--@date=\"""" + date_premiere + """\"-->
        </performance>""")

def write_dedicace(dedicace, dedicaceHeader, file):
    file.writelines(f"""
    <div type="dedicace">
            <opener>
                <salute> {dedicaceHeader}</salute>
            </opener>""")
    for index, line in enumerate(dedicace):
        # if index == len(dedicace)-1 and
        file.writelines(f"""
        <p> {line} </p>""")
    file.writelines(f"""
    </div>""")

def write_character(outputFile):
    """Write the saved characters of a play in the associated XML file.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
    """
    outputFile.writelines("""
        <castList>
                    <head> ACTEURS </head>""")
    for i, character in enumerate(counters["characterIDList"]):
        outputFile.writelines(f"""
            <castItem>
                    <role corresp="#{character}">{counters["characterFullNameList"][i]} </role>{counters["roleList"][i]}</castItem>"""
                              )
    outputFile.writelines("""
        </castList>
        </front>""")

def write_act_beginning(act_number, act_header, file):
    file.writelines(f"""
           <div type="act" xml:id=\"{act_number}\">
           <head> {act_header} </head>""")


def write_act_end(file):
    file.writelines("""</div>""")


def write_scene_beginning(scene_number, scene_header, file):
    file.writelines(f"""
        <div type="scene" xml:id=\" {scene_number} \">
            <head> {scene_header} </head>""")


def write_scene(scene, replique_number, file):
    """Writes all dialogue, speakers, and stage direction to the output file. Also returns the current number of repliques"""
    sp_opened = False
    for replique in scene["repliques"]:
        if replique["type"] == "Speaker":
            # Checking for first replique
            if sp_opened:
                file.writelines("""
                </sp>""")

            character = remove_html_tags(replique["content"])
            characterId = replique["characterId"]
            file.writelines(f"""
        <sp who=\"#{characterId}\">
            <speaker> {character} </speaker>""")
            sp_opened = True
        # TODO : Add xml id ? xml:id=\"{counters["actNb"] + str(counters["scenesInAct"]) + "-" + str(counters["repliquesinScene"])}
        if replique["type"] == "Dialogue":
            replique_number += 1
            outputFile.writelines(f"""
                        <l n=\"{replique_number}\"> {remove_html_tags(replique["content"])}</l>""")
        # TODO : add xml id ? xml:id=\"l""" + str(counters["linesInPlay"])
        if replique["type"] == "Stage":
            direction = replique["content"]
            outputFile.writelines(f"""
            <stage>{direction}</stage>""")
    if sp_opened:
        file.writelines("""
         </sp>""")
    return replique_number


def write_scene_end(outputFile):
    outputFile.writelines("""
    </div>""")


def write_play(outputFile, playContent, counters):
    act_number = 0
    replique_number = 0
    if counters["noActPlay"]:
        for scene in playContent:
            scene_number = scene["sceneNumber"]
            scene_header = scene["sceneName"]
            write_scene_beginning(scene_number, scene_header, outputFile)
            replique_number = write_scene(scene, replique_number, outputFile)
            write_scene_end(outputFile)
    else:
        for act in playContent:
            act_number += 1
            # Collecting things to write
            if not act["actNumber"]:
                act_number_string = str(act_number)
            else:
                act_number_string = act["actNumber"]
            if not act["actName"]:
                act_name = f"ACTE {act_number_string}"
            else:
                act_name = remove_html_tags_and_content(act["actName"])
            write_act_beginning(act_number_string, act_name, outputFile)
            for scene in act["Scenes"]:
                scene_number = scene["sceneNumber"]
                scene_header = scene["sceneName"]
                write_scene_beginning(scene_number, scene_header, outputFile)
                replique_number = write_scene(scene, replique_number, outputFile)
                write_scene_end(outputFile)
            write_act_end(outputFile)


def write_end(outputFile):
    """Write the end of the XML output file.

    Args:
        outputFile (TextIOWrapper): Output file to generate in XML.
    """
    outputFile.writelines("""
         <p>FIN</p>
      </div>
      </div>
   </body>
</text>
</TEI>""")