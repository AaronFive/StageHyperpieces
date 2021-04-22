# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:19:25 2021

@author: aaron
"""

import glob, os, re, sys, requests, math, csv
from xml.dom import minidom
# Get the current folder
folder = os.path.abspath(os.path.dirname(sys.argv[0]))
corpusFolder = os.path.join(folder,"corpus")

playname = "corneillep_rodogune"
play = open(playname+".xml", 'rb')
mydoc = minidom.parse(play)
tags = mydoc.getElementsByTagName('castItem')

#Returns the list of identifiers of characters, when the list is declared at the start
def get_characters_by_cast(doc):
    id_list=[]
    char_list=doc.getElementsByTagName('role')
    for c in (char_list):
        name_id= c.getAttribute("id")
        if name_id=="":
            name_id=c.getAttribute("xml:id")
        if name_id=="":
            print("Warning, role has no id nor xml:id attribute")
        id_list.append(c.getAttribute("id"))
    return(id_list)


#Returns the list of identifiers of characters, by looking at each stance
def get_characters_by_bruteforce(doc):
    id_list=[]
    repliques=doc.getElementsByTagName('sp')
    for r in repliques:
        speaker_id= r.getAttribute("who")
        if speaker_id not in id_list:
            id_list.append(speaker_id)
    return(id_list)

def get_characters(doc):
    char_list=doc.getElementsByTagName('role')
    if char_list == []:
        return(get_characters_by_bruteforce(doc))
    else:
        return(get_characters_by_cast(doc))
characters=get_characters(mydoc)
### HERE I COMPUTE THNGS BASED ONLY ON PRESENCE/ABSENCE FROM A SCENE ###

#Returns a matrix A such that A[i][j]=1 iff character j SPEAKS in scene i
#Note that this might yield a different result than considering the scenes where characts APPEAR
def get_matrix(doc, characters):
    scene_list = doc.getElementsByTagName('div2')
    A=[]
    scene_number=-1
    for s in scene_list:
        if s.getAttribute("type")=="scene":
            scene_number+=1
            A.append([0 for i in range(len(characters))])
            speakers= s.getElementsByTagName("sp")
            for sp in speakers:
                speaker_name=sp.getAttribute("who")
                speaker_number = characters.index(speaker_name)
                if A[scene_number][speaker_number]==0:
                     A[scene_number][speaker_number]=1
    return(A)
A=get_matrix(mydoc, characters)


#returns character density up to scene k
def character_density(A,k,characters):
    chi=0
    m=len(characters)
    for i in range(k):
        for j in range(m):
            chi+=A[i][j]
    return(chi/(m*(k+1)))

#returns the list of sets of scenes in which characters appear
def get_scenes_sets (characters,A):
        return([set([i for i in range(len(A)) if A[i][char]==1 ])for char in range(len(characters))])
setlist=get_scenes_sets(characters, A)

#returns the list of the number of times a character occurs in the play
def character_occurences (characters,A) :
  return([ sum(A[i][j] for i in range(len(A))) for j in range(len(characters))])
occurences=character_occurences(characters, A)

#returns the list of the number of times a character occurs in the play, normalized by the number of scenes
def character_frequencies(characters,A) :
  return([ sum(A[i][j] for i in range(len(A)))/(len(A)) for j in range(len(characters))])
frequencies=character_frequencies(characters, A)

#returns the list of number of co-occurences
def get_co_occurences(setlist,characters):
    k=len(characters)
    return([[len(setlist[i].intersection(setlist[j])) for i in range(k)]for j in range(k)])
co_occurences = get_co_occurences(setlist, characters)

def occurences_deviation(A,characters, co_occurences, occurences):
    k=len(characters)
    n=len(A)
    deviations=[[0 for i in range(k)] for j in range(k)]
    for i in range(k):
        for j in range(k):
            qi,qj=occurences[i],occurences[j]
            nij=co_occurences[i][j]
            deviations[i][j]=nij-n*qi*qj
            if (n*qi*qj)!=0:
                deviations[i][j]=(deviations[i][j])**2/(n*qi*qj) #normalisation step
    return(deviations)
deviations=occurences_deviation(A,characters, co_occurences, frequencies)

#Returns the list of confrontation factors for each character
#The confrontation factor of x is the sum over y!=x of (number of scenes where x and y appear)
def get_confrontations(A, characters):
    setlist=get_scenes_sets(characters, A)
    co_occurences= get_co_occurences(setlist, characters)
    confrontations=[0 for i in range(len(characters))]
    for i in range(len(co_occurences)):
        l=co_occurences[i]
        conf=sum(l)-l[i]
        confrontations[i]=conf
    tot= sum(confrontations)
    return([x/tot for x in confrontations])
confrontations = get_confrontations(A, characters)


###HERE I COMPUTE THINGS BASED ON THE NUMBER OF LINES SPOKEN ###

def get_lines(doc, characters):
    total_lines=0
    lines=[0 for i in range(len(characters))]
    repliques = doc.getElementsByTagName('sp')
    for r in repliques :
        speaker_name=r.getAttribute("who")
        speaker_number=characters.index(speaker_name)
        for phrases in r.childNodes:
            if phrases.nodeName == "l":
                total_lines+=1
                lines[speaker_number]+=1
    return(total_lines, lines)
total_lines, lines = get_lines(mydoc, characters)
speech_frequency=[x/total_lines for x in lines]

#Returns the list of succesions
#succesions[i][j] is the number of time character i has spoken after character j 
def get_successions(doc, characters):
    n=len(characters)
    successions=[[0 for i in range(n)  ] for j in range(n)]
    repliques = doc.getElementsByTagName('sp')
    previous_speaker_number= -1
    for r in repliques :
        speaker_name=r.getAttribute("who")
        speaker_number=characters.index(speaker_name)
        if not(previous_speaker_number == -1):
            successions[speaker_number][previous_speaker_number]+=1
        previous_speaker_number = speaker_number     
    return(successions)


successions= get_successions(mydoc, characters)
normalized_succesions= [[x/sum(l) for x in l] for l in successions]


 ### OUTPUT TO CSV : CONVERTING TO DICTIONNARIES AND WRITING
def list_to_dict(l, characters, name):
    d= {name : l[characters.index(name)] for name in characters}
    d['VALEUR']= name
    return(d)
def table_to_dict(l, characters, value):
    d={name : name for name in characters}
    d['VALEUR']= value
    dicts=[d]
    for c in l : 
        dicts.append(list_to_dict(c, characters, characters[l.index(c)]))
    return(dicts)
        
        

with open('stats'+playname+'.csv', mode='w') as csv_file:
    fieldnames = ['VALEUR']+ characters
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    freqdict=list_to_dict(frequencies, characters, "Fréquences (scènes)")
    linesdict=list_to_dict(speech_frequency, characters, "Fréquences(répliques)")
    confrdict=list_to_dict(confrontations, characters, "Confrontation")
    
    writer.writerow(freqdict)
    writer.writerow(linesdict)
    writer.writerow(confrdict)
    
    deviationsdicts=table_to_dict(deviations,characters, "Déviations")
    for d in deviationsdicts:
        writer.writerow(d)
        
    succdicts=table_to_dict(normalized_succesions,characters, "Successions")
    for d in succdicts:
        writer.writerow(d)
    
        

# folder = os.getcwd()
# corpusFolder = os.path.join(folder,"corpusTC")
# for c in os.listdir(corpusFolder):
#     print(c)
#     m,maxplay,charlist=0,"",[]
#     play = open(corpusFolder +'\\'+ c, 'rb')
#     mydoc = minidom.parse(play)
#     char=get_characters(mydoc)
#     if len(char)> m:
#         m=len(char)
#         maxplay=c
#         charlist=char
    
    
# def generate_csv(corpus, output):
#     for c in corpus:
#         play = open(c, 'rb')
#         doc = minidom.parse(play)
#         characters= get_characters(doc)
#         A = get_matrix(doc, characters)
#         setlist=get_scenes_sets(characters, A)
#         occurences=character_occurences(characters, A)
#         co_occurences = get_co_occurences(setlist, characters)
#         frequencies=character_frequencies(characters, A)
#         deviations=occurences_deviation(A,characters, co_occurences, frequencies)
        