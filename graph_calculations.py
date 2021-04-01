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

play = open("GILBERT_RODOGUNE.xml", 'rb')
mydoc = minidom.parse(play)
tags = mydoc.getElementsByTagName('castItem')

#Returns the list of identifiers of characters 
def get_characters(doc):
    id_list=[]
    char_list=doc.getElementsByTagName('castItem')
    for c in (char_list):
        id_list.append(c.firstChild.getAttribute("id"))
    return(id_list)
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

#returns the list of number of co-occurences
def get_co_occurences(setlist,characters):
    k=len(characters)
    return([[len(setlist[i].intersection(setlist[j])) for i in range(k)]for j in range(k)])

co_occurences = get_co_occurences(setlist, characters)
#returns the list of the number of times a character occurs in the play
def character_occurences (characters,A) :
  return([ sum(A[i][j] for i in range(len(A))) for j in range(len(characters))])

occurences=character_occurences(characters, A)

#returns the list of the number of times a character occurs in the play, normalized by the number of scenes
def character_frequencies(characters,A) :
  return([ sum(A[i][j] for i in range(len(A)))/(len(A)) for j in range(len(characters))])

frequencies=character_frequencies(characters, A)

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
print(deviations)

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

character_speech_frequencies=[x/total_lines for x in lines]

with open('stats.csv', mode='w') as csv_file:
    fieldnames = ['VALEUR']+ characters
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    scenedict= {name : setlist[characters.index(name)] for name in characters}
    scenedict['VALEUR']= "Scènes"
    writer.writerow(scenedict)
   
    freqdict ={ name : frequencies[characters.index(name)] for name in characters}
    freqdict['VALEUR']= "Fréquences"
    writer.writerow(freqdict)


    
    
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
        