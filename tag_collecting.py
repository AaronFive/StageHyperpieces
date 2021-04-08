# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:48:40 2021

@author: aaron
"""

from xml.dom import minidom
import glob, os, re, sys, requests, math, csv


#Compter les occurences
#Faire un run par corpus 

folder = os.path.abspath(os.path.dirname(sys.argv[0]))
corpus_list=["Corpus Boissy", "Corpus Bibdramatique", "Corpus Dramacode", "corpusTheatreClassique"] 
teststring ="a/bc/a/test"
    
#Return a dictionnary of the form {(attributes, value): #occurences} for the chosen tag over the corpus
def collect_tags (tag, corpus):
    tagdict=dict()
    for c in os.listdir(corpus):
        print(c)
        if re.match(".*\.xml", c):
            play = open(corpus +'\\'+ c, 'rb')
            mydoc = minidom.parse(play)
            tags = mydoc.getElementsByTagName(tag)
            for el in tags:
                if el.hasAttributes():
                    attributes = el.attributes
                    for i in range (attributes.length):
                        attr_name = attributes.item(i).name
                        attr_val = el.getAttribute(attr_name)
                        attr_val_list=re.split("/",attr_val)
                        for a in attr_val_list:
                            attr = (attr_name, a)
                            if attr not in tagdict:
                                tagdict[attr]=1
                            else:
                                tagdict[attr]+=1
    return(tagdict)

#Iterate over different corpus and save the result in an output
def create_outputs(corpuslist, tag):
    for c in corpus_list: 
        corpusFolder = os.path.join(folder,c)
        tagdict = collect_tags(tag, corpusFolder)
        output = open("Output"+tag+c+".txt", 'w+')
        output.write(str(tagdict))
create_outputs(corpus_list, "stage")
   
    # for c in os.listdir(corpusFolder):
    #     print(c)
    #     play = open(corpusFolder +'\\'+ c, 'rb')
    #     mydoc = minidom.parse(play)
    #     tags = mydoc.getElementsByTagName('role')
    #     for el in tags:
    #         if el.hasAttributes():
    #             attributes = el.attributes
    #             for i in range (attributes.length):
    #                 attr = attributes.item(i).name
    #                 if attr not in attribute_list:
    #                     attribute_list.append(attr)
                    

              