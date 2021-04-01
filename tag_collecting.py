# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:48:40 2021

@author: aaron
"""

from xml.dom import minidom
import glob, os, re, sys, requests, math, csv




folder = os.path.abspath(os.path.dirname(sys.argv[0]))
corpusFolder = os.path.join(folder,"corpus")

def collect_tags (tag, corpus):
    taglist=[]
    for c in os.listdir(corpus):
        print(c)
        play = open(corpusFolder +'\\'+ c, 'rb')
        mydoc = minidom.parse(play)
        tags = mydoc.getElementsByTagName(tag)
        for el in tags:
            if el.hasAttributes():
                attributes = el.attributes
                for i in range (attributes.length):
                    attr_name = attributes.item(i).name
                    attr_val = el.getAttribute(attr_name)
                    attr = (attr_name, attr_val) 
                    if attr not in taglist:
                        taglist.append(attr)
                    else:
                        
    return(taglist)
didascalist=collect_tags("stage",corpusFolder)
                    
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
                    

              