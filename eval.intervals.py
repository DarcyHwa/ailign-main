# -*- coding:utf8 -*-
# This script evaluate the accuracy of the intervals extracted with ailign to detect
# the tale aligning in Grimm corpus

import re
import os


file_ref="eval/Grimm/gold/intervals.de-fr.ref"
file_sys="eval/Grimm/ailign/KHM.de-fr.intervals.txt"

ref={}
ref_size=0

f_ref=open(file_ref,mode="r")
for line in f_ref:
    m=re.search(r'^(\d+)-(\d+)\t(\d+)-(\d+)',line)
    if m:
        x1=int(m.group(1))
        x2=int(m.group(2))
        y1=int(m.group(3))
        y2=int(m.group(4))
        x_int=(x1,x2)
        y_int=(y1,y2)
        ref_size+=(x2-x1+1)*(y2-y1+1)
        for x in range(x1,x2+1):
            for y in range(y1,y2+1):
                ref[(x,y)]=1
f_ref.close()

nb_ok=0
nb_wr=0
f_sys=open(file_sys,mode="r")
for line in f_sys:
    m=re.search(r'^(\d+)-(\d+)\t(\d+)-(\d+)',line)
    if m:
        x1=int(m.group(1))
        x2=int(m.group(2))
        y1=int(m.group(3))
        y2=int(m.group(4))
        x_int=(x1,x2)
        y_int=(y1,y2)
        for x in range(x1,x2+1):
            for y in range(y1,y2+1):
                if (x,y) in ref:
                    nb_ok+=1
                else:
                    nb_wr+=1

f_sys.close()

precision=nb_ok/(nb_ok+nb_wr)
recall=nb_ok/ref_size
f1=2*precision*recall/(precision+recall)

print(f"{precision=}\t{recall=}\t{f1=}")
