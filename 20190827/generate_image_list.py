import os
import glob

a=glob.glob("./**/lwir/*.jpg", recursive=True)
b=[]
with open("train.txt", 'w') as f:
    for i in a:
        f.write(i[2:-16]+i[-11:-4]+"\n")