import os
import numpy as np
import shutil

x=[]
for s in os.listdir('frames/'):
    # if s[-4:] == '.png':
    #     shutil.copyfile('frames/'+s, 'img/'+s)
    if s[-4:] == '.txt':
        shutil.copyfile('frames/' + s, 'annot/' + s)

