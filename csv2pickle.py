import os
import glob
import pandas as pd 
import numpy as np
import pickle
import time


for path in glob.glob("./data_CMU/*/"):
    for path2 in glob.glob(path+'*.csv'):

        if(path2 not in path):
            print(path2)
            dat = pd.read_csv(path2)
            dat = dat.drop(['Time'], axis=1)

            coors = []
            coors_final = []
            
            for col in dat.columns:
                coors.append(dat[col].values.tolist())
                
            for i in range(len(coors[0])):
                coord = []
                for j in range(len(coors)//3):
                    coord.append([coors[j*2][i], coors[j*2+1][i], coors[j*2+2][i]])
                coors_final.append(coord), 


            pickle_out = open('pickle_data'+path2[path2.rindex("\\"):len(path2)-3]+'pickle',"wb")
            pickle.dump(coors_final, pickle_out)
            pickle_out.close()