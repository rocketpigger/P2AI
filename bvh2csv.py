import os
import glob
import pandas as pd 
import numpy as np
import pickle
import time


for path in glob.glob('./data_CMU/*/'):
    for bvh in glob.glob(path+'*.bvh'):
        bvh = bvh[bvh.index('data'):]
        cmd = "bvh-converter "+ bvh
        print(cmd)
        os.system(cmd)