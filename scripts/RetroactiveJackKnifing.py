import os
import glob
import sys
import traceback

import GPRutils
import vK2KGPR

import numpy as np


def JackKnifeOld(npz, file):
    print(f"Jack knifing file {npz}")
    dC0 = GPRutils.loadNPZ(npz)
    dCf = GPRutils.dataContainer()
    dCf.load(dC0.expNum)
    GP = vK2KGPR.vonKarman2KernelGPR(dCf)
    GP.fitCorr()
    GP.fit(GP.opt_result[0])
    dCf.postFitCorr_sigmaClip(GP)
    dCf.params = dC0.params
    dCf.JackKnife(GP)
    dCf.saveFITS(file)
    
files = sorted(glob.glob("/home/fortino/thesis/??????"))

# Remove files that have already been done for some reason
for i, file in enumerate(files.copy()):
    fits = glob.glob(os.path.join(file, "*.fits"))
    if fits:
        files.remove(file)
        
sys.stderr = open("../thesis/err.err", "a+")
sys.stdout = open("../thesis/out.out", "a+")

for file in files:
    path, exp = os.path.split(file)
    npz = os.path.join(file, exp+".npz")
    if not os.path.isfile(npz):
        continue
        
    try:
        JackKnifeOld(npz, file)
    except:
        traceback.print_exc()