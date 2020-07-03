import os
import sys
import shutil
import traceback

import GPRutils
import vK2KGPR
import DESutils

import numpy as np

def main(expNum, expFile):
    dataC = GPRutils.dataContainer()
    dataC.load(expNum)
    GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=expFile)
    GP.fitCorr()
    GP.fit(dataC.fitCorrParams)
    dataC.postFitCorr_sigmaClip(GP)
    GP.optimize()
    dataC.JackKnife(GP)
    dataC.saveFITS(expFile)
    
if __name__=='__main__':
    exps = DESutils.findExpNums()

    for expNum in exps[:5]:
        expFile = os.path.join("/home/fortino/test_N", str(expNum))
        try:
            os.mkdir(expFile)
        except FileExistsError:
            shutil.rmtree(expFile)
            os.mkdir(expFile)
        
        main(expNum, expFile)

