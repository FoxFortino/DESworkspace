import os
import shutil
import sys

import GPRutils
import vK2KGPR
import plotGPR

import numpy as np
import astropy.units as u

def main(expNum, outDir):

    expFile = os.path.join(outDir, str(expNum))
    try:
        os.mkdir(expFile)
    except FileExistsError:
        shutil.rmtree(expFile)
        os.mkdir(expFile)

    dataC = GPRutils.dataContainer()
    dataC.load(expNum=expNum)
    dataC.splitData()

    GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=expFile)
    GP.fitCorr()
    GP.optimize()
    GP.fit(GP.opt_result_GP[0])
    GP.predict(dataC.Xvalid)

    dataC.saveNPZ(expFile)


if __name__=='__main__':
    expNum = int(sys.argv[1])
    outDir = sys.argv[2]

    main(expNum, outDir)