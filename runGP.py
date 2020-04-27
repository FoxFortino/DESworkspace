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
    GP.fit(GP.opt_result[0])
    
    GP.predict(dataC.Xvalid)
    mask = stats.sigma_clip(
        dataC.Yvalid - dataC.fbar_s,
        sigma=4, axis=0).mask
    mask = ~np.logical_or(*mask.T)
    dataC.Xvalid = dataC.Xvalid[mask]
    dataC.Yvalid = dataC.Yvalid[mask]
    dataC.Evalid_DES = dataC.Evalid_DES[mask]
    dataC.Evalid_GAIA = dataC.Evalid_GAIA[mask]
    
    GP.predict(dataC.Xtrain)
    mask = stats.sigma_clip(
        dataC.Ytrain - dataC.fbar_s,
        sigma=4, axis=0).mask
    mask = ~np.logical_or(*mask.T)
    dataC.Xtrain = dataC.Xtrain[mask]
    dataC.Ytrain = dataC.Ytrain[mask]
    dataC.Etrain_DES = dataC.Etrain_DES[mask]
    dataC.Etrain_GAIA = dataC.Etrain_GAIA[mask]
    
    GP.fitCorr(v0=GP.opt_result[0])
    GP.optimize()
    GP.fit(GP.opt_result_GP[0])
    GP.predict(dataC.Xvalid)

    dataC.saveNPZ(expFile)


if __name__=='__main__':
    expNum = int(sys.argv[1])
    outDir = sys.argv[2]

    main(expNum, outDir)