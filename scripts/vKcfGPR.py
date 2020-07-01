import sys
import os
import shutil
import glob
import traceback
# import warnings
# warnings.filterwarnings("ignore")

import GPRutils
import vK2KGPR
import plotGPR
import vonkarmanFT as vk

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.modeling import models, fitting


expNum = 348819
expFile = os.path.join("/home/fortino/vKcfGPR", str(expNum))

try:
    os.mkdir(expFile)
except FileExistsError:
    shutil.rmtree(expFile)
    os.mkdir(expFile)

sys.stderr = open(os.path.join(expFile, "err.err"), "a+")
sys.stdout = open(os.path.join(expFile, "out.out"), "a+")

try:
    dataC = GPRutils.dataContainer()

    dataC.load(expNum)

    GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=expFile, curlfree=True)
    GP.fitCorr()
    GP.fit(dataC.fitCorrParams)
    dataC.postFitCorr_sigmaClip(GP)
    GP.optimize(v0=dataC.fitCorrParams)
    dataC.JackKnife(GP)
    dataC.saveFITS(expFile)
except:
    traceback.print_exc()


try:
    dataC = GPRutils.dataContainer()

    dataC.load(expNum, useRMS=True)

    GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=expFile, curlfree=True)
    GP.fitCorr()
    GP.fit(dataC.fitCorrParams)
    dataC.postFitCorr_sigmaClip(GP)
    GP.optimize(v0=dataC.fitCorrParams)
    dataC.JackKnife(GP)
    dataC.saveFITS(expFile)
except:
    traceback.print_exc()


try:
    dataC = GPRutils.dataContainer()

    dataC.load(expNum, useRMS=True, useCov=True)

    GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=expFile, curlfree=True)
    GP.fitCorr()
    GP.fit(dataC.fitCorrParams)
    dataC.postFitCorr_sigmaClip(GP)
    GP.optimize(v0=dataC.fitCorrParams)
    dataC.JackKnife(GP)
    dataC.saveFITS(expFile)
except:
    traceback.print_exc()


try:
    dataC = GPRutils.dataContainer()

    dataC.load(expNum, maxDESErr=25*u.mas**2)

    GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=expFile, curlfree=True)
    GP.fitCorr()
    GP.fit(dataC.fitCorrParams)
    dataC.postFitCorr_sigmaClip(GP)
    GP.optimize(v0=dataC.fitCorrParams)
    dataC.JackKnife(GP)
    dataC.saveFITS(expFile)
except:
    traceback.print_exc()
