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

from importlib import reload

gBand = sorted(np.array([
    474262,
    361580,
    474265,
    696547,
    484483,
    367484,
    369801,
    576863,
    791229,
    791593,
    791640,
    579816
]))

for expNum in gBand[:3]:
    
    expFile = os.path.join("/home/fortino/gBand_useCov", str(expNum))
    try:
        os.mkdir(expFile)
    except FileExistsError:
        shutil.rmtree(expFile)
        os.mkdir(expFile)

    sys.stderr = open(os.path.join(expFile, "err.err"), "a+")
    sys.stdout = open(os.path.join(expFile, "out.out"), "a+")
    
    try:
        dataC = GPRutils.dataContainer()

        dataC.load(expNum, useRMS=True, useCov=True)

        GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=expFile)
        GP.fitCorr()
        GP.fit(dataC.fitCorrParams)
        dataC.postFitCorr_sigmaClip(GP)
        GP.optimize(v0=dataC.fitCorrParams)
        dataC.JackKnife(GP)
        dataC.saveFITS(expFile)
    except:
        traceback.print_exc()
