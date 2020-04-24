import os
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

    dataC = GPRutils.dataContainer(expNum)
    dataC.sigmaClip()
    dataC.splitData()

    GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=outDir)
    GP.fitCorr()
    GP.optimize()
    GP.fit(GP.opt_result_GP[0])
    GP.predict(dataC.Xpred)

    dataC.saveNPZ(expFile)

    # Don't include all this plotting. Plotting can be done after the fact.
    # Besides, probably don't want all these plots anyway. Just the data
    # inside them in order to average them together from a bunch of exposures
    # and do statistics. Have a separate script for going through directories
    # and making plots of exposures if we want to.
    # x = dataC.Xvalid[:, 0]*u.deg
    # y = dataC.Xvalid[:, 1]*u.deg
    # dx = dataC.Yvalid[:, 0]*u.mas
    # dy = dataC.Yvalid[:, 1]*u.mas
    # err = dataC.Evalid[:, 0]*u.mas

    # x2 = dataC.Xvalid[:, 0]*u.deg
    # y2 = dataC.Xvalid[:, 1]*u.deg
    # dx2 = dataC.Yvalid[:, 0]*u.mas - dataC.fbar_s[:, 0]*u.mas
    # dy2 = dataC.Yvalid[:, 1]*u.mas - dataC.fbar_s[:, 1]*u.mas
    # err2 = dataC.Evalid[:, 0]*u.mas

    # plotGPR.AstrometricResiduals(
    #     x, y, dx, dy, err,
    #     x2=x2, y2=y2, dx2=dx2, dy2=dy2, err2=err2,
    #     savePath=outDir,
    #     plotShow=False,
    #     exposure=expNum,
    #     scale=200*u.mas,
    #     arrowScale=10*u.mas)

    # plotGPR.DivCurl(
    #     x, y, dx, dy, err,
    #     x2=x2, y2=y2, dx2=dx2, dy2=dy2, err2=err2,
    #     savePath=outDir,
    #     plotShow=False,
    #     exposure=expNum,
    #     pixelsPerBin=1500)

    # plotGPR.Correlation(
    #     x, y, dx, dy,
    #     x2=x2, y2=y2, dx2=dx2, dy2=dy2,
    #     savePath=outDir,
    #     plotShow=False,
    #     exposure=expNum,
    #     ylim=(-20, 75))

    # plotGPR.Correlation2D(
    #     x, y, dx, dy,
    #     x2=x2, y2=y2, dx2=dx2, dy2=dy2,
    #     savePath=outDir,
    #     plotShow=False,
    #     exposure=self.expNum,
    #     nBins=50,
    #     vmin=0*u.mas**2,
    #     vmax=40*u.mas**2,
    #     rmax=0.50*u.deg)


if __name__=='__main__':
    expNum = sys.argv[1]
    outDir = sys.argv[2]

    main(expNum, outDir)