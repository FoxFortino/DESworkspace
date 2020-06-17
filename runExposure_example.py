import GPRutils
import vK2KGPR


outDir = # Typically a folder with the name of the exposure. All data saved here.

dataC = GPRutils.dataContainer()
dataC.load(348819)
GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=outDir)
GP.fitCorr()
GP.fit(GP.opt_result[0])
dataC.postFitCorr_sigmaClip(GP)
GP.optimize()
dataC.JackKnife(GP)
dataC.saveFITS(outDir)