#!/usr/bin/env python

import os
import argparse
import time

import GPRutils
import vK2KGPR
import DESutils

import numpy as np
import astropy.units as u
import scipy.optimize as opt


parser = argparse.ArgumentParser(description='Run a GPR model with some arguments.')
parser.add_argument("expFile", type=str, nargs="?")
parser.add_argument("-b", "--band", type=str, choices=["Y", "g", "r", "i", "z"], nargs="?")
parser.add_argument("-s", "--start", type=int, nargs="?")
parser.add_argument("-f", "--finish", type=int, nargs="?")
parser.add_argument("-n", "--expNum", type=int, nargs="+")
parser.add_argument("-e", "--eris", action="store_true")
parser.add_argument("-v", "--vSet", type=str, choices=["Subset A", "Subset B", "Subset C", "Subset D", "Subset E"], nargs="?", default="Subset A")
parser.add_argument("-r", "--RMS", action="store_true")
parser.add_argument("-c", "--curl", action="store_true")
parser.add_argument("--min", type=float, nargs="?", default=-np.inf)
parser.add_argument("--max", type=float, nargs="?", default=np.inf)
parser.add_argument("--downselect", type=float, nargs="?", default=1)
parser.add_argument("-d", "--dualOpt", action="store_true")
parser.add_argument("-a", "--altOpt", action="store_true")
args = parser.parse_args()

def main(expNum, expFile, useRMS, curl, downselect, minDESErr, maxDESErr, vSet, dualOpt, altOpt):

    t0 = time.time()*u.s
    dataC = GPRutils.dataContainer()
    dataC.load(expNum, downselect=downselect, minDESErr=minDESErr*u.mas**2, maxDESErr=maxDESErr*u.mas**2, useRMS=useRMS, vSet=vSet)
    t1 = time.time()*u.s
    
    GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=expFile, curl=curl)
    GP.fitCorr()
    t2 = time.time()*u.s
    
    dataC.JackKnife(GP, dataC.fitCorrParams, fC=True)
    t3 = time.time()*u.s

    try:
        if dualOpt:
            # Wrapper function for keeping K and OS the same while leaving d, Wx, and Wy available as inputs.
            # K and OS will be the fitCorr params.
            FoM = lambda p: GP.figureOfMerit(np.array([*dataC.fitCorrParams[:2], *p]))
            
            # Initial guess will be d, Wx, and Wy from fitCorr.
            v0 = dataC.fitCorrParams[2:]
            
            # Call the GP optimize method with the different figure of merit function
            GP.optimize(v0=v0, func=FoM, ftol=0.05, maxfun=50)
            d_Wx_Wy = GP.opt_result_GP[0].copy()

            # Wrapper function for keeping d, Wx, and Wy the same while leaving K and OS available as inputs.
            # d, Wx, and Wy will be the values from the first stage of optimization.
            FoM = lambda p: GP.figureOfMerit(np.array([*p, *d_Wx_Wy]))
            
            # Initial guess will be K and OS from fitCorr.
            v0 = dataC.fitCorrParams[:2]
            
            # Call the GP optimize method with the different figure of merit function
            GP.optimize(v0=v0, func=FoM, ftol=0.05, maxfun=50)
            K_oS = GP.opt_result_GP[0].copy()

            dataC.params = np.array([*K_oS, *d_Wx_Wy])
            
        elif altOpt:
            dataC.fitCorrParams = np.array([
                np.abs(dataC.fitCorrParams[0]),
                np.abs(dataC.fitCorrParams[1]),
                np.abs(dataC.fitCorrParams[2]),
                dataC.fitCorrParams[3],
                dataC.fitCorrParams[4]
            ])
            norm = np.array([100, 1, 0.1, 0.1, 0.1])

            FoM = lambda p: GP.figureOfMerit(np.array([*dataC.fitCorrParams[:2], *(p*norm[2:])]))
            v0 = dataC.fitCorrParams[2:] / norm[2:]
            approx_grad = True
            bounds = (np.array([
                [0.1, 2000],
                [0.1, 3],
                [0.01, 3],
                [-3, 3],
                [-3, 3]
            ]).T / norm).T
            factr = 0.001 / np.finfo(float).eps
            pgtol = 1e-10
            epsilon = 0.05
            maxfun=75
            
            GPRutils.printParams(
                v0,
                header=True,
                FoMtype="xi +",
                file=GP.paramFile,
                printing=GP.printing)

            d_Wx_Wy = opt.fmin_l_bfgs_b(
                FoM,
                v0,
                approx_grad=approx_grad,
                bounds=bounds[2:, :],
                factr=factr,
                pgtol=pgtol,
                epsilon=epsilon,
                maxfun=maxfun
            )[0]
        
            FoM = lambda p: GP.figureOfMerit(np.array([*(p*norm[:2]), *(d_Wx_Wy*norm[2:])]))
            v0 = dataC.fitCorrParams[:2] / norm[:2]
            
            GPRutils.printParams(
                v0,
                header=True,
                FoMtype="xi +",
                file=GP.paramFile,
                printing=GP.printing)
            
            K_oS = opt.fmin_l_bfgs_b(
                FoM,
                v0,
                approx_grad=approx_grad,
                bounds=bounds[:2, :],
                factr=factr,
                pgtol=pgtol,
                epsilon=epsilon,
                maxfun=maxfun
            )[0]
            
            dataC.params = np.array([*(K_oS*norm[:2]), *(d_Wx_Wy*norm[2:])])

        else:
            GP.optimize()
    except np.linalg.LinAlgError:
        with open(GP.paramFile, mode="a+") as file:
            file.write("LinAlgError    ")
        GPRutils.printParams(
            GP.dC.params,
            file=GP.paramFile,
            printing=True)
        return
    
    t4 = time.time()*u.s

    dataC.JackKnife(GP, dataC.params)
    t5 = time.time()*u.s
    
    dataC.saveFITS(expFile)
    tf = time.time()*u.s
    
    total_time = tf - t0
    load_time = t1 - t0
    fC_time = t2 - t1
    fC_jackknife_time = t3 - t2
    opt_time = t4 - t3
    opt_jackknife_time = t5 - t4
    
    with open(GP.paramFile, mode="a+") as file:
        file.write(f"Total Time                          : {total_time.to(u.hr):.3f}\n")
        file.write(f"Load Time                           : {load_time.to(u.s):.3f}\n")
        file.write(f"Correlation Fitting Time            : {fC_time.to(u.min):.3f}\n")
        file.write(f"Correlating Fitting Jackknife Time  : {fC_jackknife_time.to(u.min):.3f}\n")
        file.write(f"Optimization Time                   : {opt_time.to(u.hr):.3f}\n")
        file.write(f"Optimization Jackknife Time         : {opt_jackknife_time.to(u.min):.3f}\n")
    
    
if __name__=='__main__':
    if not os.path.isdir(args.expFile):
        raise NameError(f"{args.expFile} not a valid directory.")
    
    if args.expNum is not None:
        exps = args.expNum
    else:
        if args.eris:
            exps = DESutils.erisBandDict[args.band]
        else:
            exps = DESutils.bandDict[args.band]
            
        if (args.start is not None) and (args.finish is not None):
            exps = exps[slice(args.start, args.finsh)]
        elif args.start is not None:
            exps = [exps[args.start]]

    for expNum in exps:
        main(expNum, args.expFile, args.RMS, args.curl, args.downselect, args.min, args.max, args.vSet, args.dualOpt, args.altOpt)
