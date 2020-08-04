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


parser = argparse.ArgumentParser(
    description="Run a GPR model with some arguments.",
    epilog="Willow Fox Fortino is the author of this script and is the "
           "primary author of most of the scripts used here. For help running "
           "this, contact her: <fortino_at_sas.upenn.edu>.")

parser.add_argument(
    "expFile",
    nargs="?",
    type=str,
    help="The existing directory you want to store the output of GPR to.")
parser.add_argument(
    "-r", "--RMS",
    action="store_true",
    help="If True, replace DES supplied variances (via SExtractor) with "
         "empirically calculated RMS^2 values.")
parser.add_argument(
    "-c", "--curl",
    action="store_true",
    help="If True, take advantage of the curl-free nature of the residual "
         "field and use the off-diagonal terms of the von Kármán covariance "
         "function.")
parser.add_argument(
    "-d", "--dualOpt",
    action="store_true",
    help="If True, use the dual optimzation scheme where two rounds of "
         "Nelder-Mead optimzation is used instead of one.")
parser.add_argument(
    "-a", "--altOpt",
    action="store_true",
    help="If True, use the dual optimzation scheme were two rounds of"
         "L_BFGS_B optimization is used instead of one or two rounds of "
         "Nelder-Mead optimization.")

parser.add_argument(
    "-e", "--eris",
    action="store_true",
    help="If True, -b will use a list of exposures that contain the TNO Eris. "
         "If False, -b will use a list of exposures from DES Zone 134. See -s "
         "for more details.")
parser.add_argument(
    "-b", "--band",
    nargs="?",
    type=str,
    choices=["Y", "g", "r", "i", "z"],
    help="The DES passband you want to analyze an exposure from. "
         "Must be used with -s or -s and -f")
parser.add_argument(
    "-z", "--zone",
    nargs="?",
    type=int,
    help="What DES zone to get the bandDict or list of complete exposure for.")
parser.add_argument(
    "-s", "--start",
    nargs="?",
    type=int,
    help="The index of the exposure in the list of exposures specified by "
         "-b that will be analyzed. If -f is also used, then this represents "
         "the (inclusive) start of a range of exposures to run. If -e is used "
         "then the list of exposures specified by -b will be exposures that "
         "have the TNO Eris in them, otherwise the list of exposures will be "
         "from DES Zone 134.")
parser.add_argument(
    "-f", "--finish",
    nargs="?",
    type=int,
    help="The (exclusive) end of a range of exposures to run. Must be used "
         "with -s. See -s for more details.")

parser.add_argument(
    "-n", "--expNum",
    nargs="+",
    type=int,
    help="A space-separated list of exposure numbers to run.")

parser.add_argument(
    "-v", "--vSet",
    nargs="?",
    default="Subset A",
    type=str,
    choices=["Subset A", "Subset B", "Subset C", "Subset D", "Subset E"],
    help="Choose which of the five subsets will be used as the primary "
         "validation set.")
parser.add_argument(
    "--min",
    nargs="?",
    default=-np.inf,
    type=float,
    help="Remove all DES sources with variance less than this value in mas^2.")
parser.add_argument(
    "--max",
    nargs="?",
    default=np.inf,
    type=float,
    help="Remove all DES sources with variance greater than this value in "
         "mas^2.")
parser.add_argument(
    "--downselect",
    nargs="?",
    default=1,
    type=float,
    help="The fraction (between 0 and 1) of total data to include in "
         "alaysis.")
args = parser.parse_args()


def main(
    expNum: int,
    expFile: str,
    useRMS: bool,
    curl: bool,
    downselect: float,
    minDESErr: float,
    maxDESErr: float,
    vSet: str,
    dualOpt: bool,
    altOpt: bool
        ) -> None:
    """
    Run the GPR algorithm on one exposure.

    Arguments
    ---------
    expNum : int
        The exposure number to analyse.
    expFile : str
        The existing directory to save the solutions to.
    useRMS : bool
        If True, replace DES variances from SExtractor with empirical RMS^2
        values.
    curl : bool
        If True, use the off-diagonal elements of the von Kármán covariance
        matrix.
    downselect : float
        A number between 0 and 1. Use only this fraction of data instead of
        the entire dataset. Usually useful for running experiments that will
        finish quickly.
    minDESErr : float
        All DES sources with DES variance (from SExtractor) less than this
        value, in mas, will be removed.
    maxDESErr : float
        All DES sources with DES variance (from SExtractor) greater than this
        value, in mas, will be removed.
    vSet : str
        Which subset of the data to use as the primary validation set.
    dualOpt : bool
        If True, use the dual optimzation scheme where two rounds of
        Nelder-Mead optimzation is used instead of one.
    altOpt : bool
        If True, use the dual optimzation scheme were two rounds of L_BFGS_B
        optimization is used instead of one or two rounds of Nelder-Mead
        optimization.
    """
    t0 = time.time()*u.s
    dataC = GPRutils.dataContainer()
    dataC.load(
        expNum,
        downselect=downselect,
        minDESErr=minDESErr*u.mas**2,
        maxDESErr=maxDESErr*u.mas**2,
        useRMS=useRMS,
        vSet=vSet)

    t1 = time.time()*u.s
    GP = vK2KGPR.vonKarman2KernelGPR(
        dataC,
        printing=True,
        outDir=expFile,
        curl=curl)
    GP.fitCorr()

    t2 = time.time()*u.s
    dataC.JackKnife(GP, dataC.fitCorrParams, fC=True)

    t3 = time.time()*u.s
    try:
        if dualOpt:
            # Wrapper function for keeping K and OS the same while leaving d,
            # Wx, and Wy available as inputs.
            # K and OS will be the fitCorr params.
            def FoM(p):
                params = np.array([*dataC.fitCorrParams[:2], *p])
                val = GP.figureOfMerit(params)
                return val

            # Initial guess will be d, Wx, and Wy from fitCorr.
            v0 = dataC.fitCorrParams[2:]

            # Call the GP optimize method with the different figure of merit
            # function
            GP.optimize(v0=v0, func=FoM, ftol=0.05, maxfun=50)
            d_Wx_Wy = GP.opt_result_GP[0].copy()

            # Wrapper function for keeping d, Wx, and Wy the same while
            # leaving K and OS available as inputs.
            # d, Wx, and Wy will be the values from the first stage of
            # optimization.
            def FoM(p):
                params = np.array([*p, *d_Wx_Wy])
                val = GP.figureOfMerit(params)
                return val

            # Initial guess will be K and OS from fitCorr.
            v0 = dataC.fitCorrParams[:2]

            # Call the GP optimize method with the different figure of merit
            # function
            GP.optimize(v0=v0, func=FoM, ftol=0.05, maxfun=50)
            K_oS = GP.opt_result_GP[0].copy()

            dataC.params = np.array([*K_oS, *d_Wx_Wy])

        elif altOpt:

            # Take the absolute value of the first three parameters because
            # they shouold be positive. It doesn't matter if they are negative
            # for the purposes of calculating the kernel, but to the L_BFGS_B
            # optimizer it does matter because we are imposing bounds on it.
            # Of course the last two, wind_x and wind_y, should be allowed to
            # be negative.
            dataC.fitCorrParams = np.array([
                np.abs(dataC.fitCorrParams[0]),
                np.abs(dataC.fitCorrParams[1]),
                np.abs(dataC.fitCorrParams[2]),
                dataC.fitCorrParams[3],
                dataC.fitCorrParams[4]
            ])

            # This array is the approximate scale of each parameters. Take an
            # array of parameters and dividing it by this roughly normalizes
            # each parameter to the same scale. This is necessary because
            # L_BFGS_B takes the eps argument which is a single value, unlike
            # other optimzers which allow for an array of values, providing an
            # eps for each parameters.
            norm = np.array([100, 1, 0.1, 0.1, 0.1])

            # Wrapper function for keeping K and OS the same while leaving d,
            # Wx, and Wy available as inputs.
            # K and OS will be the fitCorr params.
            # Note that p has to be multiplied by norm[2:] because we need to
            # un-normalize the parameters.
            def FoM(p):
                params = np.array([*dataC.fitCorrParams[:2], *(p*norm[2:])])
                val = GP.figureOfMerit(params)
                return val

            # Initial guess for the optimizer. Must divide by norm[2:] to
            # normalize it.
            v0 = dataC.fitCorrParams[2:] / norm[2:]

            # Approximate the gradient numerically. Currently we have no
            # infrastructure for analytically calculating the gradient.
            approx_grad = True

            # Bounds generally chosen based on physical intution for the
            # system. The upper bound of 3 deg for the last three parameters is
            # because not only have we rarely seen values greater than this,
            # but also 3 deg is the approxmate diameter of the DECam focal
            # plane. While values larger than this are possible, it is less
            # likely and they matter less in terms of figure of merit
            # minimzation.
            # Also we divide by norm to normalize the bounds.
            bounds = (np.array([
                [0.1, 2000],  # K variance [mas^2]
                [0.1, 10],  # Outer Scale [deg]
                [0.01, 3],  # Diameter [deg]
                [-3, 3],  # Wind_x [deg]
                [-3, 3]  # Wind_y [deg]
            ]).T / norm).T

            # According to the documentation for scipy.optimize.fmin_l_bfgs_b,
            # ftol = factr * np.finfo(float).eps, where np.finfo(float).eps is
            # the machine limit for the float. ftol is the tolerance between
            # values of the figure of merit such that the optimizer will end.
            # This has a much easier to interpret meaning so that is why we
            # choose ftol and calculate factr, and not choose a factr value
            # from the beginning.
            factr = 0.001 / np.finfo(float).eps

            # Tests show that only one of the two tolerance paramteres, factr
            # and pgtol, have to be satisfied for the optimization to end.
            # Therefore, because I don't have much intuition for what pgtol
            # should be, we set it to be incredibly small so that it doesn't
            # matter
            pgtol = 1e-10

            # epsilon is the standard step size for ALL PARAMTERS. Some
            # optimizers all for an array of epsilon values but not L_BFGS_B.
            # This is why we have to normalize our parameters.
            # This value was chosen based on intuition and testing.
            epsilon = 0.05

            # A relatively arbitrary upper limit on the number of function
            # evaluations in order to not let the optimzation take too long.
            maxfun = 75

            # Print header information to the outfile.
            GPRutils.printParams(
                v0,
                header=True,
                FoMtype="xi +",
                file=GP.paramFile,
                printing=GP.printing)

            # Start the first round of optimization.
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

            # Wrapper function for keeping d, Wx, and Wy the same while
            # leaving K and OS available as inputs.
            # d, Wx, and Wy will be the values from the first stage of
            # optimization.
            # Note that p and d_Wx_Wy has to be multiplied by norm because we
            # need to un-normalize the parameters.
            def FoM(p):
                params = np.array([*(p*norm[:2]), *(d_Wx_Wy*norm[2:])])
                val = GP.figureOfMerit(params)
                return val

            # Initial guess for the optimizer. Must divide by norm[2:] to
            # normalize it.
            v0 = dataC.fitCorrParams[:2] / norm[:2]

            # Print header information to the outfile.
            GPRutils.printParams(
                v0,
                header=True,
                FoMtype="xi +",
                file=GP.paramFile,
                printing=GP.printing)

            # Start the second round of optimization.
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

            # Pack all of the parameters from the two rounds of optimization
            # into one array, making sure to normalize each set of parameters.
            dataC.params = np.array([*(K_oS*norm[:2]), *(d_Wx_Wy*norm[2:])])

        else:
            GP.optimize()

    # Sometimes, with weird parameters, creating the covariance matrix can
    # throw a LinAlgError. We want to save this and note which parameters it
    # was that caused this error so we can correct for that in the future.
    except np.linalg.LinAlgError:

        # Write the parameters to the outfile and end the algorithm here.
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
        file.write("Total Time                          : "
                   f"{total_time.to(u.hr):.3f}\n")
        file.write("Load Time                           : "
                   f"{load_time.to(u.s):.3f}\n")
        file.write("Correlation Fitting Time            : "
                   f"{fC_time.to(u.min):.3f}\n")
        file.write("Correlating Fitting Jackknife Time  : "
                   f"{fC_jackknife_time.to(u.min):.3f}\n")
        file.write("Optimization Time                   : "
                   f"{opt_time.to(u.hr):.3f}\n")
        file.write("Optimization Jackknife Time         : "
                   f"{opt_jackknife_time.to(u.min):.3f}\n")


if __name__ == '__main__':
    if not os.path.isdir(args.expFile):
        raise NameError(f"{args.expFile} not a valid directory.")

    # Check if the user wants to use the expNum argument an input expoosure
    # numbers directly.
    if args.expNum is not None:
        exps = args.expNum

    else:
        # If args.eris is False, then take exposures from the eris bandDict.
        if args.eris:
            exps = DESutils.erisBandDict[args.band]

        # If args.eris is True, then take exposures from the DES Zone 134
        # bandDict.
        else:
            # If args.zone isn't specified, assume zone134, for which we have
            # a hardcoded bandDict for.
            if args.zone is None:
                exps = DESutils.bandDict[args.band]
            else:
                zoneDir = os.path.join("/data3/garyb/tno/y6", "zone"+str(args.zone))
                ce, bd = DESutils.createBandDict(zoneDir)
                exps = bd[args.band]

        # If args.start and args.finish are used, sort out the slice of the
        # exposure list here.
        if (args.start is not None) and (args.finish is not None):
            exps = exps[slice(args.start, args.finsh)]

        # If only args.start is used, then take that exposure only. Need to
        # wrap this in brackets to make it a list (an iterable) for the for
        # loop below.
        elif args.start is not None:
            exps = [exps[args.start]]

    # Loop through all designated exposures.
    for expNum in exps:
        main(
            expNum, args.expFile,
            args.RMS, args.curl,
            args.downselect,
            args.min, args.max,
            args.vSet,
            args.dualOpt, args.altOpt)
