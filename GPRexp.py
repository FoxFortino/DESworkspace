import os

import gpr

import numpy as np
import scipy.optimize

theta0 = np.array([200, 0.1, 0.1, np.pi/2, 1])
bounds = np.array([(0.001, 1000), (0.0001, 10), (0.0001, 10), (0, np.pi), (0.1, 10)])

def g1(x):
    sigma_x = x[1]
    sigma_y = x[2]
    return -sigma_x + 10 * sigma_y + 1

def g2(x):
    sigma_x = x[1]
    sigma_y = x[2]
    return -sigma_y + 10 * sigma_x + 1

cons = (
    {'type': 'ineq', 'fun': g1},
    {'type': 'ineq', 'fun': g2}
)

datafile = 'folio2'
exposures = np.arange(450, 460)
nSigma = 4
sample = None
test_frac = 0.50

for nExposure in exposures:
    uGP = gpr.GPR('dx', nLML_print=True)
    uGP.extract(datafile, nExposure, nSigma, sample=sample)
    uGP.split_data(test_frac)
    uresult = scipy.optimize.minimize(uGP.get_nLML, theta0, method='SLSQP', bounds=bounds, constraints=cons)
    uGP.fit(uresult.x)
    uGP.summary()

    vGP = gpr.GPR('dy', nLML_print=True)
    vGP.extract(datafile, nExposure, nSigma, sample=sample)
    vGP.split_data(test_frac)
    vresult = scipy.optimize.minimize(vGP.get_nLML, uresult.x, method='SLSQP', bounds=bounds, constraints=cons)
    vGP.fit(vresult.x)

    GP = uGP.combine(vGP)

    np.savez(
        os.path.join('../exposures', f"{nExposure}.npz"),
        fbar_s=GP.fbar_s,
        sigma=GP.sigma,
        nLML=GP.nLML,
        utheta=GP.utheta,
        vtheta=GP.vtheta,
        random_state=GP.random_state,
        nExposure=GP.nExposure,
        X=GP.X,
        Xtrain=GP.Xtrain,
        Xtest=GP.Xtest,
        Y=GP.Y,
        Ytrain=GP.Ytrain,
        Ytest=GP.Ytest,
        E=GP.E,
        Etrain=GP.Etrain,
        Etest=GP.Etest,
        t0=GP.t0,
        tf=GP.tf,
        RSS=GP.RSS,
        chisq=GP.chisq,
        red_chisq=GP.red_chisq)