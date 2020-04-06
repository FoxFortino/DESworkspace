import os

import GPRutils
import vonkarmanFT as vk

import numpy as np
import astropy.units as u
import scipy.optimize as opt
from scipy.spatial.ckdtree import cKDTree


class vonKarman2KernelGPR(object):

    def __init__(self, dataContainer, FoM="A", printing=False, outDir="."):

        self.dC = dataContainer
        self.FoM = FoM
        self.printing = printing
        
        self.paramFile = os.path.join(outDir, "params.out")
        if os.path.exists(self.paramFile):
            os.remove(self.paramFile)

    def fit(self, params):
        
        self.dC.params = params

        self.ttt = vk.TurbulentLayer(
            variance=params[0],
            outerScale=params[1],
            diameter=params[2],
            wind=(params[3], params[4]))
        
        du, dv = GPRutils.getGrid(self.dC.Xtrain, self.dC.Xtrain)
        Cuv = self.ttt.getCuv(du, dv)
        Cuv[:, :, 0, 1] *= 0
        Cuv[:, :, 1, 0] *= 0
        n1, n2 = Cuv.shape[0], Cuv.shape[1]
        
        K = np.swapaxes(Cuv, 1, 2).reshape(2*n1, 2*n2)
        W = np.diag(GPRutils.flat(self.dC.Etrain)**2)
        # W = np.diag(GPRutils.flat(self.dC.Etrain)**2) * params[5] # Additional param here
        L = np.linalg.cholesky(K + W)
        
        self.alpha = np.linalg.solve(L, GPRutils.flat(self.dC.Ytrain))
        self.alpha = np.linalg.solve(L.T, self.alpha)
        
    def predict(self, X):

        du, dv = GPRutils.getGrid(X, self.dC.Xtrain)
        Cuv = self.ttt.getCuv(du, dv)
        Cuv[:, :, 0, 1] *= 0
        Cuv[:, :, 1, 0] *= 0
        n1, n2 = Cuv.shape[0], Cuv.shape[1]

        Ks = np.swapaxes(Cuv, 1, 2).reshape(2*n1, 2*n2)

        self.dC.fbar_s = GPRutils.unflat(np.dot(Ks.T, self.alpha))

    def figureOfMerit(self, params):

        self.fit(params)
        self.predict(self.dC.Xtest)

        if self.FoM == "A":
            # Figure of Merit Method A
            res = self.dC.Ytest - self.dC.fbar_s
            kdt = cKDTree(self.dC.Xtest)

            rMax = (0.2*u.deg).to(u.deg).value
            rMin = (5*u.mas).to(u.deg).value
            r = np.linspace(rMin, rMax, 100)

            xiplus = np.zeros(r.shape)
            for i, radius in enumerate(r):
                prs = kdt.query_pairs(r[i], output_type='ndarray')
                xiplus[i] = np.nanmean(res[prs][:, 0, :] * res[prs][:, 1, :])
            xiplus = np.nanmean(xiplus)

        if self.FoM == "B":
            # Figure of Merit Method B
            res = self.dC.Ytest - self.dC.fbar_s
            kdt = cKDTree(self.dC.Xtest)

            rMax = (0.02*u.deg).to(u.deg).value
            rMin = (5*u.mas).to(u.deg).value

            prs_set = kdt.query_pairs(rMax, output_type='set')
            prs_set -= kdt.query_pairs(rMin, output_type='set')
            prs = np.array(list(prs_set))

            xiplus = np.mean(np.sum(res[prs[:, 0]] * res[prs[:, 1]], axis=1))

        if self.printing:
            theta = {
                "xi_+": xiplus,
                'var': params[0],
                'oS': params[1],
                'd': params[2],
                'wind_x': params[3],
                'wind_y': params[4],
            }
            params = ' '.join(
                [f"{name:>6}: {x:<10.7f}" for name, x in theta.items()]
                )
            with open(self.paramFile, mode="a+") as file:
                file.write(params + "\n")
            print(params)

        return xiplus

    def fitCorr(self, v0=None, rmax=5*u.arcmin, nBins=50):

        # Calculate the 2D xiplpus that will be fitted
        x = self.dC.X[:, 0]*u.deg
        y = self.dC.X[:, 1]*u.deg
        dx = self.dC.Y[:, 0]*u.mas
        dy = self.dC.Y[:, 1]*u.mas
        xiplus = GPRutils.calcCorrelation2D(
            x, y, dx, dy, rmax=rmax, nBins=nBins)[0]
        xiplus = np.where(np.isnan(xiplus), 0, xiplus)

        # Generate the uniform grid that the von Karman xiplus will be 
        # analysed on
        dx = (rmax / (nBins / 2)).to(u.deg).value
        x = np.arange(-nBins / 2, nBins / 2) * dx
        xx, yy = np.meshgrid(x, x)

        def figureOfMerit(params):
            ttt = vk.TurbulentLayer(
                variance=params[0],
                outerScale=params[1],
                diameter=params[2],
                wind=(params[3], params[4]))
            
            Cuv = ttt.getCuv(xx, yy)
            xiplus_model = Cuv[:, :, 0, 0] + Cuv[:, :, 1, 1]
            xiplus_model = np.where(np.isnan(xiplus_model), 0, xiplus_model)
            
            RSS = np.sum((xiplus - xiplus_model)**2) / self.dC.nData

            if self.printing:
                theta = {
                    "RSS": RSS,
                    'var': params[0],
                    'oS': params[1],
                    'd': params[2],
                    'wind_x': params[3],
                    'wind_y': params[4]
                }
                params = ' '.join(
                    [f"{name:>6}: {x:<10.7f}" for name, x in theta.items()]
                    )
                with open(self.paramFile, mode="a+") as file:
                    file.write(params + "\n")
                print(params)
            
            return RSS

        if v0 is None:
            v0 = np.array([xiplus.max(), 1, 0.1, 0.05, 0.05])
            simplex0 = np.vstack([v0, np.vstack([v0]*v0.shape[0]) + np.diag(v0*0.15)])

        self.opt_result = opt.fmin(
            figureOfMerit,
            simplex0[0],
            xtol=5,
            ftol=1,
            maxfun=150,
            full_output=True,
            retall=True,
            initial_simplex=simplex0
        )

    def optimize(self, v0=None):
        
        if v0 is None:
            # v0 = self.opt_result[0]
            v0 = np.append(self.opt_result[0], np.array([1])) # For when using extra W param
        simplex0 = np.vstack([v0, np.vstack([v0]*v0.shape[0]) + np.diag(v0*0.15)])

        self.opt_result_GP = opt.fmin(
            self.figureOfMerit,
            simplex0[0],
            xtol=2,
            ftol=0.005,
            maxfun=150,
            full_output=True,
            retall=True,
            initial_simplex=simplex0
        )

def printParams(params):
    """This function prints the kernel parameters in a pleasing way."""
    
    theta = {
        'var': params[0],
        'oS': params[1],
        'd': params[2],
        'wind_x': params[3],
        'wind_y': params[4]
    }
    params = ' '.join([f"{name:>8}: {x:<11.8f}" for name, x in theta.items()])

    print(params)

