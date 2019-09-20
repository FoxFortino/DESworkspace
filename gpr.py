import forAustin as fa

import os

import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.stats as stats
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

class GPR(object):
    @property
    def res(self):
        return self._res
    
    @res.setter
    def res(self, res):
        if res == 'dx':
            self.col = 0
        elif res == 'dy':
            self.col = 1
        elif res == 'dxdy':
            pass
        else:
            raise ValueError("res must be either dx or dy (or dxdy when creating a combined GPR object).")
        self._res = res

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        if isinstance(theta, dict):    
            self.nParams = len(theta.keys())
            self._theta = theta
        elif isinstance(theta, (np.ndarray, list)):
            self._theta = {
                'var_s': theta[0],
                'sigma_x': theta[1],
                'sigma_y': theta[2],
                'phi': theta[3],
                'var_w': theta[4]
            }
            self.nParams = len(self._theta.keys())
        
    @property
    def sample(self):
        return self._sample
    
    @sample.setter
    def sample(self, sample):
        if sample is None:
            self._sample = sample
        else:
            assert isinstance(sample, dict)
            assert set(sample.keys()) == set(['u1', 'u2', 'v1', 'v2'])
            self._sample = sample
        
    def __init__(self, res, npz=None, nLML_print=False, nLML_factor=1, random_state=0):
        self.res = res
        self.nLML_print = nLML_print
        self.nLML_factor = nLML_factor
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        if not (npz is None):
            exposure = np.load(npz, allow_pickle=True)
            for arr in exposure:
                exec(f"self.{arr} = exposure[arr]")
            self.utheta = self.utheta.item()
            self.vtheta = self.vtheta.item()
        
    def extract(self, datafile, nExposure, nSigma, polyOrder=3, sample=None):
        if datafile == 'hoid':
            self.datafile = '/media/data/austinfortino/austinFull.fits'
        elif datafile == 'folio2':
            self.datafile = '/data4/paper/fox/DES/austinFull.fits'
        else:
            self.datafile = datafile

        self.nExposure = nExposure
        self.sample = sample

        self.fits = pf.open(self.datafile)
        self.exposure = fa.getExposure(self.fits, self.nExposure, polyOrder=polyOrder)
        
        ind_hasGaia = np.where(self.exposure['hasGaia'])[0]
        u = np.take(self.exposure['u'], ind_hasGaia)
        v = np.take(self.exposure['v'], ind_hasGaia)
        dx = np.take(self.exposure['dx'], ind_hasGaia)
        dy = np.take(self.exposure['dy'], ind_hasGaia)
        E = np.take(self.exposure['measErr'], ind_hasGaia)
        
        # Extract only data in a certain region of the exposure as specified by self.sample.
        if self.sample is not None:
            ind_u = np.logical_and(u >= self.sample['u1'], u <= self.sample['u2'])
            ind_v = np.logical_and(v >= self.sample['v1'], v <= self.sample['v2'])
            ind = np.where(np.logical_and(ind_u, ind_v))[0]
            u = np.take(u, ind, axis=0)
            v = np.take(v, ind, axis=0)
            dx = np.take(dx, ind, axis=0)
            dy = np.take(dy, ind, axis=0)
            E = np.take(E, ind)
            
        self.X = np.vstack((u, v)).T
        self.Y = np.vstack((dx, dy)).T
        self.E = E
        
        mask = stats.sigma_clip(self.Y, sigma=nSigma, axis=0).mask
        mask = ~np.logical_or(*mask.T)
            
        self.X = self.X[mask, :]
        self.Y = self.Y[mask, :]
        self.E = self.E[mask]
        
        self.i = 0
        
    def gen_synthetic_data(self, nSynth, theta):
        self.theta = theta
        self.returned = dict()
        
        self.X  = self.rng.uniform(low=-1, high=1, size=(nSynth, 2))
        self.E = np.abs(self.rng.normal(loc=0, scale=1, size=nSynth))
        
        self.C = self.make_EBF(self.theta, self.X, self.X)
        self.W = self.make_W(self.theta, self.E)
        
        self.Y = self.rng.multivariate_normal(np.zeros(nSynth), self.C + self.W)
        self.Y = np.vstack((self.Y, self.Y)).T
        
        self.i = 0
        
    def split_data(self, train_size):
        self.nData = self.X.shape[0]
        
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest, self.Etrain, self.Etest = \
            train_test_split(self.X, self.Y, self.E, train_size=train_size, random_state=self.random_state)
        
        self.nTrain = self.Xtrain.shape[0]
        self.nTest = self.Xtest.shape[0]
    
    def make_EBF(self, theta, X1, X2):
        u1, u2 = X1[:, 0], X2[:, 0]
        v1, v2 = X1[:, 1], X2[:, 1]
        
        uu1, uu2 = np.meshgrid(u1, u2)
        vv1, vv2 = np.meshgrid(v1, v2)

        a =   np.cos(theta['phi'])**2  / (2 * theta['sigma_x']**2) + np.sin(theta['phi'])**2  / (2 * theta['sigma_y']**2)
        b = - np.sin(theta['phi'] * 2) / (4 * theta['sigma_x']**2) + np.sin(theta['phi'] * 2) / (4 * theta['sigma_y']**2)
        c =   np.sin(theta['phi'])**2  / (2 * theta['sigma_x']**2) + np.cos(theta['phi'])**2  / (2 * theta['sigma_y']**2)

        uu = a * (uu1 - uu2)**2
        vv = c * (vv1 - vv2)**2
        uv = 2 * b * (uu1 - uu2)*(vv1 - vv2)
        
        return theta['var_s'] * np.exp(-(uu + vv + uv))
    
    def make_W(self, theta, E):
        return theta['var_w']**2 * np.diag(E**2)
        
    def fit(self, theta, optimizing=False):
        self.theta = theta

        self.K = self.make_EBF(self.theta, self.Xtrain, self.Xtrain)
        self.W = self.make_W(self.theta, self.Etrain)

        self.L = np.linalg.cholesky(self.K + self.W)
        
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Ytrain[:, self.col]))

        self.nLML = (-1/2) * np.dot(self.Ytrain[:, self.col], self.alpha) - np.sum(np.log(np.diag(self.L))) - (self.nTest / 2) * np.log(2 * np.pi)

        if not optimizing:
            self.Kss = self.make_EBF(self.theta, self.Xtest, self.Xtest)
            self.Ks = self.make_EBF(self.theta, self.Xtest, self.Xtrain)
            self.fbar_s = np.dot(self.Ks.T, self.alpha)
            self.v = np.linalg.solve(self.L, self.Ks)
            self.V_s = self.Kss - np.dot(self.v.T, self.v)
            self.sigma = np.sqrt(np.abs(np.diag(self.V_s)))

    def print_params(self, theta):
        print(' '.join([f"{name:>8}: {x:<10.5f}" for name, x in theta.items()]))
    
    def get_nLML(self, theta):        
        self.theta = theta
        
        if self.nLML_print:
            self.i += 1
            print(self.i, datetime.now(), ': ', end='')
            self.print_params(self.theta)
        
        self.fit(theta, optimizing=True)
        return -self.nLML / self.nLML_factor
    
    def summary(self):
        self.error()
        if self.res == 'dx' or self.res == 'dy':
            print(f"Model parameters for dimension {self.res}:")
            self.print_params(self.theta)
            
        if self.res == 'dxdy':
            print(f"Model parameters for dimension dx:")
            self.print_params(self.utheta)
            
            print(f"Model parameters for dimension dy:")
            self.print_params(self.vtheta)
        
        print(f"Negative Log Marginal Likelihood: {self.nLML}")
        print(f"RSS: {self.RSS}")
        print(f"Chisq: {self.chisq}")
        print(f"Reduced Chisq: {self.red_chisq}")
        print(f"Expected Turbulence (before modelling): {self.t0}")
        print(f"Expected Turbulence (after modelling): {self.tf}")
        
        
        self.plot_uv()
        self.plot_residuals()
        self.plot_closeup()
        self.plot_quiver()
        self.plot_resres()
        self.plot_hist()

    def error(self):
        if self.res == 'dx' or self.res == 'dy':
            self.RSS = np.sum((self.Ytest[:, self.col] - self.fbar_s)**2)
            self.chisq = np.sum(((self.Ytest[:, self.col] - self.fbar_s) / self.Etest)**2)
            self.red_chisq = self.chisq / (self.nTrain + self.nTest - self.nParams)
            self.t0 = (np.sum((self.Ytest[:, self.col] / self.Etest)**2) - self.nTest) / np.sum(self.Etest**-2)
            self.tf = (np.sum((self.fbar_s / self.Etest)**2) - self.nTest) / np.sum(self.Etest**-2)
            
        if self.res == 'dxdy':
            self.RSS = np.sum((self.Ytest - self.fbar_s)**2, axis=0)
            self.chisq = np.sum(((self.Ytest - self.fbar_s) / self.Etest[:, None])**2, axis=0)
            self.red_chisq = self.chisq / (self.nTrain + self.nTest - self.nParams)
            self.t0 = (np.sum((self.Ytest / self.Etest[:, None])**2, axis=0) - self.nTest) / np.sum(self.Etest**-2)
            self.tf = (np.sum((self.fbar_s / self.Etest[:, None])**2, axis=0) - self.nTest) / np.sum(self.Etest**-2)
        
    def combine(self, GP):
        assert self.nExposure == GP.nExposure
        assert self.res != GP.res
        assert self.res == 'dx' or self.res == 'dy'
        assert GP.res == 'dx' or GP.res == 'dy'
        assert (self.random_state == GP.random_state) and (not (self.random_state is None))
        assert np.all(self.X == GP.X) and np.all(self.Y == GP.Y) and np.all(self.E == GP.E)
        
        newGP = GPR('dxdy', random_state=self.random_state)
        
        if self.res == 'dx':
            newGP.fbar_s = np.vstack((self.fbar_s, GP.fbar_s)).T
            newGP.sigma = np.vstack((self.sigma, GP.sigma)).T
            newGP.nLML = np.array([self.nLML, GP.nLML])
            newGP.utheta = self.theta
            newGP.vtheta = GP.theta

        if self.res == 'dy':
            newGP.fbar_s = np.vstack((GP.fbar_s, self.fbar_s)).T
            newGP.sigma = np.vstack((GP.sigma, self.sigma)).T
            newGP.nLML = np.array([GP.nLML, self.nLML])
            newGP.utheta = GP.theta
            newGP.vtheta = self.theta
            
        newGP.Y = self.Y
        newGP.Ytrain = self.Ytrain
        newGP.Ytest = self.Ytest
                    
        newGP.X = self.X
        newGP.Xtrain = self.Xtrain
        newGP.Xtest = self.Xtest
        
        newGP.E = self.E
        newGP.Etrain = self.Etrain
        newGP.Etest = self.Etest
        
        newGP.nData = self.nData
        newGP.nTrain = self.nTrain
        newGP.nTest = self.nTest
        newGP.nParams = self.nParams
        newGP.nExposure = self.nExposure
            
        return newGP
    
    def savenpz(self, path):
        self.error()
        np.savez(
            os.path.join(path, f"{self.nExposure}.npz"),
            fbar_s=self.fbar_s,
            sigma=self.sigma,
            nLML=self.nLML,
            utheta=np.array(self.utheta),
            vtheta=np.array(self.vtheta),
            random_state=self.random_state,
            nExposure=self.nExposure,
            nTrain=self.nTrain,
            nTest=self.nTest,
            nData=self.nData,
            nParams=self.nParams,
            X=self.X,
            Xtrain=self.Xtrain,
            Xtest=self.Xtest,
            Y=self.Y,
            Ytrain=self.Ytrain,
            Ytest=self.Ytest,
            E=self.E,
            Etrain=self.Etrain,
            Etest=self.Etest,
            t0=self.t0,
            tf=self.tf,
            RSS=self.RSS,
            chisq=self.chisq,
            red_chisq=self.red_chisq
        )
        
    def savefits(self, path):
        u = pf.Column(name='u', array=self.Xtest[:, 0], format='D')
        v = pf.Column(name='v', array=self.Xtest[:, 1], format='D')
        dx = pf.Column(name='dx', array=self.Ytest[:, 0], format='D')
        dy = pf.Column(name='dy', array=self.Ytest[:, 1], format='D')
        E = pf.Column(name='measErr', array=self.Etest, format='D')
        fits = pf.BinTableHDU.from_columns([u, v, dx, dy, E])
        fits.writeto(os.path.join(path, str(self.nExposure) + '.fits'), overwrite=True)
        
        u = pf.Column(name='u', array=self.Xtest[:, 0], format='D')
        v = pf.Column(name='v', array=self.Xtest[:, 1], format='D')
        dx = pf.Column(name='dx', array=self.Ytest[:, 0] - self.fbar_s[:, 0], format='D')
        dy = pf.Column(name='dy', array=self.Ytest[:, 1] - self.fbar_s[:, 1], format='D')
        E = pf.Column(name='measErr', array=self.Etest, format='D')
        fits = pf.BinTableHDU.from_columns([u, v, dx, dy, E])
        fits.writeto(os.path.join(path, 'GP' + str(self.nExposure) + '.fits'), overwrite=True)
        
    def loadfits(self, GPpath, path):
        self.GPfits = pf.open(GPpath)
        self.GPexposure = self.GPfits[1].data
        self.fits = pf.open(path)
        self.exposure = self.fits[1].data
        
    def fits_summary(self):
        
        plt.figure(figsize=(8,8))
        dx, dy, u, v, wt = fa.residInPixels(self.exposure, binpix=1024)
        plt.show()
        plt.figure(figsize=(8,8))
        GPdx, GPdy, GPu, GPv, GPwt = fa.residInPixels(self.GPexposure, binpix=1024)
        plt.show()

        ###--------------------###
        
        plt.figure(figsize=(12,6))
        fa.ebPlot(dx, dy, u, v)
        plt.show()
        plt.figure(figsize=(12,6))
        fa.ebPlot(GPdx, GPdy, GPu, GPv)
        plt.show()
        
        ###--------------------###
        
        logr, xiplus, ximinus, xicross, junk = fa.vcorr(self.exposure)
        GPlogr, GPxiplus, GPximinus, GPxicross, GPjunk = fa.vcorr(self.GPexposure)
        
        f, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))
        plt.subplots_adjust(wspace=0)
        
        r = np.exp(logr)
        GPr = np.exp(GPlogr)
        
        axes[0].set_title("Real Correlation")
        axes[0].semilogx(r, xiplus, 'ro', label='xiplus')
        axes[0].semilogx(r, ximinus, 'bs', label='ximinus')
        axes[0].semilogx(r, xicross, 'gx', label='xicross')
        axes[0].grid()
        axes[0].set_xlabel('Separation (degrees)')
        axes[0].set_ylabel('xi (mas^2)')
        axes[0].legend(framealpha=0.3)
        
        axes[1].set_title("GP Correlation")
        axes[1].semilogx(GPr, GPxiplus, 'ro', label='xiplus (GP)')
        axes[1].semilogx(GPr, GPximinus, 'bs', label='ximinus(GP)')
        axes[1].semilogx(GPr, GPxicross, 'gx', label='xicross(GP)')
        axes[1].grid()
        axes[1].set_xlabel('Separation (degrees)')
        axes[1].set_ylabel('xi (mas^2)')
        axes[1].legend(framealpha=0.3)
        
        plt.show()
                
        ###--------------------###

        xiE, xiB = fa.xiEB(logr, xiplus, ximinus)
        GPxiE, GPxiB = fa.xiEB(GPlogr, GPxiplus, GPximinus)
        
        f, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))
        plt.subplots_adjust(wspace=0)
        
        axes[0].set_title("Real Correlation")
        axes[0].semilogx(r, xiE, 'ro', label='xiE')
        axes[0].semilogx(r, xiB, 'bs', label='xiB')
        axes[0].grid()
        axes[0].set_xlabel('Separation (degrees)')
        axes[0].set_ylabel('xi (mas^2)')
        axes[0].legend(framealpha=0.3)
        
        axes[1].set_title("GP Correlation")
        axes[1].semilogx(GPr, GPxiE, 'ro', label='xiE (GP)')
        axes[1].semilogx(GPr, GPxiB, 'bs', label='xiB (GP)')
        axes[1].grid()
        axes[1].set_xlabel('Separation (degrees)')
        axes[1].set_ylabel('xi (mas^2)')
        axes[1].legend(framealpha=0.3)
        
        plt.show()
        
        ###--------------------###

        xiplus, cts = fa.vcorr2d(self.exposure, rmax=0.3, bins=65)
        GPxiplus, GPcts = fa.vcorr2d(self.GPexposure, rmax=0.3, bins=65)
        
        f, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))
        plt.subplots_adjust(wspace=0)
        
        im = axes[0].imshow(xiplus, origin='lower', cmap='Spectral', interpolation='nearest', vmin=0, vmax=450)
        axes[0].set_title("Real vcorr2d")
        
        im = axes[1].imshow(GPxiplus, origin='lower', cmap='Spectral', interpolation='nearest', vmin=0, vmax=450)
        axes[1].set_title("GP vcorr2d")
        
#         cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
#         f.colorbar(im, cax=cbar_ax)
        
        plt.show()
        
        ###--------------------###

        print("Real: <dx*dx>, <dx*dy>, <dy*dy>, num pairs: ", fa.covAnnulus(self.exposure, rMax=0.02))
        print("GP: <dx*dx>, <dx*dy>, <dy*dy>, num pairs: ", fa.covAnnulus(self.GPexposure, rMax=0.02))
        
    def plot_uv(self):
        plt.figure(figsize=(8, 8))
        plt.title(f"u, v Positions")
        plt.xlabel("u (deg)")
        plt.ylabel("v (deg)")
        plt.scatter(self.Xtrain[:, 0], self.Xtrain[:, 1], alpha=0.5, marker='.', label="Training Positions")
        plt.scatter(self.Xtest[:, 0], self.Xtest[:, 1], alpha=0.5, marker='.', label="Testing Positions")
        plt.legend()
        plt.show()
              
    def plot_residuals(self):
        if self.res == 'dx' or self.res == 'dy':
            plt.figure(figsize=(8, 8))
            plt.title(f"{self.res} Residuals")
            plt.xlabel("nth Ordered Data Point")
            plt.ylabel(f"{self.res} (mas)")
            plt.ylim((-100, 100))
            ind = np.argsort(self.Ytest[:, self.col])
            plt.plot(self.Ytrain[:, self.col], linestyle="None", marker='o', alpha=0.5, label="Train Residuals")
            plt.plot(self.Ytest[ind, self.col], linestyle="None", marker='o', alpha=0.5, label="Testing Residuals")
            plt.errorbar(np.arange(self.nTest), self.fbar_s[ind], yerr=self.sigma, fmt='o', alpha=0.5, label='Posterior Predictive Mean of Residuals')
            plt.legend()
            plt.show()
        
        if self.res == 'dxdy':
            plt.figure(figsize=(8, 8))
            plt.title("dx, dy Residuals")
            plt.xlabel("dx (mas)")
            plt.ylabel("dy (mas)")
            plt.xlim((-100, 100))
            plt.ylim((-100, 100))
            plt.scatter(self.Ytrain[:, 0], self.Ytrain[:, 1], alpha=0.5, label="Training Residuals")
            plt.scatter(self.Ytest[:, 0], self.Ytest[:, 1], alpha=0.5, label="Testing Residuals")
            plt.scatter(self.fbar_s[:, 0], self.fbar_s[:, 1], alpha=0.5, label="Posterior Predictive Mean of Residuals")
            plt.legend()
            plt.show()
              
    def plot_closeup(self):
        if self.res == 'dx' or self.res == 'dy':
            return
        if self.res == 'dxdy':
            plt.figure(figsize=(8, 8))
            plt.title("Closeup of Posterior Predictive Mean of Residuals with 1$\sigma$ Error Bars")
            plt.xlabel("dx (mas)")
            plt.ylabel("dy (mas)")
            plt.errorbar(
                self.fbar_s[:, 0],
                self.fbar_s[:, 1],
                xerr=self.sigma[:, 0],
                yerr=self.sigma[:, 1],
                fmt='o',
                color='green',
                alpha=0.5)
            plt.show()
    
    def plot_quiver(self):
        if self.res == 'dx':
            f, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16, 8))
            plt.subplots_adjust(wspace=0)
            ax[0].set_title("Quiver Plot of dx residuals from the GP")
            ax[0].quiver(self.Xtest[:, 0], self.Xtest[:, 1], self.fbar_s, 0)

            ax[1].set_title("Quiver Plot of dx residuals from the FITS file")
            ax[1].quiver(self.Xtest[:, 0], self.Xtest[:, 1], self.Ytest[:, self.col], 0)
            plt.show()
        
        if self.res == 'dy':  
            f, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16, 8))
            plt.subplots_adjust(wspace=0)
            ax[0].set_title("Quiver Plot of dx residuals from the GP")
            ax[0].quiver(self.Xtest[:, 0], self.Xtest[:, 1], 0, self.fbar_s)

            ax[1].set_title("Quiver Plot of dx residuals from the FITS file")
            ax[1].quiver(self.Xtest[:, 0], self.Xtest[:, 1], 0, self.Ytest[:, self.col])
            plt.show()

        if self.res == 'dxdy':
            f, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16, 8))
            plt.subplots_adjust(wspace=0)
            ax[0].set_title("Quiver Plot of dx, dy residuals from the GP")
            ax[0].quiver(self.Xtest[:, 0], self.Xtest[:, 1], self.fbar_s[:, 0], self.fbar_s[:, 1])

            ax[1].set_title("Quiver Plot of dx, dy residuals from the FITS file")
            ax[1].quiver(self.Xtest[:, 0], self.Xtest[:, 1], self.Ytest[:, 0], self.Ytest[:, 1])
            plt.show()

    def plot_resres(self):
        if self.res == 'dx' or self.res == 'dy':
            plt.figure(figsize=(8, 8))
            plt.title(f"Test Set {self.res} Residuals (given by fits file) Subtracted by Estimated {self.res} Residuals (given by GP).")
            plt.xlabel("dx (mas)")
            plt.ylabel("dy (mas)")
            ind = np.argsort(self.Ytest[:, self.col])
            plt.plot(self.Ytest[ind, self.col] - self.fbar_s, linestyle="None", marker='o', alpha=0.5, label="Testing Residuals")
            plt.show()
            
        if self.res == 'dxdy':
            plt.figure(figsize=(8, 8))
            plt.title(f"Test Set {self.res} Residuals (given by fits file) Subtracted by Estimated {self.res} Residuals (given by GP).")
            plt.xlabel("dx (mas)")
            plt.ylabel("dy (mas)")
            plt.scatter(self.Ytest[:, 0] - self.fbar_s[:, 0], self.Ytest[:, 1] - self.fbar_s[:, 1], alpha=0.5)
            plt.show()
    
    def plot_hist(self, bins=50, fits_only=False):
        if self.res == 'dx' or self.res == 'dy':            
            plt.figure(figsize=(8, 8))
            plt.title("Histogram of Residuals (dx, dy)")
            plt.xlabel("Value (mas)")
            plt.ylabel("Probability")
            plt.hist(self.Y[:, self.col], bins=bins, histtype='step', density=True, label=f"{self.res}")
            if not fits_only:
                plt.hist(self.fbar_s, bins=bins, histtype='step', density=True, label=f"{self.res} from GP")
            plt.legend()
            plt.show()
            
        if self.res == 'dxdy':
            plt.figure(figsize=(8, 8))
            plt.title("Histogram of Residuals (dx, dy)")
            plt.xlabel("Value (mas)")
            plt.ylabel("Probability")
            plt.hist(self.Y[:, 0], bins=bins, histtype='step', density=True, label="dx")
            plt.hist(self.Y[:, 1], bins=bins, histtype='step', density=True, label="dy")
            if not fits_only:
                plt.hist(self.fbar_s[:, 0], bins=bins, histtype='step', density=True, label="dx from GP")
                plt.hist(self.fbar_s[:, 1], bins=bins, histtype='step', density=True, label="dy from GP")
            plt.legend()
            plt.show()