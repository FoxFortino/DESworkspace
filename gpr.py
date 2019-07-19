import forAustin as fa

from time import time
import numpy as np
import tensorflow as tf
import astropy.units as u
import astropy.constants as c
import astropy.stats as stats
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

from IPython import embed

class calcGPR(object):
    def __init__(self, verbose=False, nLML_factor=-1, nLML_print=False, random_state=0, eps=1.49e-8):
        self.verbose = verbose
        self.nLML_factor = nLML_factor
        self.nLML_print = nLML_print
        self.random_state = random_state
        self.eps = eps
    
    def genGPR(self):
        self.uGP = GPR(coord='u', verbose=self.verbose, nLML_factor=self.nLML_factor, nLML_print=self.nLML_print, random_state=self.random_state, eps=self.eps)
        self.vGP = GPR(coord='v', verbose=self.verbose, nLML_factor=self.nLML_factor, nLML_print=self.nLML_print, random_state=self.random_state, eps=self.eps)
        
    def fit2d(self, u_theta0, v_theta0):
        self.uGP.fit(u_theta0)
        self.vGP.fit(v_theta0)
        self.uGP.absorb(self.vGP)
        return self.uGP
    
    def load(self, sample, nSigma, test_frac, datafile=None, nExposure=None, nSynth=None, theta_true=None):
        if nSynth:
            self.uGP.gen_synthetic_data(nSynth, theta_true)
            self.vGP.gen_synthetic_data(nSynth, theta_true)
        else:
            self.uGP.extract_exposure(datafile=datafile, nExposure=nExposure)
            self.vGP.extract_exposure(datafile=datafile, nExposure=nExposure)
            self.uGP.extract_data(sample=sample)
            self.vGP.extract_data(sample=sample)

        self.uGP.remove_outliers(nSigma)
        self.vGP.remove_outliers(nSigma)
        self.uGP.split_data(test_frac)
        self.vGP.split_data(test_frac)
        
    def solve2d(self, utheta0, vtheta0, bounds, verbose=False):
        self.uGP.verbose = False
        if verbose:
            print("Optimizing model for dimension u:")
        self.u_result = minimize(self.uGP.get_nLML, utheta0, method='SLSQP', bounds=bounds)
        if verbose:
            print('Model parameters for dimension u:')
            print(' '.join([f"{x:12.6f}" for x in self.u_result.x]))
        
        self.vGP.verbose = False
        if verbose:
            print("Optimizing model for dimension v:")
        self.v_result = minimize(self.vGP.get_nLML, vtheta0, method='SLSQP', bounds=bounds)
        if verbose:
            print('Model parameters for dimension v:')
            print(' '.join([f"{x:12.6f}" for x in self.v_result.x]))
        
    def batch(self, exposures, sample, nSigma, test_frac, datafile, utheta0, vtheta0, bounds):
        for nExposure in exposures:
            self.genGPR()
            self.load(sample, nSigma, test_frac, datafile=datafile, nExposure=nExposure)
            self.solve2d(utheta0, vtheta0, bounds, verbose=False)
            GP = self.fit2d(self.u_result.x, self.v_result.x)
            GP.error()
             

class GPR(object):
    def __init__(self, coord=None, verbose=False, nLML_factor=-1, nLML_print=False, random_state=0, eps=1.49e-8):
        """Gaussian Process Regression.
        
        Arguments:
            coord : String, 'dx' or 'dy' or 'dx dy' specifying which dimension to compute a model for.
            hoid : Bool, whether this code is being ran on hoid. If folio2 is also false then datafile argument for extract_exposure must be specified
            folio2 : Bool, whether this code is being ran on folio2. If hoid is also false then datafile argument for extract_exposure must be specified
            verbose : Bool, whether to print what the GPR is working on.
            nLML_factor : Integer, a factor to multiply the negative log marginal likelihood by. 
            nLML_print : Bool, whether to print model parameters at each step of evaluation of an optimizer.
            random_state : Integer, numpy random seed.
            eps : Float, small jitter for numerical stability of the cholesky decomposition. Often not needed because the white kernel will always be positive.
        """
        
        self.coord = coord
        assert self.coord == 'u' or self.coord == 'v' or self.coord == 'u v', "See docs for GPR object for info on coord."
        self.verbose = verbose
        self.nLML_factor = nLML_factor
        self.nLML_print = nLML_print
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.eps = eps
        
        self.synth = False
        
    def absorb(self, GP):
        assert self.coord != GP.coord
        assert self.coord == 'u' or self.coord == 'v'
        assert GP.coord == 'u' or GP.coord == 'v'
        
        if self.coord == 'u':
            self.fbar_s = np.vstack((self.u_fbar_s, GP.v_fbar_s)).T
            self.sigma = np.vstack((self.u_sigma, GP.v_sigma)).T
            self.nLML = self.u_nLML + GP.v_nLML
            self.v_theta = GP.v_theta

        if self.coord == 'v':
            self.fbar_s = np.vstack((GP.u_fbar_s, self.v_fbar_s)).T
            self.sigma = np.vstack((GP.u_sigma, self.v_sigma)).T
            self.nLML = GP.u_nLML + self.v_nLML
            self.u_theta = GP.u_theta
            
        self.coord = 'u v'

    def gen_synthetic_data(self, nSynth, theta, verbose=False):
        """Generates nSynth synthetic data points on a 4 deg^2 sky with kernel parameters given by thetaS."""
        if verbose or self.verbose: print("Generating synthetic data...")

        self.synth = True
        self.nSynth = nSynth
        self.theta = theta
        self.u_theta = theta
        
        self.X  = self.rng.uniform(low=-1, high=1, size=(self.nSynth, 2))
        self.E = np.abs(self.rng.normal(loc=0, scale=1, size=self.nSynth))
        
        self.u_C = self.make_EBF(self.u_theta, self.X, self.X)
        self.v_C = self.make_EBF(self.v_theta, self.X, self.X)
        
        self.u_W = self.make_W(self.u_theta, self.E)
        self.v_W = self.make_W(self.v_theta, self.E)
        
        dx = self.rng.multivariate_normal(np.zeros(self.X.shape[0]), self.u_C + self.u_W, size=1)
        dy = self.rng.multivariate_normal(np.zeros(self.X.shape[0]), self.v_C + self.v_W, size=1)
        self.Y = np.vstack((dx, dy)).T
        
        self.synth = False
        
    def gen_2Dcoordinate_arrays(self, X1, X2):
        """Helper function to generate coordinate arrays for vectorized evaluation of the elliptical kernel."""
        
        u1, u2 = X1[:, 0], X2[:, 0]
        v1, v2 = X1[:, 1], X2[:, 1]
        
        uu1, uu2 = np.meshgrid(u1, u2)
        vv1, vv2 = np.meshgrid(v1, v2)
        
        return uu1, uu2, vv1, vv2
    
    def extract_exposure(self, datafile=None, nExposure=500, polyOrder=3, verbose=False):
        """Extracts all data from a specified exposure (self.nExposure). Currently only supports extracting one exposure's worth of data."""
        if verbose or self.verbose: print("Extracting fits file...")

        if datafile == 'hoid':
            self.datafile = '/media/data/austinfortino/austinFull.fits'
        elif datafile == 'folio2':
            self.datafile = '/data4/paper/fox/DES/austinFull.fits'
        else:
            self.datafile = datafile

        self.nExposure = nExposure
        
        self.fits = pf.open(self.datafile)
        self.exposure = fa.getExposure(self.fits, self.nExposure, polyOrder=polyOrder)
        if verbose or self.verbose: self.fits.info(); print()
        
    def extract_data(self, sample=None, verbose=False):
        """Extracts u, v, dx, dy, and measErr columns from all data points where hasGaia is True."""
        if verbose or self.verbose: print(f"Extracting data from exposure {self.nExposure}...")
            
        self.sample = sample
        
        ind_hasGaia = np.where(self.exposure['hasGaia'])[0]
        u = np.take(self.exposure['u'], ind_hasGaia)
        v = np.take(self.exposure['v'], ind_hasGaia)
        dx = np.take(self.exposure['dx'], ind_hasGaia)
        dy = np.take(self.exposure['dy'], ind_hasGaia)
        E = np.take(self.exposure['measErr'], ind_hasGaia)
        
        # Extract only data in a certain region of the exposure as specified by self.sample.
        if self.sample is not None:
            assert self.sample.shape == (4,), f"Shape of sample is {self.sample.shape}, but must be (4,)."
            ind_u = np.logical_and(u >= self.sample[0], u <= self.sample[1])
            ind_v = np.logical_and(v >= self.sample[2], v <= self.sample[3])
            ind = np.where(np.logical_and(ind_u, ind_v))[0]
            u = np.take(u, ind, axis=0)
            v = np.take(v, ind, axis=0)
            dx = np.take(dx, ind, axis=0)
            dy = np.take(dy, ind, axis=0)
            E = np.take(E, ind)
            
        self.X = np.vstack((u, v)).T
        self.Y = np.vstack((dx, dy)).T
        self.E = E
        
    def remove_outliers(self, nSigma, plot=False, verbose=False):
        """Sigma clips data."""
        if verbose or self.verbose:
            print(f"Sigma clipping to {nSigma} standard deviations.")
        
        if plot and (verbose or self.verbose):
            self.plot_hist(fits_only=True)
        
        mask = stats.sigma_clip(self.Y, sigma=nSigma, axis=0).mask
        mask = ~np.logical_or(*mask.T)

        if verbose or self.verbose:
            print(f"{np.round(np.sum(mask.astype(int)) / mask.shape[0] * 100, 4)}% ({np.sum(mask.astype(int))}) data points are being kept.")
            
        self.X = self.X[mask, :]
        self.Y = self.Y[mask, :]
        self.E = self.E[mask]
        
        if plot and (verbose or self.verbose):
            self.plot_hist(fits_only=True)
        
    def split_data(self, test_size, verbose=False):
        """Separate data into training and testing sets with sklearn.model_selection.train_test_split."""
        if verbose or self.verbose: print("Splitting data into training and testing sets...")
        
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest, self.Etrain, self.Etest = \
            train_test_split(self.X, self.Y, self.E, test_size=test_size, random_state=self.random_state)
        
        self.nTrain = self.Xtrain.shape[0]
        self.nTest = self.Xtest.shape[0]
    
    def make_EBF(self, theta, X1, X2):
        """Vectorized solution for computing the elliptical kernel."""
        
        u1, u2 = X1[:, 0], X2[:, 0]
        v1, v2 = X1[:, 1], X2[:, 1]
        
        uu1, uu2 = np.meshgrid(u1, u2)
        vv1, vv2 = np.meshgrid(v1, v2)

        a = np.cos(theta[3])**2 / (2 * theta[1]**2) + np.sin(theta[3])**2 / (2 * theta[2]**2)
        b = - np.sin(2 * theta[3]) / (4 * theta[1]**2) + np.sin(2 * theta[3]) / (4 * theta[2]**2)
        c = np.sin(theta[3])**2 / (2 * theta[1]**2) + np.cos(theta[3])**2 / (2 * theta[2]**2)

        uu = a * (uu1 - uu2)**2
        vv = c * (vv1 - vv2)**2
        uv = 2 * b * (uu1 - uu2)*(vv1 - vv2)
        
        return theta[0] * np.exp(-(uu + vv + uv))
    
    def make_W(self, theta, E):
        """Make white noise covariance matrix. The last parameter in theta (for the first half for dx and the second half for dy) must be this value."""
        return theta[4]**2 * np.diag(E**2) + self.eps * np.eye(E.shape[0])
    
    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, theta):
        self.nParams = theta.shape[0]
            
        if self.coord == 'u v' or self.synth:
            assert self.nParams % 2 == 0, "Must provide an even number of parameters. First half for the dx kernel, second half for the dy kernel."
            self._theta = theta
            self.u_theta = theta[:self.nParams // 2]
            self.v_theta = theta[self.nParams // 2:]
            
        elif self.coord == 'u':
            self.u_theta = theta
            
        elif self.coord == 'v':
            self.v_theta = theta
        
    def fit(self, theta, verbose=False):
        """Solves the posterior predictive mean"""
        
        self.theta = theta
        
        if verbose or self.verbose: print("Generating covariance functions and solving for posterior...")
        t0 = time()
        
        if 'u' in self.coord:
            # Generate covariance functions for K, Kss, Ks, and W for u.
            self.u_K = self.make_EBF(self.u_theta, self.Xtrain, self.Xtrain)
            self.u_Kss = self.make_EBF(self.u_theta, self.Xtest, self.Xtest)
            self.u_Ks = self.make_EBF(self.u_theta, self.Xtest, self.Xtrain)
            self.u_W = self.make_W(self.u_theta, self.Etrain)

            # Solve for dx
            self.u_L = np.linalg.cholesky(self.u_K + self.u_W)
            self.u_alpha = np.linalg.solve(self.u_L.T, np.linalg.solve(self.u_L, self.Ytrain[:, 0]))
            self.u_fbar_s = np.dot(self.u_Ks.T, self.u_alpha)
            self.u_v = np.linalg.solve(self.u_L, self.u_Ks)
            self.u_V_s = self.u_Kss - np.dot(self.u_v.T, self.u_v)
            self.u_sigma = np.sqrt(np.abs(np.diag(self.u_V_s)))

            self.u_nLML = (-1/2) * np.dot(self.Ytrain[:, 0], self.u_alpha) - np.sum(np.log(np.diag(self.u_L))) - (self.nTest / 2) * np.log(2 * np.pi)
            
            self.fbar_s = self.u_fbar_s
            self.sigma = self.u_sigma
            self.nLML = self.u_nLML
        
        if 'v' in self.coord:
            # Generate covariance functions for K, Kss, Ks, and W for u.
            self.v_K = self.make_EBF(self.v_theta, self.Xtrain, self.Xtrain)
            self.v_Kss = self.make_EBF(self.v_theta, self.Xtest, self.Xtest)
            self.v_Ks = self.make_EBF(self.v_theta, self.Xtest, self.Xtrain)
            self.v_W = self.make_W(self.v_theta, self.Etrain)

            # Solve for dy
            self.v_L = np.linalg.cholesky(self.v_K + self.v_W)
            self.v_alpha = np.linalg.solve(self.v_L.T, np.linalg.solve(self.v_L, self.Ytrain[:, 1]))
            self.v_fbar_s = np.dot(self.v_Ks.T, self.v_alpha)
            self.v_v = np.linalg.solve(self.v_L, self.v_Ks)
            self.v_V_s = self.v_Kss - np.dot(self.v_v.T, self.v_v)
            self.v_sigma = np.sqrt(np.abs(np.diag(self.v_V_s)))

            self.v_nLML = (-1/2) * np.dot(self.Ytrain[:, 1], self.v_alpha) - np.sum(np.log(np.diag(self.v_L))) - (self.nTest / 2) * np.log(2 * np.pi)
            
            self.fbar_s = self.v_fbar_s
            self.sigma = self.v_sigma
            self.nLML = self.v_nLML
            
        if 'u' in self.coord and 'v' in self.coord:
            self.fbar_s = np.vstack((self.u_fbar_s, self.v_fbar_s)).T
            self.sigma = np.vstack((self.u_sigma, self.v_sigma)).T
            self.nLML = self.u_nLML + self.v_nLML
            self.nLML_arr = np.array([self.u_nLML, self.v_nLML])
        
        t1 = time()
        if verbose or self.verbose: print(f"Covariance functions generated, and posterior solved for, in {t1-t0} seconds.")
    
    def get_nLML(self, theta):        
        if self.nLML_print:
            print(' '.join([f"{x:12.6f}" for x in theta]))

        self.fit(theta)
        return self.nLML_factor * self.nLML
    
    def summary(self):
        if 'u' in self.coord:
            print('Model parameters for dimension u:')
            print(' '.join([f"{x:12.6f}" for x in self.u_theta]))
        if 'v' in self.coord:
            print('Model parameters for dimension v:')
            print(' '.join([f"{x:12.6f}" for x in self.v_theta]))
        self.summary_error()
        self.plot_uv()
        self.plot_residuals()
        self.plot_closeup()
        self.plot_quiver_GP()
        self.plot_quiver_fits()
        self.plot_resres()
        self.plot_hist()
    
    def summary_error(self):
        """Summarize the various error quantities for this GP."""
        self.error()
        if self.coord == 'u v':
            print(f"Negative log marginal likelihood: {self.nLML_arr}")
        else:
            print(f"Negative log marginal likelihood: {self.nLML}")
        print(f"RSS: {self.RSS})")
        print(f"Chisq: {self.chisq}")
        print(f"Reduced Chisq: {self.red_chisq}")

    def error(self):
        """Calculate the RSS, chi squared and reduced chi squared."""
        self.RSS = np.sum((self.Ytest - self.fbar_s)**2, axis=0)
        self.chisq = np.sum(((self.Ytest - self.fbar_s) / self.Etest[:, None])**2, axis=0)
        self.red_chisq = self.chisq / (self.nTrain + self.nTest - self.nParams)
              
    def plot_uv(self):
        plt.figure(figsize=(8, 8))
        plt.title(f"u, v Positions")
        plt.xlabel("u (deg)")
        plt.ylabel("v (deg)")
        plt.scatter(self.Xtrain[:, 0], self.Xtrain[:, 1], alpha=0.5, label="Training Positions")
        plt.scatter(self.Xtest[:, 0], self.Xtest[:, 1], alpha=0.5, label="Testing Positions")
        plt.legend()
        plt.show()
              
    def plot_residuals(self):
        plt.figure(figsize=(8, 8))
        plt.title(f"dx, dy Residuals")
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
        plt.figure(figsize=(8, 8))
        plt.title("Closeup of Posterior Predictive Mean of Residuals with 1$\sigma$ Error Bars")
        plt.xlabel("dx (mas)")
        plt.ylabel("dy (mas)")
        # plt.xlim((-100, 100))
        # plt.ylim((-100, 100))
        plt.errorbar(
            self.fbar_s[:, 0],
            self.fbar_s[:, 1],
            xerr=1*self.sigma[:, 0],
            yerr=1*self.sigma[:, 1],
            fmt='o',
            color='green',
            alpha=0.5)
        plt.show()
    
    def plot_quiver_GP(self):
        plt.figure(figsize=(8, 8))
        plt.title("Quiver Plot of Error for the Test Set with Errors given by the GP")
        plt.quiver(self.Xtest[:, 0], self.Xtest[:, 1], self.fbar_s[:, 0], self.fbar_s[:, 1])
        plt.show()
              
    def plot_quiver_fits(self):
        plt.figure(figsize=(8, 8))
        plt.title("Quiver Plot of Error for the Test Set with Errors given by the fits file")
        plt.quiver(self.Xtest[:, 0], self.Xtest[:, 1], self.Ytest[:, 0], self.Ytest[:, 1])
        plt.show()
    
    def plot_resres(self):
        plt.figure(figsize=(8, 8))
        plt.title(f"Test Set Residuals (given by fits file) Subtracted by Estimated Residuals (given by GP).")
        plt.xlabel("dx (mas)")
        plt.ylabel("dy (mas)")
#         plt.xlim((-100, 100))
#         plt.ylim((-100, 100))
        plt.scatter(self.Ytest[:, 0] - self.fbar_s[:, 0], self.Ytest[:, 1] - self.fbar_s[:, 1], alpha=0.5)
        plt.show()
    
    def plot_hist(self, bins=50, fits_only=False):
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