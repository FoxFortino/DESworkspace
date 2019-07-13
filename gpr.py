import forAustin as fa

import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class GPR(object):
    
    def __init__(self, verbose=False, random_state=None, tensor=None):
        """Gaussian Process Regression."""
        self.verbose = verbose
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        self.tensor = tensor

    def gen_synthetic_data(self, nSynth, thetaS, verbose=False):
        """Generates nSynth synthetic data points on a 4 deg^2 sky with kernel parameters given by thetaS."""
        if verbose or self.verbose: print("Generating synthetic data...")
            
        self.nSynth = nSynth
        self.thetaS = thetaS
        
        self.X  = self.rng.uniform(low=-1, high=1, size=(self.nSynth, 2))
        
        self.C = self.EBF(self.thetaS, self.X, self.X)
        self.Y = self.rng.multivariate_normal(np.zeros(self.X.shape[0]), self.C, size=2).T
        self.E = np.abs(self.rng.normal(loc=0, scale=1, size=self.nSynth))
        
    def gen_2Dcoordinate_arrays(self, X1, X2):
        """Helper function to generate coordinate arrays for vectorized evaluation of the elliptical kernel."""
        
        u1, u2 = X1[:, 0], X2[:, 0]
        v1, v2 = X1[:, 1], X2[:, 1]
        
        uu1, uu2 = np.meshgrid(u1, u2)
        vv1, vv2 = np.meshgrid(v1, v2)
        
        return uu1, uu2, vv1, vv2
    
    def extract_exposure(self, datafile, nExposure, polyOrder=3, sample=None, verbose=False):
        """Extracts all data from a specified exposure (self.nExposure). Currently only supports extracting one exposure's worth of data."""
        if verbose or self.verbose: print(f"Extracting exposure data from fits file from exposure {nExposure}...")
        
        self.datafile = datafile
        self.nExposure = nExposure
        self.sample = sample
        
        self.fits = pf.open(self.datafile)
        self.exposure = fa.getExposure(self.fits, self.nExposure, polyOrder=polyOrder)
        if verbose or self.verbose: self.fits.info(); print()
        
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
        
    def remove_outliers(self, sigma, verbose=False):
        """Removes data points that have residuals greater than sigma standard deviations from the mean."""
        if verbose or self.verbose: print(f"Removing data points that have residuals greater than {sigma} standard deviations from the mean...")
        
        self.sigma = sigma
        
        percentile = lambda x, sigma: np.mean(x) + sigma * np.array([-np.std(x), np.std(x)])
        perc_dx = percentile(self.Y[:, 0], self.sigma)
        perc_dy = percentile(self.Y[:, 1], self.sigma)

        ind_dx = np.logical_and(self.Y[:, 0] > perc_dx[0], self.Y[:, 0] < perc_dx[1])
        ind_dy = np.logical_and(self.Y[:, 0] > perc_dy[0], self.Y[:, 1] < perc_dy[1])
        ind = np.where(np.logical_and(ind_dx, ind_dy))[0]
        if verbose or self.verbose: print(f"{np.round(ind.shape[0] / self.Y.shape[0] * 100, 4)}% of data points are being kept.")

        u = np.take(self.X[:, 0], ind, axis=0)
        v = np.take(self.X[:, 1], ind, axis=0)
        dx = np.take(self.Y[:, 0], ind, axis=0)
        dy = np.take(self.Y[:, 1], ind, axis=0)
        E = np.take(self.E, ind, axis=0)
        
        self.X = np.vstack((u, v)).T
        self.Y = np.vstack((dx, dy)).T
        self.E = E
        
    def split_data(self, test_size, verbose=False):
        """Separate data into training and testing sets with sklearn.model_selection.train_test_split."""
        if verbose or self.verbose: print("Splitting data into training and testing sets...")
        
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest, self.Etrain, self.Etest = \
            train_test_split(self.X, self.Y, self.E, test_size=test_size, random_state=self.random_state)
        
        self.nTrain = self.Xtrain.shape[0]
        self.nTest = self.Xtest.shape[0]
    
    def white_cov(self, eps=1.49e-8, verbose=False):
        """Generate white noise covariance matrix."""
        if verbose or self.verbose: print("Generating white noise covariance function...")
            
        self.eps = eps
        
        self.W = np.diag(self.Etrain**2) + self.eps * np.eye(self.nTrain)
        self.Wss = np.diag(self.Etest**2) + self.eps * np.eye(self.nTest)
    
    def EBF(self, theta, X1, X2):
        """Vectorized solution for computing the elliptical kernel."""

        var_s, sigma_x, sigma_y, phi = theta
        
        u1, u2 = X1[:, 0], X2[:, 0]
        v1, v2 = X1[:, 1], X2[:, 1]
        
        uu1, uu2 = np.meshgrid(u1, u2)
        vv1, vv2 = np.meshgrid(v1, v2)

        a = np.cos(phi)**2 / (2 * sigma_x**2) + np.sin(phi)**2 / (2 * sigma_y**2)
        b = - np.sin(2 * phi) / (4 * sigma_x**2) + np.sin(2 * phi) / (4 * sigma_y**2)
        c = np.sin(phi)**2 / (2 * sigma_x**2) + np.cos(phi)**2 / (2 * sigma_y**2)

        uu = a * (uu1 - uu2)**2
        vv = c * (vv1 - vv2)**2
        uv = 2 * b * (uu1 - uu2)*(vv1 - vv2)
        
        return var_s * np.exp(-(uu + vv + uv))
        
    def fit(self, theta, verbose=False):
        """Solves the posterior predictive mean"""
        if verbose or self.verbose: print("Generating elliptical covariance function...")
        self.K = self.EBF(theta, self.Xtrain, self.Xtrain)
        self.Kss = self.EBF(theta, self.Xtest, self.Xtest)
        self.Ks = self.EBF(theta, self.Xtest, self.Xtrain)
        
        if verbose or self.verbose: print("Solving for posterior...")
        self.L = np.linalg.cholesky(self.K + self.W)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Ytrain))
        self.fbar_s = np.dot(self.Ks.T, self.alpha)

        self.v = np.linalg.solve(self.L, self.Ks)
        self.V_s = self.Kss - np.dot(self.v.T, self.v)
        self.sigma = np.sqrt(np.abs(np.diag(self.V_s)))
        
        self.nLML = np.sum(np.diag((-1/2) * np.dot(self.Ytrain.T, self.alpha) - np.sum(np.log(np.diag(self.L))) - (self.nTest / 2) * np.log(2 * np.pi)))
    
    def get_nLML(self, positive=False):
        """Returns negative log marginal likelihood. Use positive=True for when using a minimizing function."""
        if positive:
            return -self.nLML
        else:
            return self.nLML
    
    def summary(self, sigma=1):
        print(f"Current Log Marginal Likelihood: {self.get_nLML()}")
        self.check_error(sigma=sigma)
        self.chisq()
        self.plot_uv()
        self.plot_residuals()
        self.plot_closeup()
        self.plot_quiver_GP()
        self.plot_quiver_fits()
        self.plot_resres()
        self.plot_hist()
    
    def check_error(self, sigma):
        """Check what percentage of test points are within sigma standard deviations of the posterior predictive mean."""
        within_x = np.sum((np.abs(self.Ytest[:, 0] - self.fbar_s[:, 0]) < sigma*self.sigma).astype(int))
        within_y = np.sum((np.abs(self.Ytest[:, 0] - self.fbar_s[:, 0]) < sigma*self.sigma).astype(int))
        print(f"Fraction of test points within {sigma} standard deviation(s) of posterior predictive mean:")
        print(f"dx: {within_x / self.nTest}; dy: {within_y / self.nTest}")
        
    def chisq(self):
        chisq = np.sum((self.Ytest - self.fbar_s)**2, axis=0)
        print(f"Chisq dx: {chisq[0]}")
        print(f"Chisq d1: {chisq[1]}")
              
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
            xerr=1*self.sigma,
            yerr=1*self.sigma,
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
    
    def plot_hist(self):
        plt.figure(figsize=(8, 8))
        plt.title("Histogram of Residuals (dx, dy)")
        plt.xlabel("Value (mas)")
        plt.ylabel("Probability")
        plt.hist(self.Y[:, 0], bins=50, histtype='step', density=True, label="dx")
        plt.hist(self.Y[:, 1], bins=50, histtype='step', density=True, label="dy")
        plt.hist(self.fbar_s[:, 0], bins=50, histtype='step', density=True, label="dx from GP")
        plt.hist(self.fbar_s[:, 1], bins=50, histtype='step', density=True, label="dy from GP")
        plt.legend()
        plt.show()