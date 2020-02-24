import gbutil
import treecorr as tc

import os
import shutil
from datetime import datetime

import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.stats as stats
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.model_selection import train_test_split

from IPython import embed

class CurlFreeGPR(object):
    """Curl Free Gaussian Process Regressor class."""
    
    def __init__(self, outdir=None, random_state=0, printing=True):
        """
        Constructor for the Curl Free GPR class.
        
        Parameters:
        random_state (int): Random state variable for the various numpy and scipy random operations.
        """
        
        if outdir:
            self.outdir = outdir
            try:
                os.mkdir(self.outdir)
            except FileExistsError:
                shutil.rmtree(self.outdir)
                os.mkdir(self.outdir)
        else:
            self.outdir = ''

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        self.printing = printing

        self.Xunit = u.deg
        self.Yunit = u.mas
        self.Eunit = u.mas
        
        self.i = None
    
    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = self.fix_params(theta)
        self.nParams = len(self._theta.keys())
        
    def load_synthetic_data(self, file):
        """
        Loads synthetic data from an npz file.
        
        Parameters:
            file (str): path to npz file to be loaded.
        """
        
        file = np.load(file, allow_pickle=True)
        self.theta = file['theta'].tolist()
        self.X = file['X']
        self.Y = file['Y']
        self.E = file['E']
        
        print(f"You just loaded data with length {self.X.shape[0]} and with kernel parameters:")
        self.print_params(self.theta)
        
    def save_synthetic_data(self, file, theta, X, Y, E):
        """Saves the releveant information generated from synthetic data in order to reload it in the future and avoid the computation cost of drawing from multivariate normal distribution of potentially thousands of dimensions.
        
        Parameters:
            file (str): The file name of the npz file to be created.
            theta (dict): The kernel parameter values. Must be a dict.
            X (ndarray): Array of astrometric positions.
            Y (ndarray): Array of astrometric residuals.
            E (ndarray): Array of astrometric measurement error.
        """
        
        theta = self.fix_params(theta)
        
        assert isinstance(theta, dict), f"Type of theta is {type(theta)} but it must be of type np.ndarray."
        np.savez(file, theta=theta, X=X, Y=Y, E=E)

    def gen_synthetic_data(self, nSynth, theta, save=False):
        """
        Generates syntheta astrometric positions (u, v) and residuals (dx, dy) based on the curl-free kernel.
        
        Parameters:
            nSynth (int): Number of data points.
            theta (dict): Dictionary of the kernel parameters.
            save (bool/str): If save is a string then the relevant information generated in this method will be saved in an npz file called save.
        """
        
        theta = self.fix_params(theta)
        
        X = self.rng.uniform(low=-1, high=1, size=(nSynth, 2))
        self.X = X
        
        E = np.abs(self.rng.normal(loc=0, scale=3, size=nSynth))
        self.E = np.vstack((E, E)).T
        
        K = self.curl_free_kernel(theta, self.X, self.X)
        W = self.white_noise_kernel(self.E)
        
        Y = self.rng.multivariate_normal(np.zeros(2 * nSynth), K + W)
        self.Y = self.unflat(Y)
        
        if save:
            self.save_synthetic_data(save, theta, self.X, self.Y, self.E)
            
    def load_fits(self, datafile):
        if datafile == 'hoid':
            self.datafile = '/media/pedro/Data/austinfortino/austinFull.fits'
        elif datafile == 'folio2':
            self.datafile = '/data4/paper/fox/DES/austinFull.fits'
        else:
            self.datafile = datafile

        self.fits = fits.open(self.datafile)
        
    def extract_data(self, nExposure, polyOrder=3, hasGaia=True, sample=None):
        """
        Extract exposure information from current self.fits object.
        
        Parameters:
            nExposure (int): Exposure number from fits file that you want to extract
            polyOrder (int): The order of the unweighted polynomial fit in (u, v) that will be removed from (dx, dy). polyOrder=None means no fit is performed.
            hasGaia (bool): Whether or not to only take data points which has Gaia solutions.
            sample (dict): Dictionary denoting the coordinates (u1, u2, v1, v2) of a rectangle in (u, v). Only points in this rectangle will be kept. sample=None means the entire exposure is used.
        """
        
        self.nExposure = nExposure
        
        self.exposure = self.fits['Residuals'].data[self.fits['Residuals'].data['exposure'] == self.nExposure]
        if polyOrder is not None:
            poly = Poly2d(polyOrder)
            poly.fit(self.exposure['u'], self.exposure['v'], self.exposure['dx'])
            self.exposure['dx'] -= poly.evaluate(self.exposure['u'], self.exposure['v'])
            poly.fit(self.exposure['u'], self.exposure['v'], self.exposure['dy'])
            self.exposure['dy'] -= poly.evaluate(self.exposure['u'], self.exposure['v'])
            
        u = self.exposure['u']
        v = self.exposure['v']
        dx = self.exposure['dx']
        dy = self.exposure['dy']
        E = self.exposure['measErr']
        
        if hasGaia:
            ind_hasGaia = np.where(self.exposure['hasGaia'])[0]
            u = np.take(self.exposure['u'], ind_hasGaia)
            v = np.take(self.exposure['v'], ind_hasGaia)
            dx = np.take(self.exposure['dx'], ind_hasGaia)
            dy = np.take(self.exposure['dy'], ind_hasGaia)
            E = np.take(self.exposure['measErr'], ind_hasGaia)
        
        if sample is not None:
            ind_u = np.logical_and(u >= sample['u1'], u <= sample['u2'])
            ind_v = np.logical_and(v >= sample['v1'], v <= sample['v2'])
            ind = np.where(np.logical_and(ind_u, ind_v))[0]
            
            u = np.take(u, ind, axis=0)
            v = np.take(v, ind, axis=0)
            dx = np.take(dx, ind, axis=0)
            dy = np.take(dy, ind, axis=0)
            E = np.take(E, ind)
            
        self.X = np.vstack((u, v)).T
        self.Y = np.vstack((dx, dy)).T
        self.E = np.vstack((E, E)).T

    def sigma_clip(self, nSigma=4):
        """
        Performs sigma clipping in (dx, dy) to nSigma standard deviations.
        
        Parameters:
            nSigma (int): Number of standard deviations to sigma clip to.
        """
        
        mask = stats.sigma_clip(self.Y, sigma=nSigma, axis=0).mask
        mask = ~np.logical_or(*mask.T)
            
        self.X = self.X[mask, :]
        self.Y = self.Y[mask, :]
        self.E = self.E[mask, :]
        
    def split_data(self, train_size=0.50, test_size=None):
        """Splits the data into training, validation, and testing sets.
        
        Example:
            If train_size=0.60 and test_size=0.50, then the data is partitioned thus:
            
            60% training
            20% validation
            20% testing
        
        Parameters:
            train_size (int or float): Numerical (if int) or fractional (if float) size of data set to be allocated for training (as opposed to validation/testing).
            test_size (int or float): Numerical (if int) or fractional (if float) size of data set to be allocated for testing (as opposed to validation). If test_size=None, then no validation set will be generated.
        """
        self.nData = self.X.shape[0]
        
        Xtrain, Xtv, Ytrain, Ytv, Etrain, Etv = train_test_split(self.X, self.Y, self.E, train_size=train_size, random_state=self.random_state)
        
        if test_size is not None:
            Xvalid, Xtest, Yvalid, Ytest, Evalid, Etest = train_test_split(Xtv, Ytv, Etv, test_size=test_size, random_state=self.random_state)
            self.Xtrain, self.Xvalid, self.Xtest = Xtrain, Xvalid, Xtest
            self.Ytrain, self.Yvalid, self.Ytest = Ytrain, Yvalid, Ytest
            self.Etrain, self.Evalid, self.Etest = Etrain, Evalid, Etest
        
        else:
            self.Xtrain, self.Xtest = Xtrain, Xtv
            self.Ytrain, self.Ytest = Ytrain, Ytv
            self.Etrain, self.Etest = Etrain, Etv

        self.nTrain = self.Xtrain.shape[0]
        self.nTest = self.Xtest.shape[0]
        
    def curl_free_kernel(self, theta, X1, X2):
        """
        This function generates the full (2N, 2N) covariance matrix.
        
        Parameters:
            theta (dict): Dictionary of the kernel parameters.
            X1 (ndarray): (N, 2) 2d array of astrometric positions (u, v).
            X2 (ndarray): (N, 2) 2d array of astrometric positions (u, v).
            
        Returns:
            K (ndarray): (N, N) 2d array, the kernel.
        """
        
        theta = self.fix_params(theta)

        sigma_s = theta['sigma_s'] 
        sigma_x = theta['sigma_x'] 
        sigma_y = theta['sigma_y'] 
        phi = theta['phi']

        #Construct elements of the inverse covariance matrix
        detC = (sigma_x * sigma_y)**2
        a = 0.5*(sigma_x**2 + sigma_y**2)
        b = 0.5*(sigma_x**2 - sigma_y**2)
        b1 = b * np.cos(2*phi)
        b2 = -b * np.sin(2*phi)
        cInv = np.array( [ [a-b1, -b2],[-b2, a+b1]]) / detC

        dX = X1[:,np.newaxis,:] - X2[np.newaxis,:,:]  # Array is N1 x N2 x 2

        cInvX = np.einsum('kl,ijl',cInv,dX)  # Another N1 x N2 x 2

        exponentialFactor = np.exp(-0.5*np.sum(cInvX*dX,axis=2))
        # Multiply the overall prefactor into this scalar array
        exponentialFactor *= sigma_s**2/np.trace(cInv)

        # Start building the master answer, as (N1,N2,2,2) array
        k = np.ones( dX.shape + (2,)) * cInv  # Start with cInv term
        # Now subtract the outer product of cInvX
        k -= cInvX[:,:,:,np.newaxis]*cInvX[:,:,np.newaxis,:]

        # And the exponential
        k = k * exponentialFactor[:,:,np.newaxis,np.newaxis]

        # change (N1,N2,2,2) to (2*N1,2*N2) array
        k = np.moveaxis(k,2,1)
        s = k.shape
        k = k.reshape(s[0]*2,s[2]*2)

        return k

    def white_noise_kernel(self, E):
        """
        This function generates the full (2N, 2N) white kernel covariance matrix.
        
        Parameters:
            theta (dict): Dictionary of the kernel parameters.
            E (ndarray): 2d array of measurement errors (standard deviations)
        """
        
        return np.diag(self.flat(E)**2)

    def get_nLML(self, theta):
        """
        This function takes in kernel paramters values, theta, fits the model and returns the negative log marginal likelihood.
        
        This function is the function you should pass to scipy.optimize.minimize or any other scipy optimizer.
        
        Parameters:
            theta (dict/list/ndarray): Dictionary or list/ndarray of the kernel parameters.
        """
        
        if self.i is not None:
            self.i += 1
        
        theta = self.fix_params(theta)
        self.print_params(theta, time=True, fix_phi=False, output=True)
        self.fit(theta)
        
        bounds = {
            'sigma_s': (0, 1e4),
            'sigma_x': (0, 1),
            'sigma_y': (0, 1),
            'phi': (-2*np.pi, 2*np.pi),
        }
        
        penalty_factor = self.nData
        for key, value in bounds.items():
            param = theta[key]

            if param <= value[0]:
                penalty = (value[0] - param) * penalty_factor
            elif param >= value[1]:
                penalty = (param - value[1]) * penalty_factor
            else:
                penalty = 0
                
        self.predict(self.Xtest)
        # self.get_chisq(self.Ytest, self.fbar_s, self.Etest)
        # return self.chisq + penalty

        u, v, dx, dy = self.Xtest[:, 0], self.Xtest[:, 1], self.Ytest[:, 0] - self.fbar_s[:, 0], self.Ytest[:, 1] - self.fbar_s[:, 1]
        logr, xiplus, ximinus, xicross, xiz2 = vcorr(u, v, dx, dy)
        
        # Calculate xiE and xiB
        dlogr = np.zeros_like(logr)
        dlogr[1:-1] = 0.5 * (logr[2:] - logr[:-2])
        tmp = np.array(ximinus) * dlogr
        integral = np.cumsum(tmp[::-1])[::-1]
        xiB = 0.5 * (xiplus - ximinus) + integral
        xiE = xiplus - xiB
    
        # print(np.nanmean(xiE[:50]), penalty)
        return np.nanmean(xiE[:50]) + penalty
    
    def correlation_fit(self, rmax=0.3, bins=75, fn=None, p0=None, bounds=None):
        """
        Fits correlation function to a specific function, fn, in order to get a better initial guess for the optimizer.
        
        Parameters:
            rmax (float): Largest separation between to points to calculate the correlation function for.
            bins (int): How many bins to put the correlation function into (will return an array of shape (bins, bins)).
            fn (function): You can supply you're own function to fit as long as it has the same inputs as the fn defined in this method.
            p0 (list/ndarray): Starting parameter values [sigma_s, sigma_x, sigma_y, phi].
            bounds (ndarray of length 2 tuples): Upper and lower bounds for the optimizer for each parameter (note the optimizer actually uses the transpose of this, but it's clunkier to specified that).
        """
        
        # Find the correlation function of the actual data.
        xiplus, counts = vcorr2d(self.X[:, 0], self.X[:, 1], self.Y[:, 0], self.Y[:, 1], rmax=rmax, bins=bins)
        observed_data = xiplus.ravel()
        
        def fn(x, sigma_s, sigma_x, sigma_y, phi):
            """
            Gary's version of a function that calculates trace of Kuv for
            a given set of x=(du,dv).
            """
            # Make the tuple du,dv into an Nx2 array.
            dX = np.vstack( (np.array(x[0]), np.array(x[1]))).transpose()
            # print(dX.shape)##

            #Construct elements of the inverse covariance matrix
            detC = (sigma_x * sigma_y)**2
            a = 0.5*(sigma_x**2 + sigma_y**2)
            b = 0.5*(sigma_x**2 - sigma_y**2)
            b1 = b * np.cos(2*phi)
            b2 = -b * np.sin(2*phi)
            cInv = np.array( [ [a-b1, -b2],[-b2, a+b1]]) / detC
            traceCInv = np.trace(cInv)

            cInvX = np.einsum('kl,jl',cInv,dX)  # Another N x 2 array

            # Now make the Tr(cInv) - Tr(cInvX * cInvX^T) vector
            k = traceCInv - np.sum(cInvX*cInvX, axis=1)

            # Multiply by the exponential
            k *= np.exp(-0.5*np.sum(cInvX*dX,axis=1))

            # And rescale to sigma_s**2 at origin.
            k *= sigma_s**2/traceCInv

            # Return a scalar if input was scalar, else an array
            if k.size==1:
                return k[0]
            else:
                return k.flatten()

        # Create the x,y grid to evaluate the model function on.
        xmin, xmax, nx = -rmax, rmax, xiplus.shape[0]
        ymin, ymax, ny = -rmax, rmax, xiplus.shape[1]
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        x = np.vstack([X.ravel(), Y.ravel()])
        
        # These are the standard starting value and bounds that seem to work well
        if p0 is None:
            p0 = np.array([1e2, 10**(-1.217), .1, 0])
            
        if bounds is None:
            bounds = np.array([
                (1, 1e8),
                (((264*u.mas).to(u.deg)).value*10, 5),
                (((264*u.mas).to(u.deg)).value*10, 5),
                (-2*np.pi, 2*np.pi)
            ])
            
        # Perform the fit.
        self.theta_fit, self.theta_fit_cov = opt.curve_fit(
            fn,
            x,
            observed_data,
            p0=p0,
            bounds=bounds.T)
        
        self.theta0 = self.theta_fit
        
    def optimize(self, v0=None):
        """
        This function calls the Nelder-Mead Scipy optimizer to minimize the nLML (negative Log Marginal Likeliood) as a function of kernel parameters.
        
        Parameters:
            v0 (None/np.ndarray/str): If v0=None, then this method will use the output of correlation_fit as the first vertex of the simplex. If v0 is np.ndarray of shape (5,) then that will be the first vertex. If v0='default' then this method will use a default, somewhat reasonable guess at the first vertex of the simplex.
            fatol (float): The function (nLML) evaluated at each vertex of the simplex must all be within fatol of each other.
            xatol (float): The maximum size of the final simplex. Smaller values will give more precise results at the cost of more function evaluations.
        """
        
        if v0 is None:
            v0 = self.theta0
        elif v0 == 'default':
            v0 = np.array([1e2, 10**(-1.217), .1, 0])
        else:
            assert isinstance(v0, (np.ndarray, list)), f"Type of v0 should be np.ndarray or list, but instead is {type(v0)}."
            assert np.array(v0).shape == (4,), f"Shape of v0 should be (4,), but instead is {np.array(v0).shape}."
            v0 = np.array(v0)
        
        # Create shape (p+1, p) array (the Nelder-Mead simplex) for p number of parameters
        simplex0 = np.vstack([v0, np.vstack([v0]*4) + np.diag(v0*0.15)])

        options = {
            "initial_simplex": simplex0,
            "fatol": 5,
            "xatol": 0.1
        }
        
        self.i = 0
        opt_result = opt.minimize(
            self.get_nLML,
            simplex0[0],
            method='Nelder-Mead',
            options=options
        )
        
        assert opt_result.success, f"Optimizer failed to end successfully.\nStatus: {opt_result.status}\nMessage: {opt_result.message}"
        self.theta = opt_result.x
        self.nLML_final = opt_result.fun
        self.opt_result = opt_result.copy()
        
        self.i = None        

    def fit(self, theta):
        """
        Fits a curl-free GPR to the data contained in self.X, self.Y, and self.E given kernel parameters theta.
        
        Parameters:
            theta (dict): Dictionary of the kernel parameters.
        """
        
        self.theta = theta

        self.K = self.curl_free_kernel(self.theta, self.Xtrain, self.Xtrain)
        self.W = self.white_noise_kernel(self.Etrain)
        self.L = np.linalg.cholesky(self.K + self.W)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.flat(self.Ytrain)))
        self.nLML = -((-1/2) * np.dot(self.flat(self.Ytrain), self.alpha) - np.sum(np.log(np.diag(self.L))) - (self.nTest / 2) * np.log(2 * np.pi))

    def predict(self, X, full=False):     
        """
        Predicts new astrometric residuals (dx, dy) based on the curl-free GPR model and some astrometric positions.
        
        Parameters:
            Xnew (ndarray): Shape (N, 2) array of astrometric positions (u, v).
            full (bool): Compute posterior predictive variance (V_s) as well as the posterior predictive mean (fbar_s)
        """
    
        self.Ks = self.curl_free_kernel(self.theta, self.Xtrain, X)
        self.fbar_s = self.unflat(np.dot(self.Ks.T, self.alpha))

        if full:
            self.Kss = self.curl_free_kernel(self.theta, X, X)
            self.v = np.linalg.solve(self.L, self.Ks)
            self.V_s = self.Kss - np.dot(self.v.T, self.v)
            self.sigma = np.sqrt(np.abs(np.diag(self.V_s)))
            
    def get_chisq(self, Y, Yhat, sigma):
        self.chisq = np.sum(((Y[:, 0] - Yhat[:, 0]) / sigma[:, 0])**2 + ((Y[:, 1] - Yhat[:, 1]) / sigma[:, 1])**2)
        
    def print_params(self, theta, time=False, fix_phi=True, output=False):
        """This function prints the kernel parameters in a pleasing way."""
        
        theta = self.fix_params(theta)
        
        if self.i is not None:
            i = f'{self.i:<3}: '
        else:
            i = ''
        
        if fix_phi:
            theta['phi'] = self.fix_phi(theta['phi'])
        
        if time:
            param_time = f'{datetime.now()}: '
        else:
            param_time = ''
        
        params = i + param_time + ' '.join([f"{name:>8}: {x:<15.5f}" for name, x in theta.items()])
        
        if output:
            with open(os.path.join(self.outdir, "params.out"), mode='a+') as file:
                file.write(params + '\n')

        if self.printing:
            print(params)
        
    def fix_phi(self, phi):
        """
        Takes the angle, phi, and adds or subtracts multiples of pi until phi is between 0 and pi. Because the kernel evaluated at phi and at phi + pi are equivalent, this does not change the kernel in anyway. This is only useful for a human looking at the value of phi.
        
        Parameters:
            phi (float): An angle in radians."""
        
        while phi < 0:
            phi += np.pi
            
        while phi > np.pi:
            phi -= np.pi

        return phi
    
    def fix_params(self, theta):
        """
        This function takes in the kernel parameters array/dict, theta, and puts it in dict form if it's not already.
        
        Parameters:
            theta (np.ndarray, list, dict): Kernel parameter values.
        """

        if isinstance(theta, dict):
            return theta
        
        elif isinstance(theta, (np.ndarray, list)):
            theta_new =  {
                'sigma_s': theta[0],
                'sigma_x': theta[1],
                'sigma_y': theta[2],
                'phi': theta[3],
            }    
            return theta_new
        
        else:
            raise TypeError(f"type(theta) is {type(theta)} when it should be dict or list/ndarray.")

        
    def flat(self, arr):
        """
        Flattens 2d arrays in row-major order.
        
        Useful because the more natural way to process data is to use arrays of shape (N, 2), but for a curl-free kernel a shape (2N,) array is necessary.
        
        Parameters:
            arr (ndarray): 2d array to be flattened in row-major order (C order).
            
        Returns:
            arr (ndarray): 1d array.
        """
        
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2
        return arr.flatten(order='C')
    
    def unflat(self, arr):
        """Unflattens a 1d array in row-major order.
        
        Turns a shape (2N,) array into a shape (N, 2) array.
        
        Parameters:
            arr (ndarray): 1d array to be unflattened in row-major order (C order).
            
        Returns:
            arr (ndarray): 2d array.
        """
        
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
        return arr.reshape((arr.shape[0] // 2, 2), order='C')
    
    def wrap(self):
        
        self.plot_residuals(self.Xtest, self.Ytest, self.Etest, save=True)
        self.plot_residuals(self.Xtest, self.Ytest - self.fbar_s, self.Etest, save=True, ext='GPR_applied_')
        
        self.plot_div_curl(self.Xtest, self.Ytest, self.Etest, save=True)
        self.plot_div_curl(self.Xtest, self.Ytest - self.fbar_s, self.Etest, save=True, ext='GPR_applied_')
        
        self.plot_Emode_2ptcorr(self.Xtest, self.Ytest, Y2=self.Ytest - self.fbar_s, save=True)
        self.plot_xiplus_2d(self.Xtest, self.Ytest, Y2=self.fbar_s, save=True)
        
        nAvg = 50
        with open(os.path.join(self.outdir, "params.out"), mode='a+') as file:
            file.write(f"Mean of first {nAvg} points (Emode (Observed)): {np.nanmean(self.xiE[:nAvg])}\n")
            file.write(f"Mean of first {nAvg} points (Bmode (Observed)): {np.nanmean(self.xiB[:nAvg])}\n")
            file.write(f"Mean of first {nAvg} points (Emode (GPR Applied)): {np.nanmean(self.xiE2[:nAvg])}\n")
            file.write(f"Mean of first {nAvg} points (Bmode (GPR Applied)): {np.nanmean(self.xiB2[:nAvg])}\n")
            file.write(f"Ratio of E modes: {np.nanmean(self.xiE[:nAvg]) / np.nanmean(self.xiE2[:nAvg])}\n")
            file.write(f"Ratio of B modes: {np.nanmean(self.xiB[:nAvg]) / np.nanmean(self.xiB2[:nAvg])}\n")
        
        np.savez(
            os.path.join(self.outdir, f"{self.nExposure}.npz"),
            fbar_s=self.fbar_s,
#             nLML=self.nLML,
#             chisq=self.chisq,
#             theta=np.array(self.theta),
#             theta0=np.array(self.theta0),
            random_state=self.random_state,
            nExposure=self.nExposure,
            nTrain=self.nTrain,
            nTest=self.nTest,
            nData=self.nData,
#             nParams=self.nParams,
            X=self.X,
            Xtrain=self.Xtrain,
            Xtest=self.Xtest,
            Y=self.Y,
            Ytrain=self.Ytrain,
            Ytest=self.Ytest,
            E=self.E,
            Etrain=self.Etrain,
            Etest=self.Etest,
            xiE=self.xiE,
            xiB=self.xiB
        )
        
    def plot_residuals(self, X, Y, E, binpix=512, save=False, ext=''):
        """
        Plots astrometric residuals as a function of astrometric position on the sky.
        
        Parameters:
            X (ndarray): (N, 2) array of (u, v) astrometric positions.
            Y (ndarray): (N, 2) array of (dx, dy) astrometric residuals.
            E (ndarray): (N,) array of measurement errors.
        """
        
        u, v, dx, dy, arrowScale = residInPixels(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1], E[:, 0], binpix=binpix)

        plt.figure(figsize=(8, 8))
        quiver = plt.quiver(
            u, v, dx, dy,
            pivot='middle',
            color='green',
            angles='xy',
            scale_units='xy',
            scale=arrowScale,
            units='x')

        quiverkey = plt.quiverkey(quiver, 0.5, 0.5, 20, '{:.1f} mas'.format(20), coordinates='axes', color='red', labelpos='N', labelcolor='red')
        
        plt.xlim(np.min(u) - xyBuffer, np.max(u) + xyBuffer)
        plt.ylim(np.min(v) - xyBuffer, np.max(v) + xyBuffer)

        plt.gca().set_aspect('equal')
        plt.grid()
        
        if save:
            plt.savefig(os.path.join(self.outdir, ext + "residuals.pdf"))
        
        plt.show()
        
    def plot_div_curl(self, X, Y, E, binpix=1024, save=False, ext=''):
        """
        Plot div and curl of the specified vector field, assumed to be samples from a grid.
        
        Parameters:
            X (ndarray): (N, 2) array of (u, v) astrometric positions.
            Y (ndarray): (N, 2) array of (dx, dy) astrometric residuals.
            E (ndarray): (N,) array of measurement errors.
        """
        
        u, v, dx, dy, arrowScale = residInPixels(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1], E[:, 0], binpix=binpix)

        # This line has been replaced because sometimes this step happens to be zero and messes it all up
        # Removing all the points from np.diff(u) that are zero seems to have little to no effect on the
        # resulting plot and helps get rid of this fairly common error.
        # 31 January 2020
        # step = np.min(np.abs(np.diff(u)))
        step = np.min(np.abs(np.diff(u)[np.diff(u) != 0]))
        
        ix = np.array( np.floor(u/step+0.1), dtype=int)
        iy = np.array( np.floor(v/step+0.1), dtype=int)
        ix = ix - np.min(ix)
        iy = iy - np.min(iy)
        dx2d = np.zeros( (np.max(iy)+1,np.max(ix)+1), dtype=float)
        dy2d = np.array(dx2d)
        valid = np.zeros_like(dx2d, dtype=bool)
        valid[iy,ix] = True
        dx2d[iy,ix] = dx
        dy2d[iy,ix] = dy
        div, curl = calcEB(dy2d, dx2d, valid)
        
        scale = 50
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        divplot = axes[0].imshow(div, origin='lower', cmap='Spectral', vmin=-scale, vmax=scale)
        axes[0].axis('off')
        axes[0].set_title('Divergence')

        curlplot = axes[1].imshow(curl, origin='lower', cmap='Spectral', vmin=-scale, vmax=scale)
        axes[1].axis('off')
        axes[1].set_title('Curl')
        
        fig.colorbar(divplot, ax=fig.get_axes())
        
        if save:
            plt.savefig(os.path.join(self.outdir, ext + "div_curl.pdf"))
        
        plt.show()
        
        # Some stats:
        vardiv = gbutil.clippedMean(div[div==div],5.)[1]
        varcurl = gbutil.clippedMean(curl[div==div],5.)[1]
        print("RMS of div: {:.2f}; curl: {:.2f}".format(np.sqrt(vardiv), np.sqrt(varcurl)))
        
    def plot_Emode_2ptcorr(self, X, Y, Y2=None, Bmode=True, rrange=(5./3600., 1.5), nbins=100, nAvg=30, plot_avg_line=True, save=False):
        """
        Use treecorr to produce angle-averaged 2-point correlation functions of astrometric error for the supplied sample of data, using brute-force pair counting.
        
        Returns:
            logr (ndarray): mean log of radius in each bin
            xi_+ (ndarray): <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
            xi_- (ndarray): <vr1 vr2 - vt1 vt2>
        """
        
        # Initialize the pyplot figure
        plt.figure(figsize=(8, 8))
        plt.title("E Mode Correlation")
        plt.xlabel('Separation (degrees)')
        plt.ylabel('xi (mas^2)')
        
        # Solve for weighted and pixelized residuals, as well as angle averaged correlation function.
        u, v, dx, dy = X[:, 0], X[:, 1], Y[:, 0], Y[:, 1]
        logr, xiplus, ximinus, xicross, xiz2 = vcorr(u, v, dx, dy)
        
        # Calculate xiE and xiB
        dlogr = np.zeros_like(logr)
        dlogr[1:-1] = 0.5 * (logr[2:] - logr[:-2])
        tmp = np.array(ximinus) * dlogr
        integral = np.cumsum(tmp[::-1])[::-1]
        self.xiB = 0.5 * (xiplus - ximinus) + integral
        self.xiE = xiplus - self.xiB
        
        # Plot xiE
        plt.semilogx(np.exp(logr), self.xiE, 'r.', label="E Mode (Observed)")
        print(f"Mean of first {nAvg} points (Emode (Observed)): ", np.nanmean(self.xiE[:nAvg]))

        if Bmode:
            # Plot xiB
            plt.title("E and B Mode Correlation")
            plt.semilogx(np.exp(logr), self.xiB, 'b.', label="B Mode (Observed)")
            plt.legend(framealpha=0.3)
            print(f"Mean of first {nAvg} points (Bmode (Observed)): ", np.nanmean(self.xiB[:nAvg]))
        
        
        if Y2 is not None:
            
            # Solve for weighted and pixelized residuals, as well as angle averaged correlation function.
            u, v, dx, dy = X[:, 0], X[:, 1], Y2[:, 0], Y2[:, 1]
            logr2, xiplus2, ximinus2, xicross2, xiz22 = vcorr(u, v, dx, dy)
            
            # Calculate xiE and xiB
            dlogr2 = np.zeros_like(logr2)
            dlogr2[1:-1] = 0.5 * (logr2[2:] - logr2[:-2])
            tmp2 = np.array(ximinus2) * dlogr2
            integral2 = np.cumsum(tmp2[::-1])[::-1]
            self.xiB2 = 0.5 * (xiplus2 - ximinus2) + integral2
            self.xiE2 = xiplus2 - self.xiB2
            
            # Plot xiE
            plt.semilogx(np.exp(logr2), self.xiE2, 'rx', label="E Mode (GPR Applied)")    
            print(f"Mean of first {nAvg} points (Emode (GPR Applied)): ", np.nanmean(self.xiE2[:nAvg]))
            
            if Bmode:
                # Plot xiB
                plt.semilogx(np.exp(logr2), self.xiB2, 'bx', label='B Mode (GPR Applied)')
                print(f"Mean of first {nAvg} points (Bmode (GPR Applied)): ", np.nanmean(self.xiB2[:nAvg]))
            
            
            plt.legend(framealpha=0.3)
            
            print(f"Ratio of E modes: {np.nanmean(self.xiE[:nAvg]) / np.nanmean(self.xiE2[:nAvg])}")
            print(f"Ratio of B modes: {np.nanmean(self.xiB[:nAvg]) / np.nanmean(self.xiB2[:nAvg])}")

            
        if plot_avg_line:
            plt.axvline(x=np.exp(logr)[nAvg], color='k', linestyle='--')
        
        # Show plots
        plt.grid()
        
        if save:
            plt.savefig(os.path.join(self.outdir, "Emode.pdf"))
        
        plt.show()
        
    def plot_xiplus_2d(self, X, Y, Y2=None, rmax=0.3, bins=75, vmin=-100, vmax=450, save=False):

        ncols = 1
        xiplus, counts = vcorr2d(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1], rmax=rmax, bins=bins)
        if Y2 is not None:
            ncols += 2
            xiplus2, counts = vcorr2d(X[:, 0], X[:, 1], Y2[:, 0], Y2[:, 1], rmax=rmax, bins=bins)
            xiplus3, counts = vcorr2d(X[:, 0], X[:, 1], Y[:, 0] - Y2[:, 0], Y[:, 1] - Y2[:, 1], rmax=rmax, bins=bins)

        if vmin is None:
            vmin = xiplus.min()
        if vmax is None:
            vmax = xiplus.max()

        fig, axes = plt.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True, figsize=(8*ncols, 8*ncols))

        if Y2 is None:
            axes.set_title("2D Correlation")
            axes.set_xlabel("Bins")
            axes.set_ylabel("Bins")
            im = axes.imshow(xiplus, origin='lower', cmap='Spectral', interpolation='nearest', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im)

        else:
            fig.subplots_adjust(wspace=0)

            axes[0].set_title("Observed")
            axes[0].set_xlabel("Bins")
            axes[0].set_ylabel("Bins")
            im = axes[0].imshow(xiplus, origin='lower', cmap='Spectral', interpolation='nearest', vmin=vmin, vmax=vmax)

            axes[1].set_title("GPR")
            axes[1].set_xlabel("Bins")
            im = axes[1].imshow(xiplus2, origin='lower', cmap='Spectral', interpolation='nearest', vmin=vmin, vmax=vmax)

            axes[2].set_title("GPR Applied")
            axes[2].set_xlabel("Bins")
            im = axes[2].imshow(xiplus3, origin='lower', cmap='Spectral', interpolation='nearest', vmin=vmin, vmax=vmax)

            cb_ax = fig.add_axes([0.92, 0.375, 0.025, 0.25])
            cbar = fig.colorbar(im, cax=cb_ax)

        cbar.set_label('Correlation (mas^2)', rotation=270)
        
        if save:
            plt.savefig(os.path.join(self.outdir, f"{save}xiplus_2d.pdf"))
            
        plt.show()
        
    def summarize(self):
        print(f"Log Marginal Likelihood: {self.nLML}")
        print(f"chisq: {self.chisq}")
        self.print_params(self.theta)
        self.plot_residuals(self.Xtest, self.Ytest, self.Etest)
        self.plot_residuals(self.Xtest, self.Ytest - self.fbar_s, self.Etest)
        self.plot_div_curl(self.Xtest, self.Ytest, self.Etest)
        self.plot_div_curl(self.Xtest, self.Ytest - self.fbar_s, self.Etest)
        self.plot_Emode_2ptcorr(self.Xtest, self.Ytest, Y2=self.Ytest - self.fbar_s)
        self.plot_xiplus_2d(self.Xtest, self.Ytest, Y2=self.fbar_s)



class Poly2d:
    # Polynomial function of 2 dimensions
    def __init__(self, order, sumOrder=True):
        # Set up order; sumOrder=True if order is max sum of powers of x and y.
        self.order = order
        mask = np.ones((self.order+1, self.order+1), dtype=bool)
        if sumOrder:
            # Clear mask where sum of x & y orders is >order
            for i in range(1, self.order+1):
                mask[i, self.order-i+1:] = False
        self.use = mask
        self.sumOrder = sumOrder
        self.coeffs = None
        return
    
    def evaluate(self, x, y):
        xypow = np.ones((x.shape[0], self.order+1), dtype=float)
        for i in range(1, self.order+1):
            xypow[:, i] = xypow[:, i-1] * x
        tmp = np.dot(xypow, self.coeffs)
        for i in range(1, self.order+1):
            xypow[:, i] = xypow[:, i-1] * y
        return np.sum(tmp * xypow, axis=1)
    
    def fit(self, x, y, z):
        # Least-square fit polynomial coefficients to P(x,y)=z
        # Make cofactor array
        npts = x.shape[0]
        xpow = np.ones((npts, self.order+1), dtype=float)
        for i in range(1, self.order+1):
            xpow[:, i] = xpow[:, i-1] * x
        ypow = np.ones_like(xpow)
        for i in range(1, self.order+1):
            ypow[:, i] = ypow[:, i-1] * y
        A = xpow[:, :, np.newaxis] * ypow[:, np.newaxis, :]
        # Retain powers wanted
        A = A.reshape(npts, (self.order+1) * (self.order+1))[:, self.use.flatten()]
        b = np.linalg.lstsq(A, z, rcond=None)[0]
        self.coeffs = np.zeros((self.order+1, self.order+1), dtype=float)
        self.coeffs[self.use] = b
        return
    
    def getCoeffs(self):
        # Return the current coefficients in a vector.
        return self.coeffs[self.use]
    
    def setCoeffs(self,c):
        # Set the coefficients to the specified vector.
        if len(c.shape) != 1 or c.shape[0] != np.count_nonzero(self.use):
            print("Poly2d.setCoeffs did not get proper-size array", c.shape)
            sys.exit(1)
        self.coeffs[self.use] = c
        return
    
pixScale = 264.   # nominal mas per pixel
xyBuffer = 0.05   # buffer in degrees
    
def residInPixels(u, v, dx, dy, E, binpix=512, scaleFudge=1., maxErr=50):
    '''
    Return a 2d vector diagram of weighted mean astrometric residual in pixelized areas.
    Residuals shown in milliarcsec, and the binned resid arrays are returned.
    binpix is the width of cells (in nominal pixel size)
    scaleFudge is multiplier of arrow length scaling (and key size) 
    to apply to default choice. By default, arrows are scaled so that 
    typical arrow is of length equal to the distance between arrows (cell size).

    maxErr is the largest error that a binned vector can have (mas) and still be plotted,
    to avoid cluttering up the plot with noisy arrows.
    '''
    
    noData = np.nan
    
    if len(u) < 100:
        # Not enough residuals to even try.
        return None
    # use exposure coordinates:
    x = u  ### Problem is these are not relative to array center!!
    y = v
    residx = dx
    residy = dy
    sig = E
    
    cellSize = binpix * pixScale / (1000.*3600.)

    weight = np.where(sig>0, 1./(sig*sig), 0.)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    
    xsize = int(np.ceil( (xmax-xmin) / cellSize))
    ysize = int(np.ceil( (ymax-ymin) / cellSize))

    # Figure out the image pixel for each residual point
    ix = np.array( np.floor( (x-xmin) / cellSize), dtype=int)
    ix = np.clip(ix, 0, xsize-1)
    iy = np.array( np.floor( (y-ymin) / cellSize), dtype=int)
    iy = np.clip(iy, 0, ysize-1)
    index = iy*xsize + ix
    
    sumxw = np.histogram(index, bins=xsize*ysize, range=(-0.5,xsize*ysize+0.5),
                        weights=weight*residx)[0]
    sumyw = np.histogram(index, bins=xsize*ysize, range=(-0.5,xsize*ysize+0.5),
                        weights=weight*residy)[0]
    sumw = np.histogram(index, bins=xsize*ysize, range=(-0.5,xsize*ysize+0.5),
                        weights=weight)[0]
    # Smallest weight that we'd want to plot a point for
    minWeight = maxErr**(-2)
    # Value to put in where weight is below threshold:
    sumxw = np.where( sumw > minWeight, sumxw / sumw, noData)
    sumyw = np.where( sumw > minWeight, sumyw / sumw, noData)
    sumw = np.where( sumw > minWeight, 1./sumw, noData)
    rmsx = np.std(sumxw[sumw>0.])
    rmsy = np.std(sumyw[sumw>0.])
    print('RMSx, RMSy, noise:', rmsx, rmsy, np.sqrt(np.mean(sumw[sumw>0.])))

    # Make an x and y position for each cell to be the center of its cell
    xpos = np.arange(xsize*ysize,dtype=int)
    xpos = ((xpos%xsize)+0.5)*cellSize + xmin
    ypos = np.arange(xsize*ysize,dtype=int)
    ypos = ((ypos//xsize)+0.5)*cellSize + ymin

    useful = np.logical_and(sumw!=0, sumw<(maxErr*maxErr))
    dx = sumxw[useful]
    dy = sumyw[useful]
    xpos = xpos[useful]
    ypos = ypos[useful]

    arrowScale = 20 / cellSize
    arrowScale /= scaleFudge   # Adjust lengths of arrows if desired
    
    return xpos, ypos, dx, dy, arrowScale

def calcEB(dx, dy, valid):
    """
    Given vector displacement (dx,dy) defined on identical 2d grids,
    and boolean 2d array valid which is True where data are valid,
    return arrays giving divergence and curl of the vector field.
    These will have NaN in pixels without useful info.
    """
    dxdx = dx[2:,1:-1] - dx[:-2,1:-1]
    dydx = dy[2:,1:-1] - dy[:-2,1:-1]
    dxdy = dx[1:-1,2:] - dx[1:-1,:-2]
    dydy = dy[1:-1,2:] - dy[1:-1,:-2]
    use = np.logical_and(valid[1:-1,:-2], valid[1:-1,2:])
    use = np.logical_and(use, valid[:-2,1:-1])
    use = np.logical_and(use, valid[2:,1:-1])
    div = np.where(use, dxdx+dydy, np.nan)
    curl= np.where(use, dydx-dxdy, np.nan)
    return div, curl

def vcorr(u, v, dx, dy, rmin=5./3600., rmax=1.5, dlogr=0.05):
    """
    Produce angle-averaged 2-point correlation functions of astrometric error
    for the supplied sample of data, using brute-force pair counting.
    Output are the following functions:
    logr - mean log of radius in each bin
    xi_+ - <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
    xi_- - <vr1 vr2 - vt1 vt2>
    xi_x - <vr1 vt2 + vt1 vr2>
    xi_z2 - <vx1 vx2 - vy1 vy2 + 2 i vx1 vy2>
    """

#     print("Length ",len(u))
    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(u))
    # Omit self-pairs
    use = i1!=i2
    i1 = i1[use]
    i2 = i2[use]
    del use
    
    # Make complex separation vector
    dr = 1j * (v[i2]-v[i1])
    dr += u[i2]-u[i1]

    # log radius vector used to bin data
    logdr = np.log(np.absolute(dr))
    logrmin = np.log(rmin)
    bins = int(np.ceil(np.log(rmax/rmin)/dlogr))
    hrange = (logrmin, logrmin+bins*dlogr)
    counts = np.histogram(logdr, bins=bins, range=hrange)[0]
    logr = np.histogram(logdr, bins=bins, range=hrange, weights=logdr)[0] / counts

    # First accumulate un-rotated stats
    vec =  dx + 1j*dy
    vvec = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram(logdr, bins=bins, range=hrange, weights=vvec)[0]/counts
    vvec = vec[i1] * vec[i2]
    xiz2 = np.histogram(logdr, bins=bins, range=hrange, weights=vvec)[0]/counts

    # Now rotate into radial / perp components
#     print(type(vvec),type(dr)) ###
    tmp = vvec * np.conj(dr)
    vvec = tmp * np.conj(dr)
    dr = dr.real*dr.real + dr.imag*dr.imag
    vvec /= dr
    del dr
    ximinus = np.histogram(logdr, bins=bins, range=hrange, weights=vvec)[0]/counts
    xicross = np.imag(ximinus)
    ximinus = np.real(ximinus)

    return logr, xiplus, ximinus, xicross, xiz2

def vcorr2d(u, v, dx, dy, rmax=1., bins=513):
    """
    Produce 2d 2-point correlation function of total displacement power
    for the supplied sample of data, using brute-force pair counting.
    Output are 2d arrays giving the 2PCF and then the number of pairs that
    went into each bin.  The 2PCF calculated is
    xi_+ - <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
    Output function will be symmetric about origin.
    """

    hrange = [ [-rmax,rmax], [-rmax,rmax] ]

#     print("Length ",len(u))
    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(u))
    # Omit self-pairs
    use = i1 != i2
    i1 = i1[use]
    i2 = i2[use]
    del use
    
    # Make separation vectors and count pairs
    yshift = v[i2] - v[i1]
    xshift = u[i2] - u[i1]
    counts = np.histogram2d(xshift, yshift, bins=bins, range=hrange)[0]

    # Accumulate displacement sums
    vec =  dx + 1j*dy
    vvec = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=vvec)[0]/counts

    xiplus = 0.5*(xiplus + xiplus[::-1,::-1])  # Combine pairs
    return xiplus, counts