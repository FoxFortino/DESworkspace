# Standard modules
import os

# Willow Fox Fortino's modules
import GPRutils
import vonkarmanFT as vk

# Science modules
import scipy.linalg
import numpy as np
import astropy.units as u
import scipy.optimize as opt

from IPython import embed


class vonKarman2KernelGPR(object):
    """
    2D Gaussian Process Regression (GPR) with a von Karman kernel.

    This class performs GPR on a 2D dataset (2D input values and 2D target
    values) using a kernel function based on the von Karman model of
    atmospheric turbulence.


    """

    def __init__(
        self,
        dataContainer: object,
        curl: bool = False,
        printing: bool = True,
        outDir: str = None
            ) -> None:
        """
        Initialize a vonKarman2KernelGPR object.

        Takes in a few arguments to determine some of the behavior of the
        model.

        Arguments
        ---------
        dataContainer : GPRutils.dataContainer
            This object will keep track of all of the astrometric data
            necessary for performing GPR on DES astrometry data.

        Keyword Arguments
        -----------------
        curl : bool
            Whether or not to take advantage of the expected curl-free nature
            of the atmospheric turbulence field.
        printing : bool
            Whether or not to print kernel parameters at each stage of
            optimization of self.fitCorr and self.optimize to std.out.
        outDir : str
            The path to the directory that params.out, the file that all of
            the kernel parameters  from self.fitCorr and self.optimize, will
            be written to. Typically there will exists some directory: "path
            to/??????" where ?????? is the exposure number. Inside that
            directory is normally where params.out and the completed FITS file
            is saved (see GPRutils.dataContainer.saveFITS)
        """
        self.dC = dataContainer
        self.dC.curl = curl
        self.printing = printing

        # If outDir is specified then create a file, params.out, that kernel
        # parameters will be written to.
        if outDir is not None:
            # Sort out the filename of the param file
            self.paramFile = ["GPR", str(self.dC.expNum)]
            ext = self.dC.band
            if self.dC.useRMS:
                ext += "R"
            if self.dC.curl:
                ext += "C"
            self.paramFile.append(ext)
            self.paramFile.append("out")
            self.paramFile = ".".join(self.paramFile)

            self.paramFile = os.path.join(outDir, self.paramFile)
            if os.path.exists(self.paramFile):
                os.remove(self.paramFile)

        # If outDir is not specified then it is convenient for self.paramFile
        # to be None
        else:
            self.paramFile = None

    @u.quantity_input
    def fitCorr(
        self,
        v0: np.ndarray = None,
        rmax: u.arcmin = 5*u.arcmin,
        nBins: int = 50
            ) -> None:
        """
        Fit the correlation function of the data to a von karman model.

        Fit the two-dimensional two-point correlation function (xi_+) of the
        astrometric residual field against the two-dimensional two-point
        correlatioon function (xi_+) of the von karman power spectrum model of
        atmospherc turbulence. This method uses the Nelder-Mead optmizer to
        optimize the von karman kernel parameters.

        The von karman model is essential an approximation of the power
        spectrum of the atmospheric turbulence field. With a fourier transform
        you can go between the power spectrum and the correlation function.
        This power spectrum, and therefore the correlation function, depends
        on a set of parameters. We call these parameters the kernel
        parameters, or kernel hyper-parameters, because we use the von karman
        model in the context of GPR.

        Keyword Arguments
        ---------
        v0 : np.ndarray
            Define an initial guess of the kernel parameters for the optimizer.
        rmax : Quantity object (angle)
            Maximum pair separation to calculate xi_+ to.
        nBins : int
            Number of bins that xi_+ is calculated in. The output of the
            xi_+ array will be of shape (nBins, nBins).
        """
        # Define the function that will accept a set of kernel parameters and
        # then calculate and return the figure-of-merit for those parameters.
        def figureOfMerit_fitCorr(params, xx, yy, xiplus):
            ttt = vk.TurbulentLayer(
                variance=params[0],
                outerScale=params[1],
                diameter=params[2],
                wind=(params[3], params[4]))

            Cuv = ttt.getCuv(xx, yy)
            xiplus_model = Cuv[:, :, 0, 0] + Cuv[:, :, 1, 1]
            xiplus_model = np.where(np.isnan(xiplus_model), 0, xiplus_model)

            RSS = np.sum((xiplus - xiplus_model)**2) / self.dC.nTrain

            GPRutils.printParams(
                params,
                FoM=RSS,
                file=self.paramFile,
                printing=self.printing
                )

            return RSS

        # Calculate the 2D xiplus of the raw data that will be fitted against
        # the xiplus of the von karman power spectrum model of atmospheric
        # turbulence.
        x = self.dC.Xtrain[:, 0]*u.deg
        y = self.dC.Xtrain[:, 1]*u.deg
        dx = self.dC.Ytrain[:, 0]*u.mas
        dy = self.dC.Ytrain[:, 1]*u.mas
        xiplus = GPRutils.calcCorrelation2D(
            x, y, dx, dy, rmax=rmax, nBins=nBins)[0]

        # Be sure no NaNs get into the array.
        xiplus = np.where(np.isnan(xiplus), 0, xiplus)

        # Generate the uniform grid that the von Karman xiplus will be
        # calculated on.
        dx = (rmax / (nBins / 2)).to(u.deg).value
        x = np.arange(-nBins / 2, nBins / 2) * dx
        xx, yy = np.meshgrid(x, x)

        # This is the default initial guess. It seems to do fine.
        if v0 is None:
            v0 = np.array([xiplus.max(), 1, 0.1, 0.05, 0.05])

        # Form the initial simplex for the Nelder-Mead optimizer. If there are
        # 5 parameters then the simplex will be of shape (6, 5) where
        # simplex0[0, :] will be v0, the initial guess. simplex0[1, :] will be
        # the same as  v0 except the first parameter will be 15% greater.
        # Similarly, simplex0[2, :] will be the same as v0 except the second
        # parameter will be 15% greater, and so on. This is the form of
        # simplex that Nelder-Mead likes.

        # If there are p parameters then this line generates a shape (p, p)
        # array where each row is v0.
        simplex0 = np.vstack([v0]*v0.shape[0])

        # This makes each diagonal element of the array to be 15% greater.
        simplex0 += np.diag(v0*0.15)

        # This adds v0 one more time to make an array of shape (p+1, p).
        simplex0 = np.vstack([v0, simplex0])

        # If printing or writing to self.paramFile, print the header
        # information.
        GPRutils.printParams(
            v0,
            header=True,
            FoMtype="RSS",
            file=self.paramFile,
            printing=self.printing
            )

        # Call the Nelder-Mead optimizer. xtol, ftol, and maxfun are chosen
        # based on expoerience as a compromise between speed and accuracy.
        # These parameters are probably not optimal.
        args = (xx, yy, xiplus)
        self.opt_result = opt.fmin(
            figureOfMerit_fitCorr,
            simplex0[0],
            args=args,
            xtol=2.5,
            ftol=.1,
            maxfun=150,
            full_output=True,
            retall=True,
            initial_simplex=simplex0
        )

        # Store these final set of parameters in the dataContainer object.
        self.dC.fitCorrParams = self.opt_result[0]

    def fit(self, params: np.ndarray) -> None:
        """
        Fit a GPR von Karman model giiven a set of kernel parameters.

        For a given set of kernel parameters, fit a von Karman GPR model. This
        function relies on data being sufficiently processed and formatted in
        a dataContainer object.

        Arguments
        ---------
        params : np.ndarray
            Kernel parameters. Typically there are five parameters: K
            variance, Outer Scale, Diameter, Wind X, and Wind Y. Order of the
            parameters matters.
        """
        # Set the params attribute for the dataContainer object. This is for
        # convenience so that after you fit a model you don't have to tell the
        # dataContainer object what the kernel parameters are manually.
        self.dC.params = params

        # Initialize a von Karman TurbulentLayer object. This is the object
        # that you can use to input a set of parameters and get out a
        # covariance function (the kernel function).
        self.ttt = vk.TurbulentLayer(
            variance=params[0],
            outerScale=params[1],
            diameter=params[2],
            wind=(params[3], params[4]))

        # Calculate the covariance function of the von Karman model on this
        # grid of points. This grid will represent all pairs of points within
        # the training set.
        du, dv = GPRutils.getGrid(self.dC.Xtrain, self.dC.Xtrain)
        C = self.ttt.getCuv(du, dv)

        # If you want to take advantage of the curl-free nature of the
        # turbulence field, then things wll be handled differently. This
        # method will be slower than the alternatively method. Specifically,
        # the inversion step (through cholesky decomposition) will be
        # theoretically 4x slower.
        if self.dC.curl:
            # Reshape the covariance matrix from (N, N, 2, 2) to (2N, 2N).
            n1, n2 = C.shape[0], C.shape[1]
            K = np.swapaxes(C, 1, 2).reshape(2*n1, 2*n2)

            # Calculate the White kernels.
            W_GAIA, W_DES = GPRutils.makeW(
                self.dC.Etrain_GAIA, self.dC.Etrain_DES,
                useRMS=self.dC.useRMS, curl=True)

            # # Perform cholesky decomposition.
            # L = np.linalg.cholesky(K + W_GAIA + W_DES)

            # # Calculate alpha, which is an intermediate step in finding the
            # # posterior predictive mean (fbar_s). This is useful because if
            # # you want to use the model to predict on multiple new values at
            # # different times then you only have to do this once as long as
            # # you save this value, alpha.
            # self.alpha = np.linalg.solve(L, GPRutils.flat(self.dC.Ytrain))
            # self.alpha = np.linalg.solve(L.T, self.alpha)
            
            # Try using cho_factor and cho_solve.
            cho_factor = scipy.linalg.cho_factor(K + W_GAIA + W_DES)
            self.alpha = scipy.linalg.cho_solve(cho_factor, GPRutils.flat(self.dC.Ytrain))

        # If you don't want to take advantage of the curl-free nature of the
        # turbulence field, then you will effectively have two separation GPR
        # models.
        else:
            # These are the two kernels for the two GPR models which predict,
            # separately, on each component x and y (or u and v).
            Ku = C[:, :, 0, 0]
            Kv = C[:, :, 1, 1]

            # Calculate the White kernels.
            Wu_GAIA, Wv_GAIA, Wu_DES, Wv_DES = GPRutils.makeW(
                self.dC.Etrain_GAIA, self.dC.Etrain_DES,
                useRMS=self.dC.useRMS)

            # # Perform the cholesky decomposition for both models.
            # Lu = np.linalg.cholesky(Ku + Wu_GAIA + Wu_DES)
            # Lv = np.linalg.cholesky(Kv + Wv_GAIA + Wv_DES)

            # # Calculate alpha for each model.
            # self.alpha_u = np.linalg.solve(Lu, self.dC.Ytrain[:, 0])
            # self.alpha_v = np.linalg.solve(Lv, self.dC.Ytrain[:, 1])

            # self.alpha_u = np.linalg.solve(Lu.T, self.alpha_u)
            # self.alpha_v = np.linalg.solve(Lv.T, self.alpha_v)
            
            # Try using cho_factor and cho_solve.
            cho_factor_u = scipy.linalg.cho_factor(Ku + Wu_GAIA + Wu_DES)
            self.alpha_u = scipy.linalg.cho_solve(cho_factor_u, self.dC.Ytrain[:, 0])
            
            cho_factor_v = scipy.linalg.cho_factor(Kv + Wv_GAIA + Wv_DES)
            self.alpha_v = scipy.linalg.cho_solve(cho_factor_v, self.dC.Ytrain[:, 1])

    def predict(self, X: np.ndarray) -> None:
        """
        Use the currently trained model to predict on a set of inputs.

        Using the current model (which is effectively just the alpha array),
        we can calculate the posterior predictive mean (fbar_s). fbar_s is
        essentially the models estimate of the atmospherc turbulence.

        Arguments
        ---------
        X : np.ndarray
            The (M, 2) array of input astrometric positions that you want to
            find the atmospheric turbulence for.
        """
        # Calculate the covariance function of the von Karman model on this
        # grid of points. This grid will represent all pairs of points between
        # the training set and the new inputs
        du, dv = GPRutils.getGrid(X, self.dC.Xtrain)
        Cs = self.ttt.getCuv(du, dv)

        # When taking advantage of the curl-free nature of the turbulence
        # field, as in self.fit, the calculation is a bit different.
        if self.dC.curl:
            # Reshape the covariance matrix from (M, N, 2, 2) to a (2M, 2N)
            # array. Note that I may haeve M and N backwards.
            n1, n2 = Cs.shape[0], Cs.shape[1]
            Ks = np.swapaxes(Cs, 1, 2).reshape(2*n1, 2*n2)

            # Calculate the posterior predictive mean.
            self.dC.fbar_s = GPRutils.unflat(np.dot(Ks.T, self.alpha))

        # When not taking advantage of the curl-free nature of the turbulence
        # field, things are a bit simpler.
        else:
            # These are the two kernels for the two GPR models which predict,
            # separately, on each component x and y (or u and v).
            Ksu = Cs[:, :, 0, 0]
            Ksv = Cs[:, :, 1, 1]

            # Calculate the posterior predictive mean.
            fbar_s_u = np.dot(Ksu.T, self.alpha_u)
            fbar_s_v = np.dot(Ksv.T, self.alpha_v)

            self.dC.fbar_s = np.vstack([fbar_s_u, fbar_s_v]).T

    def figureOfMerit(self, params: np.ndarray) -> float:
        """
        Given a set of kernel parameters, calculate the goodness of fit.

        This function calls self.fit on those kernel parameters, and then
        calls self.predict on the validation set. The model has not been
        trained on the validation set, only on the training set, so we predict
        on the validation set and calculate the reduction in small scale
        correlation and use this as our figure of merit.

        Arguments
        ---------
        params : np.ndarray
            Kernel parameters. Typically there are five parameters: K
            variance, Outer Scale, Diameter, Wind X, and Wind Y. Order of the
            parameters matters.

        Returns
        -------
        xiplus : float
            We use this value to measure how good the model is. A scipy
            optimizer uses this value to minimize.
        """
        # Fit the model and predict on the validation set.
        self.fit(params)
        self.predict(self.dC.Xvalid)

        # Calculate the goodness of fit using xi_0.02 as our figure of merit.
        xiplus, Uerr, Verr, pairs = GPRutils.getXi(
            self.dC.Xvalid, self.dC.Yvalid - self.dC.fbar_s,
            rMax=0.02*u.deg, rMin=5*u.mas)

        # Print these parameters to std.out or write them to self.paramFile.
        GPRutils.printParams(
            params,
            FoM=xiplus,
            file=self.paramFile,
            printing=self.printing
            )

        # Return the figure of merit. A scipy optimizer (Nelder-Mead)
        # optimizes this value.
        return np.abs(xiplus)

    def optimize(
        self,
        v0: np.ndarray = None,
        xtol=2.5,
        ftol=0.025,
        maxfun=150,
        func=None) -> None:
        """
        Call the Nelder-Mead optimizer routine to optimze the model.

        This method makes it easy to perform optimization of this modelbecause
        it condenses these few steps into a simple function call.

        Keyword Arguments
        ---------
        v0 : np.ndarray
            Define an initial guess of the kernel parameters for the optimizer.
        """
        # We use the output of self.fitCorr as the initial guess. This is the
        # whole purpose of the fitCorr routine. You can optionally only use
        # self.fitCorr and skip this optimization step. self.fitCorr is
        # probably more than 20 times faster than this method, however this
        # method does a much better job.
        if v0 is None:
            v0 = self.dC.fitCorrParams

        # Form the initial simplex for the Nelder-Mead optimizer. If there are
        # 5 parameters then the simplex will be of shape (6, 5) where
        # simplex0[0, :] will be v0, the initial guess. simplex0[1, :] will be
        # the same as  v0 except the first parameter will be 15% greater.
        # Similarly, simplex0[2, :] will be the same as v0 except the second
        # parameter will be 15% greater, and so on. This is the form of
        # simplex that Nelder-Mead likes.

        # If there are p parameters then this line generates a shape (p, p)
        # array where each row is v0.
        simplex0 = np.vstack([v0]*v0.shape[0])

        # This makes each diagonal element of the array to be 15% greater.
        simplex0 += np.diag(v0*0.15)

        # This adds v0 one more time to make an array of shape (p+1, p).
        simplex0 = np.vstack([v0, simplex0])

        # If printing or writing to self.paramFile, print the header
        # information.
        GPRutils.printParams(
            v0,
            header=True,
            FoMtype="xi +",
            file=self.paramFile,
            printing=self.printing
            )

        # Call the optimizer.
        # Call the Nelder-Mead optimizer. xtol, ftol, and maxfun are chosen
        # based on expoerience as a compromise between speed and accuracy.
        # These parameters are probably not optimal.
        if func is None:
            func = self.figureOfMerit
        self.opt_result_GP = opt.fmin(
            func,
            simplex0[0],
            xtol=2.5,
            ftol=0.025,
            maxfun=150,
            full_output=True,
            retall=True,
            initial_simplex=simplex0
        )
