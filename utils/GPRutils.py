# Standard modules
import os
import sys
import glob

# Willow Fox Fortino's modules
import DESutils
import plotGPR
import vonkarmanFT as vk

# Professor Gary Bernstein's modules
import getGaiaDR2 as gaia
import gbutil

# Science modules
import numpy as np
import astropy.units as u
import astropy.coordinates as co
import astropy.table as tb
import astropy.io.fits as fits
import astropy.stats as stats
from astropy.time import Time
from scipy.spatial.ckdtree import cKDTree

from IPython import embed


class dataContainer(object):
    """
    Contains methods that help facilitate Gaussian Process Regression.

    This class contains methods that help import and format DES and Gaia
    astrometric data in order to make the data readable for the
    vonKarman2KernelGPR class in vK2KGPR.py.
    """

    def __init__(self, randomState: int = 0) -> None:
        """
        Initialize a dataContainer object.

        Simply specifies the numpy random state variable so that all numpy or
        scipy random operatioins are repeatable.

        Keyword Arguments
        -----------------
        randomState : init
            Seed for the various numpy and scipy random operations.
        """
        self.randomState = randomState

    def summarize(self, plot: bool = True) -> None:
        """
        Summarize the data in this object.

        Method to be called after GPR analysis is completed and solutions are
        obtained and Jackknifed (k-fold cross validation, k=5). This method
        prinits relevant data such as the exposure number, the passband, and
        the GPR kernel parameters.

        The six plots are:
        1. Cuu, Cvv, Cuv, and Cuu + Cvv (the kernel functions) for the set of
        kernel parameters obtained after the fitCorr step in GPR analysis.
        2. The same as above but for the finial set of kernel parameters
        obtanied after the optitmization step.
        3. Plot of the residual field before and after the GPR model is
        applied.
        4. Plot of the divergence and curl of the residual field before and
        after the GPR model is applied.
        5. E and B mode s of the angle averaged 2pt correlation function of
        the residual field before and after the GPR model is applied.
        6. 2 dimensional 2pt correlation function. First panel is the observed
        residual field. Second panel is the GPR's estimate of the observed
        residual field. Third panel is the subtraction of the observed
        residual field from the GPR's estimate.
        
        Arugments
        ---------
        OUTfile : str
            Name of the .out file that was created along with the .fits file.

        Keyword Arguments
        -----------------
        plot : bool
            If False, no plots will be calculated. This makes the function run
            much faster if you only want to know the xi_0.02 and kernel
            parameters.
        """
#         out = DESutils.parseOutfile(OUTfile)
#         if not out.finished:
#             print(f"{OUTfile} not finished.")
#             return

        # Print a simply header.
        exp_band = f"{self.expNum} {self.band}"
        print(f"#####{exp_band:-^20}#####")
#         print(f"Total Time: {out.totalTime:.3f}")
#         print(f"Average GP Calculation Tmie: {out.avgGPTime:.3f}")
        
#         print("-"*30)
        # This is in a try-except block because some some FITS files won't
        # have this information because they are old. Eventually the
        # try-except block here won't be necessary.
        print("  Fitted von Kármán kernel parameters:")
        printParams(self.fitCorrParams, header=True, printing=True)
        printParams(self.fitCorrParams, printing=True)
        
#         print("-"*30)
#         print(f"  Correlation Fitting Time: {out.fitCorrTime:.3f}, {out.nfC} steps")

#         N = len(self.TV[self.TV["MaskCorrFit"]])
#         starDensity = (N / (3*u.deg**2)).to(u.arcmin**-2)
#         print(f"  Star Density: {starDensity:.3f}")

        xi0 = self.header["fC_xi0"]
        Xerr = self.header["fC_xi0_Xerr"]
        Yerr = self.header["fC_xi0_Yerr"]
        xi0err = np.sqrt(Xerr**2 + Yerr**2)
        print(f"    xi0: {xi0:.3f} ± {xi0err:.3f}")

        xif = self.header["fC_xif"]
        Xerr = self.header["fC_xif_Xerr"]
        Yerr = self.header["fC_xif_Yerr"]
        xiferr = np.sqrt(Xerr**2 + Yerr**2)
        print(f"    xif: {xif:.3f} ± {xiferr:.3f}")

        red = xi0/xif
        rederr = np.sqrt(((xi0err/xi0)**2 + (xiferr/xif)**2) * red**2)
        print(f"    Reduction: {red:.3f} ± {rederr:.3f}")
        
#         print("-"*30)
        print("  Final von Kármán kernel parameters:")
        printParams(self.params, header=True, printing=True)
        printParams(self.params, printing=True)
#         print("-"*30)
#         print(f"  Optmization Time: {out.optTime:.3f}, {out.nGP} ({out.nOpt1} {out.nOpt2}) steps")
        
#         N = len(self.TV[self.TV["MaskJackKnife"]])
#         starDensity = (N / (3*u.deg**2)).to(u.arcmin**-2)
#         print(f"  Star Density: {starDensity:.3f}")
        
        xi0 = self.header["xi0"]
        Xerr = self.header["xi0_Xerr"]
        Yerr = self.header["xi0_Yerr"]
        xi0err = np.sqrt(Xerr**2 + Yerr**2)
        print(f"    xi0: {xi0:.3f} ± {xi0err:.3f}")

        xif = self.header["xif"]
        Xerr = self.header["xif_Xerr"]
        Yerr = self.header["xif_Yerr"]
        xiferr = np.sqrt(Xerr**2 + Yerr**2)
        print(f"    xif: {xif:.3f} ± {xiferr:.3f}")

        red = xi0/xif
        rederr = np.sqrt(((xi0err/xi0)**2 + (xiferr/xif)**2) * red**2)
        print(f"    Reduction: {red:.3f} ± {rederr:.3f}")
        print()


        if not plot:
            return

        ttt = vk.TurbulentLayer(
            variance=self.fitCorrParams[0],
            outerScale=self.fitCorrParams[1],
            diameter=self.fitCorrParams[2],
            wind=(self.fitCorrParams[3], self.fitCorrParams[4]))
        vk.plotCuv(ttt)

        ttt = vk.TurbulentLayer(
            variance=self.params[0],
            outerScale=self.params[1],
            diameter=self.params[2],
            wind=(self.params[3], self.params[4]))
        vk.plotCuv(ttt)

        x = self.TV["X"][self.TV["Maskf"]]
        y = self.TV["Y"][self.TV["Maskf"]]
        dx = self.TV["dX"][self.TV["Maskf"]]
        dy = self.TV["dY"][self.TV["Maskf"]]
        if self.header["useRMS"]:
            RMSx2 = self.TV[self.TV["Maskf"]]["DES variance"][:, 0, 0]
            RMSy2 = self.TV[self.TV["Maskf"]]["DES variance"][:, 1, 1]
            err = np.sqrt(0.5 * (RMSx2 + RMSy2))
        else:
            err = np.sqrt(self.TV[self.TV["Maskf"]]["DES variance"])

        x2 = x
        y2 = y
        dx2 = dx - self.TV["fbar_s dX"][self.TV["Maskf"]]
        dy2 = dy - self.TV["fbar_s dY"][self.TV["Maskf"]]
        err2 = err

        plotGPR.AstrometricResiduals(
            x, y, dx, dy, err,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2, err2=err2,
            exposure=self.expNum,
            pixelsPerBin=450,
            scale=200*u.mas,
            arrowScale=10*u.mas)

        plotGPR.DivCurl(
            x, y, dx, dy, err,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2, err2=err2,
            exposure=self.expNum,
            pixelsPerBin=750,
            scale=50)

        # Can't use all pairs here. Need to only take one subset (Subset A)
        # because I can't figure out how to rewrite the following two plotting
        # functions (and the functions that calculate the angle-averaged and
        # 2d 2pt correlation functions) to not include pairs of points from
        # the same subset. We have concluded that it isn't proper to do this
        # when calculating the 2pt correlation function, so I only calculate
        # these two functions with only one subset of data. The plots are
        # noisier, however it doesn't matter so much because the real value
        # that we care about is xi_0.02 and that is easy to calculate and
        # excluding these pairs.
        x = self.TV["X"][self.TV["Maskf"] & self.TV["Subset A"]]
        y = self.TV["Y"][self.TV["Maskf"] & self.TV["Subset A"]]
        dx = self.TV["dX"][self.TV["Maskf"] & self.TV["Subset A"]]
        dy = self.TV["dY"][self.TV["Maskf"] & self.TV["Subset A"]]
        if self.header["useRMS"]:
            RMSx2 = self.TV[self.TV["Maskf"] &
                            self.TV["Subset A"]]["DES variance"][:, 0, 0]
            RMSy2 = self.TV[self.TV["Maskf"] &
                            self.TV["Subset A"]]["DES variance"][:, 1, 1]
            err = np.sqrt(0.5 * (RMSx2 + RMSy2))
        else:
            err = np.sqrt(self.TV[self.TV["Maskf"] &
                                  self.TV["Subset A"]]["DES variance"])

        x2 = x
        y2 = y
        dx2 = dx - self.TV["fbar_s dX"][self.TV["Maskf"] & self.TV["Subset A"]]
        dy2 = dy - self.TV["fbar_s dY"][self.TV["Maskf"] & self.TV["Subset A"]]
        err2 = err

        plotGPR.Correlation(
            x, y, dx, dy,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2,
            xiplus_ON=True,
            exposure=self.expNum,
            ylim=(None, None))

        plotGPR.Correlation2D(
            x, y, dx, dy,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2,
            exposure=self.expNum,
            nBins=100,
            vmin=-100*u.mas**2,
            vmax=100*u.mas**2,
            rmax=0.50*u.deg)

    @u.quantity_input
    def load(
        self,
        expNum: int,
        earthRef: str = "/home/fortino/DESworkspace/data/"
                        "y6a1.exposures.positions.fits.gz",
        tol: u.arcsec = 0.5*u.arcsec,
        nSigma: int = 4,
        vSet: str = "Subset A",
        maxDESErr: u.mas**2 = np.inf*u.mas**2,
        minDESErr: u.mas**2 = -np.inf*u.mas**2,
        downselect: float = 1.0,
        useRMS: bool = False,
        returnObs: bool = False
            ):
        """
        This function formats DES and Gaia astrometry data.

        Arguments
        ---------
            expNum : int
                DES exposure number.

        Keyword Arguments
        -----------------
            earthRef : str
                FITS file  that relates DES exposure number to the MJD of the
                exposure.
            tol : Quantity object (angle)
                Tolerance for matching DES objects and Gaia objects.
            nSigma : int
                How many standard deviations to sigma clip to.
            vSet : str
                Must be one of "Subset A", "Subset B", "Subset C", "Subset D",
                or "Subset E". Specifies which subset to use as the first
                validation set in the jackknifing (k-fold cross validation)
                routine.
            maxDESErr : Quantity object (solid angle)
                Maximum DES error to allow in the dataset.
            minDESErr : Quantity object (sold angle)
                Minimum DES error to allow in the dataset.
            downselect : float
                Fraction of the dataset (after sigma clipping and maxDESErr or
                minDESErr have been appled) to keep. The rest will also be
                masked out with Mask0.
            useRMS : bool
                Whether or not to replace each object's DES variance value
                with an empirically calculated 2x2 diagonal covariance matrix.
                Use this if you do not trust the given DES estimates for the
                variance.
            returnObs : bool
                If True, returns the time of exposure and does not complete
                the full method.
        """
        self.expNum = expNum
        self.earthRef = earthRef
        self.tol = tol.to(u.arcsec)
        self.nSigma = nSigma
        self.vSet = vSet
        self.maxDESErr = maxDESErr.to(u.mas**2)
        self.minDESErr = minDESErr.to(u.mas**2)
        self.downselect = downselect
        self.useRMS = useRMS

        # Use earthRef to find the center (ra, dec) of the exposure as well as
        # the MJD of the exposure.
        pos_tab = tb.Table.read(earthRef, hdu=1)
        pos_tab = pos_tab[pos_tab["expnum"] == self.expNum]
        ra0 = np.array(pos_tab["ra"])[0]*u.deg
        dec0 = np.array(pos_tab["dec"])[0]*u.deg
        DES_obs = Time(np.array(pos_tab["mjd_mid"])[0], format="mjd")

        # Return time of observation. Likely for plotting purposes.
        if returnObs:
            return DES_obs

        # Grab filenames for each tile for this exposure.
        tilefiles = DESutils.findTiles(self.expNum, confirmTiles=True)

        # Create an empty astropy table with all of the necessary columns.
        file0 = tilefiles[0]
        tab0 = tb.Table.read(file0)
        DES_tab = tab0.copy()
        DES_tab.remove_rows(np.arange(len(DES_tab)))
        del tab0

        # Loop through each tile, open the table, and append (with tb.vstack)
        # the data to our empty table.
        for file in tilefiles:
            tab = tb.Table.read(file)
            tab = tab[tab["EXPNUM"] == self.expNum]
            DES_tab = tb.vstack([DES_tab, tab])

        # For future reference, save the passband of the exposure.
        self.band = np.unique(DES_tab["BAND"])[0]

        # Print a short summary of the DES data.
        print(f"DES Exposure : {self.expNum}")
        print(f"DES Passband : {self.band}")
        print(f"DES nSources : {len(DES_tab)}")

        # Initialize variables for the relevant columns.
        DES_ra = np.array(DES_tab["NEW_RA"])*u.deg
        DES_dec = np.array(DES_tab["NEW_DEC"])*u.deg
        DES_err = ((np.array(DES_tab["ERRAWIN_WORLD"])*u.deg).to(u.mas))**2

        # Retrieve Gaia data and initialize variables for the relevant
        # columns.
        GAIA_tab = gaia.getGaiaCat(ra0.value, dec0.value, 2.5, 2.5)
        GAIA_obs = Time("J2015.5", format="jyear_str", scale="tcb")

        # Adjust Gaia RA values to be between -180 and 180
        GAIA_ra = np.array(GAIA_tab["ra"])*u.deg
        GAIA_ra[GAIA_ra > 180*u.deg] -= 360*u.deg

        GAIA_dec = np.array(GAIA_tab["dec"])*u.deg
        GAIA_pmra_cosdec = np.array(GAIA_tab["pmra"])*u.mas/u.yr
        GAIA_pmdec = np.array(GAIA_tab["pmdec"])*u.mas/u.yr
        GAIA_parallax = np.array(GAIA_tab["parallax"])*u.mas

        # Full 5-parameter covariance matrix
        GAIA_cov = np.array(GAIA_tab["cov"])*u.mas**2
        GAIA_cov = np.reshape(GAIA_cov, (GAIA_cov.shape[0], 5, 5))

        # Initialize astropy SkyCoord objects to take advantage of astropy's
        # `match_coordinates_sky` routine.
        X_DES = co.SkyCoord(DES_ra, DES_dec)
        X_GAIA = co.SkyCoord(GAIA_ra, GAIA_dec)

        # Match DES objects with Gaia counterparts based on how close together
        # they are on the sky.
        idx, sep2d, dist3d = co.match_coordinates_sky(X_GAIA, X_DES)

        # This slice can index the Gaia catalog for only the stars that have a
        # DES match within tol.
        self.ind_GAIA = np.where(sep2d < tol)[0]

        # This slice can index the DES catalog for only the stars that have a
        # Gaia match within tol. Will be in the same order as ind_GAIA
        self.ind_DES = idx[self.ind_GAIA]

        print(f"There were {self.ind_GAIA.size} matches within {tol}.")

        # Transformation matrix from ICRS to gnomonic projection about
        # (ra0, dec0).
        M = np.array([
          [-np.sin(ra0),               np.cos(ra0), 0],
          [-np.cos(ra0)*np.sin(dec0), -np.sin(ra0)*np.sin(dec0), np.cos(dec0)],
          [np.cos(ra0)*np.cos(dec0),   np.sin(ra0)*np.cos(dec0), np.sin(dec0)]
        ])

        # Compute ICRS coordinates of DES catalog
        X_ICRS_DES = np.array([
            np.cos(DES_dec) * np.cos(DES_ra),
            np.cos(DES_dec) * np.sin(DES_ra),
            np.sin(DES_dec)
        ])

        # Compute ICRS coordinates of Gaia catalog.
        X_ICRS_GAIA = np.array([
            np.cos(GAIA_dec) * np.cos(GAIA_ra),
            np.cos(GAIA_dec) * np.sin(GAIA_ra),
            np.sin(GAIA_dec)
        ])

        # Perform gnomonic projection on DES catalog.
        xproj, yproj, zproj = np.dot(M, X_ICRS_DES)
        X_gn_DES = ((xproj/zproj, yproj/zproj)*u.rad).to(u.deg).T

        # Perform gnomonic projection on Gaia catalog.
        xproj, yproj, zproj = np.dot(M, X_ICRS_GAIA)
        X_gn_GAIA = ((xproj/zproj, yproj/zproj)*u.rad).to(u.deg).T

        # Calculate gnomonic projection of the observatory coordinates
        X_E = np.array(pos_tab["observatory"])[0]
        X_gn_E = np.dot(M, X_E)

        # Calculate time difference between DES and Gaia observations.
        dt = DES_obs - GAIA_obs
        dt = dt.sec
        dt = (dt*u.s).to(u.yr).value

        # Calculate coefficient matrix in appropriate units.
        A = np.array([
            [1, 0, -X_gn_E[0], dt, 0],
            [0, 1, -X_gn_E[1], 0, dt]
        ])

        # Calculate the variable array in appropriate units.
        X = np.vstack([
            X_gn_GAIA[:, 0].value,
            X_gn_GAIA[:, 1].value,
            GAIA_parallax.to(u.deg).value,
            GAIA_pmra_cosdec.to(u.deg/u.yr).value,
            GAIA_pmdec.to(u.deg/u.yr).value
        ])

        # Perform epoch transformation
        X_gn_transf_GAIA = np.dot(A, X).T*u.deg

        # Find covariance matrix for X_gn_transf_GAIA.
        cov = np.dot(A, np.dot(GAIA_cov, A.T))

        # Swap axes to make the shape of the array more natural to humans.
        cov = np.swapaxes(cov, 1, 0)

        # Declare attribute for the full X array (DES star positions).
        # This array will be split into the training, validation,
        # and prediction sets.
        self.X = X_gn_DES

        # This line performs the subtraction between Gaia stars (indexed by
        # ind_GAIA which takes only the Gaia stars that have DES counterparts)
        # and DES stars (indexed by ind_DES which takes only the DES stars
        # that have DES counterparts). This creates the astrometric residuals
        # between the "true" star positions (Gaia) and the "observed" star
        # positions (DES).
        self.Y = X_gn_transf_GAIA[self.ind_GAIA] - X_gn_DES[self.ind_DES]

        # This is an array of (2, 2) covariance matrices. This line selects
        # only the covariance matrices for Gaia stars that have a DES
        # counterpart.
        self.E_GAIA = cov[self.ind_GAIA, :, :]

        # Declare attribute for ERRAWIN_WORLD (see SExtractor) measurement
        # error for DES stars. There is not a (2, 2) covariance matrix for
        # each detection. The useRMS kwarg can be used to calculate an
        # empirical (diagonal) 2x2 covariance matrix to replace these values.
        self.E_DES = DES_err

        # Load the TV set (Training + Validation) into an astropy table.
        x = tb.Column(data=self.X[self.ind_DES][:, 0], name=f"X")
        y = tb.Column(data=self.X[self.ind_DES][:, 1], name=f"Y")
        dx = tb.Column(data=self.Y[:, 0].to(u.mas), name=f"dX")
        dy = tb.Column(data=self.Y[:, 1].to(u.mas), name=f"dY")
        err_DES = tb.Column(data=self.E_DES[self.ind_DES],
                            name=f"DES variance")
        err_GAIA = tb.Column(data=self.E_GAIA,
                             name=f"GAIA covariance")
        self.TV = tb.QTable([x, y, dx, dy, err_DES, err_GAIA])

        # Sigma clip on the TV residuals. This usually removes about 100
        # objects.
        Y = np.vstack([self.TV["dX"], self.TV["dY"]]).T.value
        mask = stats.sigma_clip(Y, sigma=nSigma, axis=0).mask
        mask = tb.Column(
            data=~np.logical_or(*mask.T),
            name="Mask0",
            description="False if excluded in initial sigma clipping.")
        self.TV.add_column(mask)

        # Remove stars that have less variance than minDESErr and stars
        # that have more variance than maxDESErr. Fold this into Mask0 for
        # simplicity.
        minMask = self.TV["DES variance"] > self.minDESErr
        maxMask = self.TV["DES variance"] < self.maxDESErr
        self.TV["Mask0"] = self.TV["Mask0"] & minMask & maxMask

        if self.useRMS:
            x = self.TV[self.TV["Mask0"]]["X"].value
            y = self.TV[self.TV["Mask0"]]["Y"].value
            dx = self.TV[self.TV["Mask0"]]["dX"].value
            dy = self.TV[self.TV["Mask0"]]["dY"].value
            err = self.TV[self.TV["Mask0"]]["DES variance"].value

            # Get an array of indices to index the entire table.
            table_inds = np.arange(len(self.TV))

            # Sort the table by increasing DES variance.
            sort = np.argsort(err)

            # Get the indices from table_inds that only include the stars kept
            # by Mask0, and then sort them.
            sort_inds = table_inds[self.TV["Mask0"]][sort]

            # Get the sorted residual field
            resid_x = dx[sort]
            resid_y = dy[sort]

            # Split arrays into groups of nStars. This is an arbitrary choice.
            # 256 seems to work well for a catalog of stars with about 1 star
            # per arcmin^2 though.
            nStars = 256

            # Split the residual field arrays and the indices array into
            # sub-arrays of size approximately equal to nStars. See
            # documentation for np.array_split for exactly how the split is
            # made. Note that each sub-array will not be length (nStars,)
            # exactly.
            sort_inds = np.array_split(sort_inds, len(sort_inds)//nStars)
            resid_x = np.array_split(resid_x, len(dx)//nStars)
            resid_y = np.array_split(resid_y, len(dy)//nStars)

            # Find RMS of each sub-array.
            RMSx = np.array([np.std(arr) for arr in resid_x])
            RMSy = np.array([np.std(arr) for arr in resid_y])

            # Replace the DES variance column with a diagonal 2x2 covariance
            # matrix with RMSx^2 and RMSy^2 on the diagonals. For each of the
            # sub-arrays, an RMSx and RMSy value is calculated. For each star
            # in these sub-arrays, the DES variance is placed with these RMS
            # values. If the original DES variance estimates are wrong, this
            # should help things.
            DEScov = np.zeros((len(self.TV), 2, 2))
            for i, (ind, rmsx, rmsy) in enumerate(zip(sort_inds, RMSx, RMSy)):
                DEScov[ind, 0, 0] = rmsx**2
                DEScov[ind, 1, 1] = rmsy**2
            DESvar = tb.Column(data=DEScov, name="DES variance", unit=u.mas**2)
            self.TV["DES variance"] = DESvar

        # Define some placeholder arrays for removing a polynomial fit.
        x = self.TV["X"][self.TV["Mask0"]]
        y = self.TV["Y"][self.TV["Mask0"]]
        dx = self.TV["dX"][self.TV["Mask0"]]
        dy = self.TV["dY"][self.TV["Mask0"]]

        # Remove a 3rd order polynomial fit from the residuals.
        poly = Poly2d(3)
        poly.fit(x, y, dx)
        dxfit = poly.evaluate(x, y) * self.TV["dX"].unit
        self.TV["dX"][self.TV["Mask0"]] -= dxfit
        poly.fit(x, y, dy)
        dyfit = poly.evaluate(x, y)*self.TV["dY"].unit
        self.TV["dY"][self.TV["Mask0"]] -= dyfit

        # Get some useful values for performing the downselect.
        n = len(self.TV)
        n0 = len(self.TV[self.TV["Mask0"]])
        nTrue = int(np.floor(n0 * self.downselect))

        # Find the indices (relative to the entire table) of the rows
        # where Mask0 = True. Then take a random fraction of those
        # (specified by kwarg downselect) that will be true.
        inds = np.arange(n)
        inds_mask0 = inds[self.TV["Mask0"]]
        rng = np.random.RandomState(self.randomState)
        rng.shuffle(inds_mask0)
        inds_mask0_True = inds_mask0[:nTrue]

        # Create a False array of the length of the entire table. Use the
        # above indices to specify which of the rows (that already have
        # Mask0 = True) will be True in this new mask.
        downselectMask = np.zeros(n, dtype=bool)
        downselectMask[inds_mask0_True] = True

        # Fold this new downselect mask into Mask0 for simplicity.
        self.TV["Mask0"] = self.TV["Mask0"] & downselectMask

        # Make an index array for the entire TV set that will be used for
        # performing the k-fold cross validation (jackknifing).
        nTV = len(self.TV)
        tv_ind = np.arange(nTV)

        # Shuffle the index array.
        rng.shuffle(tv_ind)

        # Split the index array into 5 approximately equal arrays.
        index_arrays = np.array_split(tv_ind, 5)

        # Create a boolean placeholder array. This is used to turn the index
        # arrays into boolean masks of length nTV.
        self.arr = np.zeros_like(tv_ind).astype(bool)

        # The names of the subsets.
        letters = ["A", "B", "C", "D", "E"]

        # Create the boolean submasks.
        for letter, index_arr in zip(letters, index_arrays):
            mask = tb.Column(data=self.arr.copy(), name=f"Subset {letter}")
            mask[index_arr] = True
            mask = np.logical_and(mask, self.TV["Mask0"])
            self.TV.add_column(mask)

        # Create the prediction dataset table that represents all DES objects
        # that don't have a Gaia counterpart.
        Xpred = np.delete(self.X, self.ind_DES, axis=0)
        Epred = np.delete(self.E_DES, self.ind_DES, axis=0)

        x = tb.Column(data=Xpred[:, 0], name="X")
        y = tb.Column(data=Xpred[:, 1], name="Y")
        e = tb.Column(data=Epred, name="DES variance")

        self.Pred = tb.QTable([x, y, e])

        # Make the training and validation sets into attributes of this object
        # like how vK2KGPR.py expects it.
        Train, Valid = makeSplit(self.TV, self.vSet)
        self.makeArrays(Train, Valid)

        # Make placeholder columns for fbar_s_x and fbar_s_y.
        fbar_s_x = tb.Column(
            data=self.arr.copy(),
            name=f"fbar_s dX",
            dtype=float, unit=u.mas,
            description="posterior predictive mean (dx)")
        fbar_s_y = tb.Column(
            data=self.arr.copy(),
            name=f"fbar_s dY",
            dtype=float, unit=u.mas,
            description="posterior predictive mean (dy)")
        self.TV.add_columns([fbar_s_x, fbar_s_y])

        # Make placeholder columns for fbar_s_x and fbar_s_y after the fitCorr
        # stage.
        fbar_s_x = tb.Column(
            data=self.arr.copy(),
            name=f"fbar_s dX fC",
            dtype=float, unit=u.mas,
            description="posterior predictive mean (dx) for fitCorr params")
        fbar_s_y = tb.Column(
            data=self.arr.copy(),
            name=f"fbar_s dY fC",
            dtype=float, unit=u.mas,
            description="posterior predictive mean (dy) for fitCorr params")
        self.TV.add_columns([fbar_s_x, fbar_s_y])

        # Create a placefolder column for the mask that will be generated from
        # this sigma clipping.
        maskX = tb.Column(data=self.arr.copy(), name=f"MaskCorrFit")
        self.TV.add_column(maskX)

        # Initialize the column that will hold the new mask.
        maskjk = tb.Column(data=self.arr.copy(), name="MaskJackKnife")
        self.TV.add_column(maskjk)

        # Generate these values which may be useful in the future.
        self.nData = len(self.TV) + len(self.Pred)
        self.nPred = len(self.Pred)
        self.nTrain = len(Train)
        self.nValid = len(Valid)

    def makeArrays(
        self,
        Train: tb.table.Table,
        Valid: tb.table.Table
            ) -> None:
        """
        Method to make validation and training set arrays.

        Uses astropy tables from the GPRutils.makeSplit function to generate
        the validation and training set arrays that vK2KGPR.py will be used
        to. This method will extract the data from the tables into arrays and
        make sure that the astropy units are set correctly. However, the
        arrays will be numpy arrays, not Quantity objects. Not using units
        greatly simplifies vK2KGPR.py

        Arguments
        ---------
        Train : tb.table.Table
            Table with the training set. See GPRutils.makeSplit.
        Valid : tb.table.Table
            Table with the validation set. See GPRutils.makeSplit.
        """
        xt = Train["X"].to(u.deg).value
        yt = Train["Y"].to(u.deg).value
        dxt = Train["dX"].to(u.mas).value
        dyt = Train["dY"].to(u.mas).value
        self.Xtrain = np.vstack([xt, yt]).T
        self.Ytrain = np.vstack([dxt, dyt]).T
        self.Etrain_GAIA = Train["GAIA covariance"].to(u.mas**2).value
        self.Etrain_DES = Train["DES variance"].to(u.mas**2).value

        xv = Valid["X"].to(u.deg).value
        yv = Valid["Y"].to(u.deg).value
        dxv = Valid["dX"].to(u.mas).value
        dyv = Valid["dY"].to(u.mas).value
        self.Xvalid = np.vstack([xv, yv]).T
        self.Yvalid = np.vstack([dxv, dyv]).T
        self.Evalid_GAIA = Valid["GAIA covariance"].to(u.mas**2).value
        self.Evalid_DES = Valid["DES variance"].to(u.mas**2).value

    def JackKnife(
        self,
        GP: object,
        params: np.ndarray,
        fC: bool = False,
        nomask: bool = False
            ) -> None:
        """
        Perform k-fold cross-validation.

        Jackknife (or perform k-fold cross-validation) on the GPR model by
        rotating each of the 5 subsets as the validation set.

        Arguments
        ---------
        GP : vK2KGPR object
            vK2KGPR object containing fit and predict methods that are
            necessary for doing this sigma clipping.
        params : np.ndarray
            Kernel parameters that make the covariance function.

        Keyword Arguments
        -----------------
        fC : bool
            Whether or not this method is being called in the context of
            coming directly after GP.fitCorr.
        nomask : bool
            If True, this method will not create any additional masks
            MaskCorrFit, MaskJackKnife, or Maskf). Useful when wanting to
            jackknife the dataset for a new set of parameters, but the the
            TV table is already complete.
        """

        # Loop through each subset.
        subsets = ["Subset A", "Subset B", "Subset C", "Subset D", "Subset E"]
        for i in range(5):
            if nomask:
                # Generating training and validation set arrays for this
                # particular vSet (validation set) in this iteration of the
                # loop. Maskf will already have been created since this method
                # is really used reloading the data and trying to fit the same
                # data with new parameters
                Train, Valid = makeSplit(self.TV[self.TV["Maskf"]], subsets[i])
                self.makeArrays(Train, Valid)

                # Fit the GPR model to the validation set and predict to get
                # fbar_s values for this validation set.
                GP.fit(params)
                GP.predict(self.Xvalid)

                # Load these fbar_s values into the fbar_s columns in the TV
                # table. The proper masks to access the relevant rows are
                # the Maskf mask and the current subset mask.
                index = self.TV[subsets[i]] & self.TV["Maskf"]
                self.TV["fbar_s dX"][index] = self.fbar_s[:, 0]*u.mas
                self.TV["fbar_s dY"][index] = self.fbar_s[:, 1]*u.mas

            elif fC:
                # Generating training and validation set arrays for this
                # particular vSet (validation set) in this iteration of the
                # loop. If fC is True, then the only masks that will have been
                # created yet are the subset masks (which include Mask0).
                Train, Valid = makeSplit(self.TV, subsets[i])
                self.makeArrays(Train, Valid)

                # Fit the GPR model to the validation set and predict to get
                # fbar_s values for this validation set.
                GP.fit(params)
                GP.predict(self.Xvalid)

                # Load these fbar_s values into the fbar_s columns in the TV
                # table. The proper mask to access the relevant rows is only
                # the current subset mask
                index = self.TV[subsets[i]]
                self.TV["fbar_s dX fC"][index] = self.fbar_s[:, 0]*u.mas
                self.TV["fbar_s dY fC"][index] = self.fbar_s[:, 1]*u.mas

                # If fC is True, then we will want to create the MaskCorrFit
                # mask.
                mask = stats.sigma_clip(
                    self.fbar_s, sigma=self.nSigma, axis=0).mask
                mask = ~np.logical_or(*mask.T)
                self.TV["MaskCorrFit"][index] = mask

            else:
                # Generating training and validation set arrays for this
                # particular vSet (validation set) in this iteration of the
                # loop. If fC and nomask are False, then the only mask that
                # has been created so far will be MaskCorrFit, so that is what
                # we use here.
                Train, Valid = makeSplit(
                    self.TV[self.TV["MaskCorrFit"]], subsets[i])
                self.makeArrays(Train, Valid)

                # Fit the GPR model to the validation set and predict to get
                # fbar_s values for this validation set.
                GP.fit(params)
                GP.predict(self.Xvalid)

                # Load these fbar_s values into the fbar_s columns in the TV
                # table. The proper masks to access the relevant rows are the
                # MaskCorrFit mask and the current subset mask.
                index = self.TV[subsets[i]] & self.TV["MaskCorrFit"]
                self.TV["fbar_s dX"][index] = self.fbar_s[:, 0]*u.mas
                self.TV["fbar_s dY"][index] = self.fbar_s[:, 1]*u.mas

                # If fC and nomask are False, then we will want to create the
                # MaskJackKnife mask.
                mask = stats.sigma_clip(
                    self.fbar_s, sigma=self.nSigma, axis=0).mask
                mask = ~np.logical_or(*mask.T)
                self.TV["MaskJackKnife"][index] = mask
                
        if (not fC) and (not nomask):
            # Additionally, if fC and nomask are False, then we are at the
            # end of optimization, so we want to create the Maskf mask.
            # This mask will be able to index all of the points that have
            # survived the three rounds of sigma clipping (from Mask0,
            # MaskCorrFit, and MaskJackKnife).
            subsetMasks = \
                self.TV["Subset A"] + \
                self.TV["Subset B"] + \
                self.TV["Subset C"] + \
                self.TV["Subset D"] + \
                self.TV["Subset E"]
            sigmaclipMasks = \
                self.TV["MaskJackKnife"] & \
                self.TV["MaskCorrFit"]
            maskf = subsetMasks & sigmaclipMasks
            maskf = tb.Column(data=maskf, name="Maskf")
            self.TV.add_column(maskf)

    @u.quantity_input
    def JackKnifeXi(
        self,
        fC: bool = False,
        rMax: u.deg = 0.02*u.deg
            ):
        """
        The angle averaged 2pt correlation function of the jackknifed data.

        This method specifically calculates xi_0.02, the angle-averaged 2pt
        correlation function for pairs of points with separation less than
        0.02 deg. This is a reasonable approximation (although it will be a
        underestimate) of the 2pt correlation function at zero separation
        ("zero lag"). This method calculates xi_0.02 and its errors for the
        raw data and the data after the GPR model has been subtracted.

        This method also returns errors in x and y (or u and v, depending on
        which convention you are using) for xi_0.02. These errors reduce with
        the number of pairs, so including as many pairs as possible is
        important to getting a good idea of what xi_0.02 actually is. This is
        why we do the jackknifing as it gets us 5 times more pairs of points
        than if we didn't do it

        You might think that if each subset is ~1/5th of the total data set,
        then jackknifing gets us GPR solution values (fbar_s values) for 5
        times as many stars and therefore there are 25 times as many pairs of
        stars. However, this function is careful not to include pairs of
        points between one set and another set (inter-set pairs) and only
        includes pairs of points within one set (intra-set pairs). Through
        experimentation we found that it is not correct to include these
        initer-set pairs; one star from Subset A being in a pair with a star
        from Subset B is improper because they have triained off of each other.

        Keyword Arguments
        -----------------
        fC : bool
            Whether or not this method is being called in the context of
            coming directly after GP.fitCorr.
        rMax : Quantity object (angle)
            Maximum separation of sources to consider for calculating the
            correlation function.

        Returns
        -------
            xi : tuple
                xi_0.02 value and x/y errors for the raw data.
            xi2 : tuple
                xi_0.02 value and x/y errors for the GPR subtracted data.
        """

        # Initialize lists for holding all of the pairs of points.
        # Variables with the '2' suffix will be refering to the GPR subtracted
        # data, whereas the other variables with no suffix will be referencing
        # the raw data.
        prs_list = []
        prs_list2 = []

        # Looop through each subset
        subsets = ["Subset A", "Subset B", "Subset C", "Subset D", "Subset E"]
        for subset in subsets:
            # Get the data for only that subset
            data = self.TV[self.TV[subset] & self.TV["Maskf"]]

            # Form arrays of shape (N, 2) which are convenient for the
            # GPRutils.getXi function.
            X = np.vstack([data["X"], data["Y"]]).T
            Y = np.vstack([data["dX"], data["dY"]]).T
            Y2 = Y.copy()

            # If fC is True, take the data that was jackknifed with the
            # fitCorr params. Else, take the data that was jacknifed with the
            # normal params.
            if fC:
                Y2 -= np.vstack([data["fbar_s dX fC"], data["fbar_s dY fC"]]).T
            else:
                Y2 -= np.vstack([data["fbar_s dX"], data["fbar_s dY"]]).T

            # Call getXi to find the pairs of points with separation less than
            # 0.02 deg
            xi, Uerr, Verr, prs = getXi(X, Y, rMax=rMax)
            xi2, Uerr2, Verr2, prs2 = getXi(X, Y2, rMax=rMax)

            prs_list.append(Y[prs])
            prs_list2.append(Y2[prs2])

        # Combine all pairs together into one list rather than a list of lists.
        prs = np.vstack(prs_list)
        prs2 = np.vstack(prs_list2)

        # Calculate the correlation function.
        xiplus = np.mean(np.sum(prs[:, 0] * prs[:, 1], axis=1))
        xiplus2 = np.mean(np.sum(prs2[:, 0] * prs2[:, 1], axis=1))

        # Find the errors in these values.
        err = np.std(prs[:, 0] * prs[:, 1], axis=0) / np.sqrt(prs.shape[0])
        err2 = np.std(prs2[:, 0] * prs2[:, 1], axis=0) / np.sqrt(prs2.shape[0])

        # Package up the results nicely.
        xi = (xiplus, err[0], err[1])
        xi2 = (xiplus2, err2[0], err2[1])
        
        return xi, xi2

    def saveFITS(self, savePath: str, overwrite: bool = True) -> None:
        """
        Saves data to a FITS file.

        Saves the TV table into an astropy BinTableHDU. Saves the prediction
        table into an astropy BinTableHDU. Saves relevant metadata like
        exposure number, band, etc., in the header of the FITS file.
        Additionally save the kernel paramaters (after the fitCorr step and
        after the optimization step), as well as the xi_0.02 value before and
        after GPR subtraction from the raw residual field.

        Arguments
        ---------
        savePath : str
            The path that the FITS file will be written to.
        overwrite : bool
            Whether or not to allow overwriting of files if the same FITS file
            already exists.
        """
        # Header information
        # Add the various kwargs from this class into the header.
        hdr = fits.Header()
        hdr["expNum"] = self.expNum
        hdr["band"] = self.band
        hdr["earthRef"] = self.earthRef
        hdr["tol"] = self.tol.value
        hdr["nSigma"] = self.nSigma
        hdr["vSet"] = self.vSet
        hdr["randomState"] = self.randomState
        if np.abs(self.maxDESErr) == np.inf:
            hdr["maxDESErr"] = 0
        else:
            hdr["maxDESErr"] = self.maxDESErr.value
        if np.abs(self.minDESErr) == np.inf:
            hdr["minDESErr"] = 0
        else:
            hdr["minDESErr"] = self.minDESErr.value
        hdr["downselect"] = self.downselect
        hdr["useRMS"] = self.useRMS
        hdr["curl"] = self.curl

        # Add the jacknknifed xi_0.02 information to the header.
        xi, xi2 = self.JackKnifeXi()
        hdr["xi0"] = xi[0].value
        hdr["xi0_Xerr"] = xi[1].value
        hdr["xi0_Yerr"] = xi[2].value
        hdr["xif"] = xi2[0].value
        hdr["xif_Xerr"] = xi2[1].value
        hdr["xif_Yerr"] = xi2[2].value

        # Add the jacknknifed xi_0.02 information (from fitCorr) to the header.
        xi, xi2 = self.JackKnifeXi(fC=True)
        hdr["fC_xi0"] = xi[0].value
        hdr["fC_xi0_Xerr"] = xi[1].value
        hdr["fC_xi0_Yerr"] = xi[2].value
        hdr["fC_xif"] = xi2[0].value
        hdr["fC_xif_Xerr"] = xi2[1].value
        hdr["fC_xif_Yerr"] = xi2[2].value

        # Add the final kernel parameters to the header.
        hdr["var"] = self.params[0]
        hdr["outerScale"] = self.params[1]
        hdr["diameter"] = self.params[2]
        hdr["wind_x"] = self.params[3]
        hdr["wind_y"] = self.params[4]

        # Add the kernel parmaters (from fitCorr) to the header.
        hdr["fC_var"] = self.fitCorrParams[0]
        hdr["fC_outerScale"] = self.fitCorrParams[1]
        hdr["fC_diameter"] = self.fitCorrParams[2]
        hdr["fC_wind_x"] = self.fitCorrParams[3]
        hdr["fC_wind_y"] = self.fitCorrParams[4]

        # Initialize the HDU objects and the HDUList.
        prim_HDU = fits.PrimaryHDU(header=hdr)
        TV_HDU = fits.BinTableHDU(data=self.TV)
        Pred_HDU = fits.BinTableHDU(data=self.Pred)
        hdul = fits.HDUList([prim_HDU, TV_HDU, Pred_HDU])

        # Sort out the filename
        filename = ["GPR", str(self.expNum)]
        ext = self.band
        if self.useRMS:
            ext += "R"
        if self.curl:
            ext += "C"
        filename.append(ext)
        filename.append("fits")

        hdul.writeto(
            os.path.join(savePath, ".".join(filename)),
            overwrite=overwrite)


def loadFITS(FITSfile: str) -> dataContainer:
    """
    Load a FITS file with DES atmospheric solutions.

    FITS file must have all the correct header and table information. This
    function should match up perfectly with dataContainer.saveFITS such that
    the file that method generates can be opened up by this function.

    Arguments
    ---------
    FITSfile : str
        The path to the FITS file that will be loaded.

    Returns
    -------
    dataC : dataContainer
        The object that contains all of the FITS information.
    """
    # Open the FITS file to get the HDUlist.
    file = glob.glob(FITSfile)[0]
    hdul = fits.open(file)

    # Initialize the dataContainer object and load the header information.
    # First load set the kwarg metadata.
    dataC = dataContainer()
    dataC.FITSfile = FITSfile
    dataC.OUTfile = os.path.splitext(FITSfile)[0]+".out"
    dataC.header = hdul[0].header
    dataC.expNum = hdul[0].header["expNum"]
    dataC.band = hdul[0].header["band"]
    dataC.earthRef = hdul[0].header["earthRef"]
    dataC.tol = hdul[0].header["tol"]*u.arcsec
    dataC.nSigma = hdul[0].header["nSigma"]
    dataC.vSet = hdul[0].header["vSet"]
    dataC.randomState = hdul[0].header["randomState"]
    dataC.maxDESErr = hdul[0].header["maxDESErr"]*u.mas**2
    dataC.minDESErr = hdul[0].header["minDESErr"]*u.mas**2
    dataC.downselect = hdul[0].header["downselect"]
    dataC.useRMS = hdul[0].header["useRMS"]
    dataC.curl = hdul[0].header["curl"]

    # Load in the TV set (training and validation sets).
    dataC.TV = tb.QTable(hdul[1].data)
    dataC.TV["X"].unit = u.deg
    dataC.TV["Y"].unit = u.deg
    dataC.TV["dX"].unit = u.mas
    dataC.TV["dY"].unit = u.mas
    dataC.TV["DES variance"].unit = u.mas**2
    dataC.TV["GAIA covariance"].unit = u.mas**2
    dataC.TV["fbar_s dX"].unit = u.mas
    dataC.TV["fbar_s dY"].unit = u.mas

    # Load the final kernel parameters.
    dataC.params = np.zeros(5)
    dataC.params[0] = hdul[0].header["var"]
    dataC.params[1] = hdul[0].header["outerScale"]
    dataC.params[2] = hdul[0].header["diameter"]
    dataC.params[3] = hdul[0].header["wind_x"]
    dataC.params[4] = hdul[0].header["wind_y"]

    try:
        # Load the kernel parameters from after the fitCorr step.
        dataC.fitCorrParams = np.zeros(5)
        dataC.fitCorrParams[0] = hdul[0].header["fC_var"]
        dataC.fitCorrParams[1] = hdul[0].header["fC_outerScale"]
        dataC.fitCorrParams[2] = hdul[0].header["fC_diameter"]
        dataC.fitCorrParams[3] = hdul[0].header["fC_wind_x"]
        dataC.fitCorrParams[4] = hdul[0].header["fC_wind_y"]
        dataC.TV["fbar_s dX fC"].unit = u.mas
        dataC.TV["fbar_s dY fC"].unit = u.mas
    except Exception:
        # Load the kernel parameters from after the fitCorr step.
        # Use the old style of key. The fbar_s fC columns won't exist.
        dataC.fitCorrParams = np.zeros(5)
        dataC.fitCorrParams[0] = hdul[0].header["fcvar"]
        dataC.fitCorrParams[1] = hdul[0].header["fcouterScale"]
        dataC.fitCorrParams[2] = hdul[0].header["fcdiameter"]
        dataC.fitCorrParams[3] = hdul[0].header["fcwind_x"]
        dataC.fitCorrParams[4] = hdul[0].header["fcwind_y"]

    # Load in the prediction set.
    dataC.Pred = tb.QTable(hdul[2].data)
    dataC.Pred["X"].unit = u.deg
    dataC.Pred["Y"].unit = u.deg
    dataC.Pred["DES variance"].unit = u.mas**2

    Train, Valid = makeSplit(dataC.TV, dataC.vSet)
    dataC.makeArrays(Train, Valid)
    dataC.arr = np.zeros(len(dataC.TV)).astype(bool)

    # Return the dataContainer object
    return dataC


def loadNPZ(file):

    data = np.load(file, allow_pickle=True)

    expNum = data["expNum"].item()
    randomState = data["randomState"].item()
    ind_GAIA = data["ind_GAIA"]
    ind_DES = data["ind_DES"]

    dataC = dataContainer(randomState)
    dataC.expNum = expNum
    dataC.ind_GAIA = ind_GAIA
    dataC.ind_DES = ind_DES

    dataC.X = data["X"]
    dataC.Xtrain = data["Xtrain"]
    dataC.Xtrain0 = data["Xtrain0"]
    dataC.Xvalid = data["Xvalid"]
    dataC.Xvalid0 = data["Xvalid0"]
    dataC.Xpred = data["Xpred"]

    dataC.Y = data["Y"]
    dataC.Ytrain = data["Ytrain"]
    dataC.Ytrain0 = data["Ytrain0"]
    dataC.Yvalid = data["Yvalid"]
    dataC.Yvalid0 = data["Yvalid0"]

    dataC.E_GAIA = data["E_GAIA"]
    dataC.Etrain_GAIA = data["Etrain_GAIA"]
    dataC.Etrain0_GAIA = data["Etrain0_GAIA"]
    dataC.Evalid_GAIA = data["Evalid_GAIA"]
    dataC.Evalid0_GAIA = data["Evalid0_GAIA"]

    dataC.E_DES = data["E_DES"]
    dataC.Etrain_DES = data["Etrain_DES"]
    dataC.Etrain0_DES = data["Etrain0_DES"]
    dataC.Evalid_DES = data["Evalid_DES"]
    dataC.Evalid0_DES = data["Evalid0_DES"]
    dataC.Epred_DES = data["Epred_DES"]

    dataC.params = data["params"]
    dataC.fbar_s = data["fbar_s"]
    dataC.fbar_s_train = data["fbar_s_train"]
    dataC.fbar_s_valid = data["fbar_s_valid"]

    return dataC


def makeSplit(TV: tb.table.QTable, vSet: str) -> tuple:
    """
    Split the TV dataset into training and validation set tables.

    Depending on which subset you want to be the validation set (vSet) ths
    function will split the TV dataset table into a validation set table and a
    training set table.

    Arguments
    ---------
        TV: tb.table.QTable
            The TV dataset table with the entire training and validation set.
            Must have columns for the five subset masks.
        vSet: str
            The subset you want to become the validation set. One of
            "Subset A", "Subset B", "Subset C", "Subset D", or "Subset E".

    Returns
    -------
    Train : tb.table.QTable
        Table that contains all of the same columns as TV but only the rows
        corresponding to the chosen training set.
    Valid : tb.table.QTable
        Table that contains all of the same columns as TV but only the rows
        corresponding to the chosen validation set.
    """
    # Create the masks that index only the training set and only the
    # validation set.
    subsets = ["Subset A", "Subset B", "Subset C", "Subset D", "Subset E"]
    subsets.remove(vSet)

    train_mask = \
        TV[subsets[0]] + \
        TV[subsets[1]] + \
        TV[subsets[2]] + \
        TV[subsets[3]]
    valid_mask = TV[vSet]

    # Index the TV array and get the new training and validation set tables.
    Train = TV[train_mask]
    Valid = TV[valid_mask]

    return Train, Valid


@u.quantity_input
def getXi(
    X: np.ndarray,
    Y: np.ndarray,
    rMax: u.deg = 0.02*u.deg,
    rMin: u.mas = 5*u.mas
        ):
    """
    Calculates angle-averaged 2pt correlation function.

    Given a minimum and maximum separation, this function calculcations the
    angle-averaged 2pt correlation function of the DES-Gaia astrometric
    residual field. When rMax is set to 0.02 deg and rMin is set to something
    smaller than the smallest likely separation between any two stars on the
    sky, this function calculates what we call xi_0.02. This approximates
    (underestimates) the correlation function at 0 separation.

    Arguments
    ---------
    X : np.ndarray
        Shape (N, 2) array of astrometric positions of DES stars ((x, y) or
        (u, v) depending on which convention you like.).
    Y : np.ndarray
        Shape (N, 2) array of astrometric residuals between DES star positions
        and Gaia star positions ((dx, dy) or (du, dv) depending on which
        convention you like.).
    rMax : Quantity object (angle)
        All pairs of stars with angular separation less than this value will
        be included in the calculation.
    rMin : Quantity object (angle)
        All pairs of stars with angular separation greater than this value
        will be included in the calculation.

    Returns
    -------
    xiplus : float
        Angle-averaged 2pt correlation function of all pairs of points of a
        given residual field with separation less than rMax and greater than
        rMin.
    Uerr : float
        Standard deviation of xiplus in the x (or u) direction.
    Verr : float
        Standard deviation of xiplus in the y (or v) direction.
    prs : np.ndarray
        The list of unique pairs of points in the residual field that was used
        to calculate the correlation function.
    """
    rMax = rMax.to(u.deg).value
    rMin = rMin.to(u.deg).value

    # Use cKDTree to make a tree of the astrometric positions.
    kdt = cKDTree(X)

    # Query the tree for all pairs of points with separation less than rMax.
    prs_set = kdt.query_pairs(rMax, output_type='set')

    # Query the tree for all pairs of points with separation less than rMin.
    # Subtract this set from the previous set of points. Because these ar
    # sets, this operation removes all pairs of points with separation less
    # than rMin from prs_set. Now prs_set contains all pairs of points with
    # separation less than rMax and greater than rMin.
    prs_set -= kdt.query_pairs(rMin, output_type='set')

    # Convert prs_set from a set to a numpy array in order to do the following
    # calculations.
    prs = np.array(list(prs_set))

    # Calculate the correlation function:
    # xi(rMin < r < rMax) = <Y_i * Y_j>
    xiplus = np.mean(np.sum(Y[prs[:, 0]] * Y[prs[:, 1]], axis=1))

    # Find the errors on this measurement
    err = np.std(Y[prs[:, 0]] * Y[prs[:, 1]], axis=0)
    err /= np.sqrt(prs.shape[0])
    Uerr, Verr = err

    return xiplus, Uerr, Verr, prs


def makeW(
    E_GAIA: np.ndarray,
    E_DES: np.ndarray,
    useRMS: bool = False,
    curl: bool = False
        ) -> tuple:
    """
    Makes the White kernel matrices for DES and Gaia errors for GPR.

    Takes arrays of DES and Gaia positional errors and turns them into the
    correctly shaped White kernel matrix for us in GPR (see vK2KGPR.py)

    Arguments
    ---------
    E_GAIA : np.ndarray
        Shape (N, 2, 2) array. This is a length N list of (2, 2) covariance
        matrices for Gaia astrometric positions.
    E_DES : np.ndarray
        If useRMS is False, this is a shape (N,) array of DES astrometric
        positions errors. If useRMS is True, this is a shape (N, 2, 2) array.
        A length N list of (2, 2) diagonal covariance matrices for DES
        astrometric positions.
    useRMS : bool
        Whether or not DES variance values have been replaced with empirically
        calculated RMS values. See dataContainer.load()
    curl : bool
        Whether or not the GPR algorithm takes advantage of the curlfree-ness
        of the atmospheric turbulence field. (See vK2KGPR.py)

    Returns
    -------
    W_GAIA : np.ndarray
        Shape (2N, 2N) array. Used when curl = True. White Kernel matrix for
        the Gaia data.
    W_DES : np.ndarray
        Shape (2N, 2N) array. Used when curl = True. White Kernel matrix for
        the DES data.
    W_GAIAx : np.ndarray
        Shape (N, N) array. Used when curl = False. White kernel matrix for
        the x component of the Gaia data.
    W_GAIAy : np.ndarray
        Shape (N, N) array. Used when curl = False. White kernel matrix for
        the y component of the Gaia data.
    W_DESx : np.ndarray
        Shape (N, N) array. Used when curl = False. White kernel matrix for
        the x component of the DES data.
    W_DESy : np.ndarray
        Shape (N, N) array. Used when curl = False. White kernel matrix for
        the y component of the DES data.
    """
    # DES errors need to be handled a bit differently when useRMS is True.
    if useRMS:
        # E_DES will be shape (N, 2, 2) (essentially an array of matrices) and
        # the entries we want are along the diagonals.
        Ex = E_DES[:, 0, 0]
        Ey = E_DES[:, 1, 1]
    else:
        # E_DES will beshape (N,). In this case we assume the errors for x and
        # y (or u and v) are the same.
        Ex = E_DES
        Ey = E_DES

    # When we make the assumption that the residual field is curlfree, we need
    # (2N, 2N) matrices for our White kernel.
    if curl:
        # Stack the DES errors into shape (N, 2). The flat function makes this
        # into shape (2N,) and does so in "C" order (see flat)
        E = np.vstack([Ex, Ey]).T
        W_DES = np.diag(flat(E))

        # Create an array of shape (N, N, 2, 2) because I know how to properly
        # turn this into a (2N, 2N) array. This stores the 2x2 covariance
        # matrix info for Gaia.
        N = E_GAIA.shape[0]
        out = np.zeros((N, N, 2, 2))
        out[:, :, 0, 0] = np.diag(E_GAIA[:, 0, 0])
        out[:, :, 1, 1] = np.diag(E_GAIA[:, 1, 1])
        out[:, :, 1, 0] = np.diag(E_GAIA[:, 1, 0])
        out[:, :, 0, 1] = np.diag(E_GAIA[:, 0, 1])
        W_GAIA = np.swapaxes(out, 1, 2).reshape((2*N, 2*N))

        return W_GAIA, W_DES

    # If we aren't assuming a curlfree residual field then things are much
    # simpler but we need White kernels for both the x and y components.
    else:
        # The DES white kernel for each component.
        W_DESx = np.diag(Ex)
        W_DESy = np.diag(Ey)

        # The Gaia white kernel for each component. A diagonal matrix where
        # E_GAIA[:, 0, 0] is the variance on the x component for each star,
        # and E_GAIA[:, 1, 1] is the variance on the y component for each star.
        W_GAIAx = np.diag(E_GAIA[:, 0, 0])
        W_GAIAy = np.diag(E_GAIA[:, 1, 1])

        return W_GAIAx, W_GAIAy, W_DESx, W_DESy


def getGrid(X1: np.ndarray, X2: np.ndarray) -> tuple:
    """
    Define uniform grid of points to evaluate a kernel function on.

    This function basically accepts two (N, 2) arrays that represent two sets
    of astrometric positions (e.g., the validation and training sets). This
    function then uses meshgrid  to create a uniform grid of points. See
    np.meshgrid for further details.

    Arguments
    ---------
    X1 : np.ndarray
        Shape (N, 2) array. Represents the first coordinate axis of the grid.
    X2 : np.ndarray
        Shape (N, 2) array. Represents the second coordinate axis of the grid.

    Returns
    -------
    du : np.ndarray
        (N, N) array of points on a uniform grid corresponding to the first
        coordinate axis.
    dv : np.ndarray
        (N, N) array of points on a uniform grid corresponding to the second
        coordinate axis.
    """
    # Extract the coordinate arrays from the input (N, 2) arrays.
    u1, u2 = X1[:, 0], X2[:, 0]
    v1, v2 = X1[:, 1], X2[:, 1]

    # Create the grid.
    uu1, uu2 = np.meshgrid(u1, u2)
    vv1, vv2 = np.meshgrid(v1, v2)

    # du and dv basically represent the separation between two points on this
    # grid.
    du = uu1 - uu2
    dv = vv1 - vv2

    return du, dv


def flat(arr: np.ndarray) -> np.ndarray:
    """
    Flattens (N, 2) arrays in row-major order.

    Useful because the more natural way to process data is to use arrays of
    shape (N, 2), but for a curl-free kernel a shape (2N,) array is necessary.

    Arguments
    ---------
    arr : np.ndarray
        Shape (N, 2) array. Will be flattened in row-major order (C order).

    Returns
    -------
    arr : np.ndarray
        Shape (2N,) array. Flattened array.
    """
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    return arr.flatten(order='C')


def unflat(arr: np.ndarray) -> np.ndarray:
    """
    Unflattens a (2N,) array in row-major order.

    Turns a shape (2N,) array into a shape (N, 2) array in row-major order.
    This form of an array is useful for plotting and general manipulation.

    Arguments
    ---------
    arr : np.ndarray
        Shape (2N,) array. Will be reshaped in row-major order (C order) to
        shape (N, 2).

    Returns
    -------
    arr : np.ndarray
        Shape (N, 2) array. Reshaped array.
    """
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    return arr.reshape((arr.shape[0] // 2, 2), order='C')


def printParams(
    params: np.ndarray,
    header: bool = False,
    FoM: float = None,
    FoMtype: str = None,
    file: str = None,
    printing: bool = True
        ) -> None:
    """
    Print GPR kernel parameters to std.io and/or write them to a file.

    This function can print kernel parameters in a pleasing format. Also
    allows functionality for printing standard column headers so that the
    output of many rows of this function is readable.

    Arguments
    ---------
    params : np.ndarray
        Array containing kernel parameters to be printed.

    Keyword Arguments
    -----------------
    header : bool
        Whether or not to be printing column headers rather than kernel
        parameters.
    FoM : float
        The value of the figure-of-merit to be printed. header must be False.
    FoMtype : str
        The name of the figure-of-merit to be printed as a column header.
        header must be True.
    file : str
    
    XXX
    """

    # If you want to print column headers.
    if header:
        # Names of the column headers
        names = ["K Variance", "Outer Scale", "Diameter", "Wind X", "Wind Y"]

        # Add the figure-of-merit column header.
        if FoMtype is not None:
            names.insert(0, FoMtype)

        # If there happens to be 6 parameters instead of 5 then it is most
        # likely a W Variance parameter. Add the column header.
        if params.size == 6:
            names.append("W Variance")

        line = "".join([f"{name:<15}" for name in names])

    # If you want to print kernel parameters
    else:
        # Add the figure-of-merit column.
        if FoM is not None:
            params = np.insert(params, 0, FoM)
            
        # Form the line that will be printed.
        line = "".join([f"{np.round(param, 7):<15}" for param in params])

    # Write the line to a file.
    if file is not None:
        with open(file, mode="a+") as f:
            f.write(line + "\n")

    # Print to std.out
    if printing:
        print(line)


@u.quantity_input
def calcPixelGrid(
    x: u.deg, y: u.deg, dx: u.mas, dy: u.mas, err: u.mas,
    minPoints: int = 100, pixelsPerBin: int = 500, maxErr: u.mas = 50*u.mas
        ):
    """
    Calculates a pixel grid to make a weighted and binned 2d vector diagram.

    Return a 2d vector diagram (quiver plot) of weighted mean astrometric
    residuals in pixelized areas. Takes in positions (x, y) and vectors (dx,
    dy) to make a quiver plot, but the plot is binned (according to
    pixelsPerBin) and weighted (according to err).

    Arguments
    ---------
    x : Quantity object (angle)
        Shape (N,) array. Specifies first component of astrometric
        position.
    y : Quantity object (angle)
        Shape (N,) array. Specifies second component of astrometric
        position.
    dx : Quantity object (angle)
        Shape (N,) array. Specifies first component of astrometric
        residuals.
    dy : Quantity object (angle)
        Shape (N,) array. Specifies second component of astrometric
        residuals.
    err : Quantity object (angle)
        Shape (N,) array. Specifies error on each measurement (dx, dy)

    Keyword Arguments
    -----------------
    minPoints : int
        Minimum number of points that will be plotted.
    pixelsPerBin : int
        Number of pixels that are represented by one square bin.
    maxErr : Quantity object  (angle)
        Largest error that a bin can have and still be plotted. Avoids
        clutteriing the plot with noiisy arrows.

    Returns
    -------
    x : Quantity object (angle)
        Shape (M,) array. Specifies first component of binned astrometric
        position.
    y : Quantity object (angle)
        Shape (M,) array. Specifies second component of binned astrometric
        position.
    dx : Quantity object (angle)
        Shape (M,) array. Specifies first component of binned astrometric
        residuals.
    dy : Quantity object (angle)
        Shape (M,) array. Specifies second component of binned astrometric
        residuals.
    errors : tuple
        (3,). RMS error of binned x and y, and noise.
    cellSize : Quantity object (angle)
        Helpful value for making the arrow scale on a quiver plot.
    """
    # Check that all arrays (x, y, dx, dy, err) are of shape (N,)
    if not (
        np.all([arr.shape == x.shape for arr in [x, y, dx, dy, err]])
        and np.all([arr.ndim == 1 for arr in [x, y, dx, dy, err]])
    ):
        raise TypeError(
            f"x, y, dx, dy, and err arrays must be 1 dimensional and the "
            f"same shape, but are {x.shape}, {y.shape}, {dx.shape}, "
            f"{dy.shape}, and {err.shape} respectively.")

    # Check that there are enough data points to do this computation
    if x.shape[0] < minPoints:
        raise ValueError(
            f"There are not enough points to do this calculation. The minimum "
            f"number of points is {minPoints} and the length of the dataset "
            f"is {x.shape[0]}.")

    # Set arrays to correct units.
    x = x.to(u.deg)
    y = y.to(u.deg)
    dx = dx.to(u.mas)
    dy = dy.to(u.mas)
    err = err.to(u.mas)
    maxErr = maxErr.to(u.mas)

    # Find the min and max values of x and y in order calculate the bin grid.
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # Calculate the size of each bin in degrees.
    pixelScale = 264.*u.mas  # Nominal mas per pixel
    cellSize = (pixelsPerBin * pixelScale).to(u.deg)

    # Calculate the size of the bin grid for x and y.
    binGridSize_x = int(np.ceil((max_x - min_x) / cellSize))
    binGridSize_y = int(np.ceil((max_y - min_y) / cellSize))

    # Find the bin that each data point belongs to
    ix = np.array(np.floor((x - min_x) / cellSize), dtype=int)
    iy = np.array(np.floor((y - min_y) / cellSize), dtype=int)

    # Make sure that every point is in a bin between the first bin (0)
    # and the last bin (binGridSize - 1)
    ix = np.clip(ix, 0, binGridSize_x - 1)
    iy = np.clip(iy, 0, binGridSize_y - 1)

    # ???
    index = iy * binGridSize_x + ix

    # Converts from standard deviation to 1/variance
    # Watch out for how np.where handles units.
    weight = np.where(err.value > 0, err.value**-2, 0)

    # ???
    totalBinGridSize = binGridSize_x * binGridSize_y
    sumWeights_x = np.histogram(
        index,
        bins=totalBinGridSize,
        range=(-0.5, totalBinGridSize + 0.5),
        weights=(weight * dx))[0].value
    sumWeights_y = np.histogram(
        index,
        bins=totalBinGridSize,
        range=(-0.5, totalBinGridSize + 0.5),
        weights=(weight * dy))[0].value
    sumWeights = np.histogram(
        index,
        bins=totalBinGridSize,
        range=(-0.5, totalBinGridSize + 0.5),
        weights=weight)[0]

    # Smallest weight that we'd want to plot a point for
    minWeight = maxErr.value**-2

    # If a value is below minWeight, then replace it with noData
    sumWeights_x = np.where(
        sumWeights > minWeight,
        sumWeights_x / sumWeights,
        np.nan)
    sumWeights_y = np.where(
        sumWeights > minWeight,
        sumWeights_y / sumWeights,
        np.nan)
    sumWeights = np.where(
        sumWeights > minWeight,
        sumWeights**-1.,
        np.nan)

    # Make an x and y position for each cell to be the center of its cell
    x = np.arange(totalBinGridSize, dtype=int)
    x = (((x % binGridSize_x) + 0.5) * cellSize) + min_x
    y = np.arange(totalBinGridSize, dtype=int)
    y = (((y // binGridSize_x) + 0.5) * cellSize) + min_y

    # Finally take the relevant points to plot
    usefulIndices = np.logical_and(sumWeights != 0,
                                   sumWeights < maxErr.value**2)
    x = x[usefulIndices]
    y = y[usefulIndices]
    dx = sumWeights_x[usefulIndices]*u.mas
    dy = sumWeights_y[usefulIndices]*u.mas

    # Calculate rms and noise and print it.
    RMS_x = np.std(sumWeights_x[sumWeights > 0.])*u.mas
    RMS_y = np.std(sumWeights_y[sumWeights > 0.])*u.mas
    noise = np.sqrt(np.mean(sumWeights[sumWeights > 0.]))*u.mas

    return x, y, dx, dy, (RMS_x, RMS_y, noise), cellSize


@u.quantity_input
def calcDivCurl(
    x: u.deg, y: u.deg, dx: u.mas, dy: u.mas
        ):
    """
    Calculate divergence and curl of the given vector field.

    Given vector displacement (dx, dy) defined on identical 2d grids, return
    arrays giving divergence and curl of the vector field. These will have NaN
    in pixels without useful info.

    Arguments
    ---------
    x : Quantity object (angle)
        Shape (N,) array. Specifies first component of astrometric
        position.
    y : Quantity object (angle)
        Shape (N,) array. Specifies second component of astrometric
        position.
    dx : Quantity object (angle)
        Shape (N,) array. Specifies first component of astrometric
        residuals.
    dy : Quantity object (angle)
        Shape (N,) array. Specifies second component of astrometric
        residuals.

    Returns
    -------
    div : np.ndarray
        2d array. Divergence of the residual field.
    curl : np.ndarray
        2d array. Curl of the residual field.
    RMSdiv : float
        RMS of the divergence of the residual field.
    RMScurl : float
        RMS of the curl of the residual field.
    """
    # This line has been replaced because sometimes this line happens to be
    # zero and messes everything up. Removing all the points from np.diff(x)
    # that are zero seems to have little to no effect on the resulting plot
    # and helps get rid of this fairly frequent error.
    # 31 January 2020
    # step = np.min(np.abs(np.diff(u)))
    step = np.min(np.abs(np.diff(x)[np.diff(x) != 0]))

    ix = np.array(np.floor((x / step) + 0.1), dtype=int)
    iy = np.array(np.floor((y / step) + 0.1), dtype=int)

    ix -= np.min(ix)
    iy -= np.min(iy)

    dx2d = np.zeros((np.max(iy) + 1, np.max(ix) + 1), dtype=float)
    dy2d = np.array(dx2d)

    valid = np.zeros_like(dx2d, dtype=bool)
    valid[iy, ix] = True
    use = np.logical_and(valid[1:-1, :-2], valid[1:-1, 2:])
    use = np.logical_and(use, valid[:-2, 1:-1])
    use = np.logical_and(use, valid[2:, 1:-1])

    dx2d[iy, ix] = dx
    dy2d[iy, ix] = dy

    # XXX These lines may be wrong?
    dxdx = dy2d[2:, 1:-1] - dy2d[:-2, 1:-1]
    dydx = dx2d[2:, 1:-1] - dx2d[:-2, 1:-1]
    dxdy = dy2d[1:-1, 2:] - dy2d[1:-1, :-2]
    dydy = dx2d[1:-1, 2:] - dx2d[1:-1, :-2]

    div = np.where(use, dxdx + dydy, np.nan)
    curl = np.where(use, dydx - dxdy, np.nan)

    RMSdiv = np.sqrt(gbutil.clippedMean(div[div == div], 5.)[1])
    RMScurl = np.sqrt(gbutil.clippedMean(curl[div == div], 5.)[1])

    return div, curl, RMSdiv, RMScurl


@u.quantity_input
def calcCorrelation(
    x: u.deg, y: u.deg, dx: u.mas, dy: u.mas,
    rmin: u.arcsec = 5*u.arcsec, rmax: u.arcsec = 1.5*u.deg,
    dlogr: float = 0.05
        ):
    """
    Produce angle-averaged 2-point correlation functions of astrometric error.

    Using bute-force pair counting, calculate the angle-averaged 2-point
    correlation functions (xi_+, xi_-, xi_z^2, xi_x) for the supplied sample
    of data. See appendix of Bernstein et al. (2017) for more detaile
    explanation of calculations.

    Arguments
    ---------
    x : Quantity object (angle)
        Shape (N,) array. Specifies first component of astrometric
        position.
    y : Quantity object (angle)
        Shape (N,) array. Specifies second component of astrometric
        position.
    dx : Quantity object (angle)
        Shape (N,) array. Specifies first component of astrometric
        residuals.
    dy : Quantity object (angle)
        Shape (N,) array. Specifies second component of astrometric
        residuals.

    Keyword Arguments
    -----------------
        rmin : Quantity object
            Minimum separation between points to care about.
        rmax : Quantity object
            Maximum separation between points to care about.
        dlogr : float
            Logarithmic step size between rmin and rmax.

    Return
    ------
    logr : np.ndarray
        Mean log(radius) in each bin in degrees.
    xiplus : np.ndarray
        Angle-averaged two-point correlation function in mas^2:
        <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
    ximinus : np.ndarray
        <vr1 vr2 - vt1 vt2> in mas^2
    xicross : np.ndarray
        <vr1 vt2 + vt1 vr2> in mas^2
    xiz2 : np.ndarray
        <vx1 vx2 - vy1 vy2 + 2 i vx1 vy2> in mas^2
    xiE : np.ndarray
        E-modes of angle-averaged two-point correlation function in mas^2.
    xiB : np.ndarray
        B-modes of angle-averaged two-point correlation function ni mas^2.
    """
    # Check that all arrays (x, y, dx, dy) are of shape (N,)
    if not (
        np.all([arr.shape == x.shape for arr in [x, y, dx, dy]])
        and np.all([arr.ndim == 1 for arr in [x, y, dx, dy]])
    ):
        raise TypeError(
            f"x, y, dx, dy, and err arrays must be 1 dimensional and the "
            f"same shape, but are {x.shape}, {y.shape}, {dx.shape}, "
            f"and {dy.shape} respectively.")

    # Check that the kwarg dlogr is a float.
    if not isinstance(dlogr, float):
        raise TypeError(f"dlogr must be a float but is {type(dlogr)}.")

    # Make sure everything is in the correct units and then just take the
    # value. Don't feel like integrating units into the calculation.
    rmin = rmin.to(u.deg).value
    rmax = rmax.to(u.deg).value
    x = x.to(u.deg).value
    y = y.to(u.deg).value
    dx = dx.to(u.mas).value
    dy = dy.to(u.mas).value

    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(x))

    # Omit self-pairs
    use = i1 != i2
    i1 = i1[use]
    i2 = i2[use]
    del use

    # Make complex separation vector
    dr = 1j * (y[i2] - y[i1])
    dr += x[i2] - x[i1]

    # Make log(radius) vector to bin data
    logdr = np.log(np.absolute(dr))
    logrmin = np.log(rmin)
    nBins = int(np.ceil(np.log(rmax / rmin) / dlogr))
    hrange = (logrmin, logrmin + (nBins * dlogr))
    counts = np.histogram(logdr, bins=nBins, range=hrange)[0]
    logr = np.histogram(logdr, bins=nBins, range=hrange, weights=logdr)[0]
    logr /= counts

    # Calculate xi_+
    vec = dx + 1j*dy
    vvec = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram(logdr, bins=nBins, range=hrange, weights=vvec)[0]
    xiplus /= counts

    # Calculate xi_z^2
    vvec = vec[i1] * vec[i2]
    xiz2 = np.histogram(logdr, bins=nBins, range=hrange, weights=vvec)[0]
    xiz2 /= counts

    # Calculate xi_- and xi_x
    tmp = vvec * np.conj(dr)
    vvec = tmp * np.conj(dr)
    dr = dr.real*dr.real + dr.imag*dr.imag
    vvec /= dr
    del dr
    ximinus = np.histogram(logdr, bins=nBins, range=hrange, weights=vvec)[0]
    ximinus /= counts
    xicross = np.imag(ximinus)
    ximinus = np.real(ximinus)

    # Calculate xi_E and xi_B
    dlogr = np.zeros_like(logr)
    dlogr[1:-1] = 0.5 * (logr[2:] - logr[:-2])
    tmp = np.array(ximinus) * dlogr
    integral = np.cumsum(tmp[::-1])[::-1]
    xiB = 0.5 * (xiplus - ximinus) + integral
    xiE = xiplus - xiB

    return logr, xiplus, ximinus, xicross, xiz2, xiE, xiB


@u.quantity_input
def calcCorrelation2D(
    x: u.deg, y: u.deg, dx: u.mas, dy: u.mas,
    rmax: u.deg = 1*u.deg, nBins: int = 250
        ):
    """
    Produce 2d 2-point correlation functions of an astrometric residual field.

    Produce 2d 2-point correlation function of total displacement power for
    the supplied sample of data. Uses brute-force pair counting.

    Arguments
    ---------
    x : Quantity object (angle)
        Shape (N,) array. Specifies first component of astrometric
        position.
    y : Quantity object (angle)
        Shape (N,) array. Specifies second component of astrometric
        position.
    dx : Quantity object (angle)
        Shape (N,) array. Specifies first component of astrometric
        residuals.
    dy : Quantity object (angle)
        Shape (N,) array. Specifies second component of astrometric
        residuals.

    Keyword Arguments
    -----------------
        rmax : Quantity object
            Maximum separation between points to consider. Will consider all
            pairs of points at all angles with less separation than rmax.
        nBins : int
            Final 2d array will be of shape (nBins, nBins).

    Return
    ------
        xiplus : np.ndarray
            2d 2pt correlation function in mas^2:
            xi_+ - <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
        counts : np.ndarray
            Number of pairs in each separation bin
    """
    # Check that all arrays (x, y, dx, dy) are of shape (N,)
    if not (
        np.all([arr.shape == x.shape for arr in [x, y, dx, dy]])
        and np.all([arr.ndim == 1 for arr in [x, y, dx, dy]])
    ):
        raise TypeError(
            f"x, y, dx, dy, and err arrays must be 1 dimensional and the "
            f"same shape, but are {x.shape}, {y.shape}, {dx.shape}, "
            f"and {dy.shape} respectively.")

    # Check that the kwarg nBins is an int.
    if not isinstance(nBins, int):
        raise TypeError(f"nBins must be int but is {type(nBins)}.")

    # Make sure everything is in the correct units and then just take the
    # value. Don't feel like integrating units into the calculation.
    rmax = rmax.to(u.deg).value
    x = x.to(u.deg).value
    y = y.to(u.deg).value
    dx = dx.to(u.mas).value
    dy = dy.to(u.mas).value

    # Final plot dimensions.
    hrange = [[-rmax, rmax], [-rmax, rmax]]

    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(x))

    # Omit self-pairs
    use = i1 != i2
    i1 = i1[use]
    i2 = i2[use]
    del use

    # Make separation vectors and count pairs
    yshift = y[i2] - y[i1]
    xshift = x[i2] - x[i1]
    counts = np.histogram2d(xshift, yshift, bins=nBins, range=hrange)[0]

    # Accumulate displacement sums
    vvec = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram2d(
        xshift,
        yshift,
        bins=nBins,
        range=hrange,
        weights=vvec)[0]
    xiplus /= counts

    # Combine pairs
    xiplus = 0.5 * (xiplus + xiplus[::-1, ::-1])

    return xiplus, counts


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
        A = A.reshape(
            npts, (self.order+1) * (self.order+1))[:, self.use.flatten()]
        b = np.linalg.lstsq(A, z, rcond=None)[0]
        self.coeffs = np.zeros((self.order+1, self.order+1), dtype=float)
        self.coeffs[self.use] = b
        return

    def getCoeffs(self):
        # Return the current coefficients in a vector.
        return self.coeffs[self.use]

    def setCoeffs(self, c):
        # Set the coefficients to the specified vector.
        if len(c.shape) != 1 or c.shape[0] != np.count_nonzero(self.use):
            print("Poly2d.setCoeffs did not get proper-size array", c.shape)
            sys.exit(1)
        self.coeffs[self.use] = c
        return
