# Standard modules
import os
import shutil
import time

# Willow Fox Fortino's modules
import vK2KGPR
import plotGPR
import vonkarmanFT as vk

# Professor Gary Bernstein's modules
import getGaiaDR2 as gaia
import gbutil

# Science modules
import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.coordinates as co
import astropy.table as tb
import astropy.io.fits as fits
import astropy.stats as stats
from astropy.time import Time
from scipy.spatial.ckdtree import cKDTree

from IPython import embed


class dataContainer(object):

    def __init__(self, randomState=0):
        """
        Stores data for GPR analysis.

        Loads astrometry data from fits files and contains a few methods to
        easily format it in a way that GPR methods can deal with.

        Keyword Arguments
            randomState -- (int) Seed for the various numpy and scipy random 
                operations.
        """

        self.randomState = randomState
        
    def summarize(self):
        print(f"Exposure: {self.expNum}")
        print(f"Band: {self.band}")
        
        x = self.TV["X"][self.TV["Mask0"]]
        y = self.TV["Y"][self.TV["Mask0"]]
        dx = self.TV["dX"][self.TV["Mask0"]]
        dy = self.TV["dY"][self.TV["Mask0"]]
        X = np.vstack([x, y]).T
        Y = np.vstack([dx, dy]).T
        xi0, Xerr, Yerr, pairs = getXi(X, Y)
        print("Angle Averaged 2pt Correlation Function of Residual Field")
        print("--All pairs with separations less than 0.02 degrees included in calculation.")
        print("--All stars included according to maxDESErr and minDESErr")
        print("    keyword arguments, as well as all stars that make it through")
        print("    the first round of sigma clipping in the load method.")
        print(f"xi = {np.round(xi0, 3)} ± {np.round(np.sqrt(Xerr**2 + Yerr**2), 3)} mas^2")
        print()
        
        try:
            print("Kernel Parameters from 2d Correlation Fitting")
            vK2KGPR.printParams(
                self.fitCorrParams,
                header=True,
                printing=True
                )
            vK2KGPR.printParams(
                self.fitCorrParams,
                printing=True
                )
            print()

            print("Kernel Parameters from GPR Optimization")
            vK2KGPR.printParams(
                self.params,
                header=True,
                printing=True
                )
            vK2KGPR.printParams(
                self.params,
                printing=True
                )
            print()

            print("Jackknifed xi+ (Inter-set pairs excluded)")
            xi0 = self.header["xi0"]
            Xerr = self.header["xi0_Xerr"]
            Yerr = self.header["xi0_Yerr"]
            print(f"xi0: {np.round(xi0, 3)} ± {np.round(np.sqrt(Xerr**2 + Yerr**2), 3)} mas^2")
            xif = self.header["xif"]
            Xerr = self.header["xif_Xerr"]
            Yerr = self.header["xif_Yerr"]
            print(f"xif: {np.round(xif, 3)} ± {np.round(np.sqrt(Xerr**2 + Yerr**2), 3)} mas^2")
            print(f"Reduction: {np.round(xi0/xif, 3)}")
            print()
            
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
            
            if self.TV["DES variance"].ndim == 3:
                err = np.sqrt(self.TV[self.TV["Maskf"]]["DES variance"][:, 0, 1])
            else:
                err = np.sqrt(self.TV["DES variance"][self.TV["Maskf"]])

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

            plotGPR.Correlation(
                x, y, dx, dy,
                x2=x2, y2=y2, dx2=dx2, dy2=dy2,
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

        except Exception as E:
            print(E)

    def load(
        self,
        expNum,
        zoneDir="/data3/garyb/tno/y6/zone134",
        tile0="DES2203-4623_final.fits",
        earthRef="/home/fortino/y6a1.exposures.positions.fits.gz",
        tileRef="/home/fortino/expnum_tile.fits.gz",
        tol=0.5*u.arcsec,
        nSigma=4,
        vSet="Subset A",
        maxDESErr=np.inf*u.mas**2,
        minDESErr=-np.inf*u.mas**2,
        downselect=1.0,
        useRMS=False,
        useCov=False
        ):
        """
        Docs go here :)
        """
        self.expNum = expNum
        self.zoneDir = zoneDir
        self.tile0 = tile0
        self.earthRef = earthRef
        self.tileRef = tileRef
        self.tol = tol
        self.nSigma = nSigma
        self.vSet = vSet

        # Load in data from a reference tile (tile0). This tile is arbitrary.
        # At the time of implementation, I do not have access to a reference
        # file relating tiles to zones, and exposures to tiles. I only have a
        # reference file (tileRef) that relates exposures to tiles. Therefore,
        # this block of code opens a tile (tile0) that is actually in our
        # (arbitrary) zone of interest for this thesis (zone 134). From this
        # tile I can pick one of its constituent exposures. This way I have
        # chosen an exposure that I know is in zone 134 (the zone I have
        # access to).
        file0 = os.path.join(zoneDir, tile0)
        tab0 = tb.Table.read(file0)

        #--------------------#

        # Use earthRef to find the center (ra, dec) of the exposure as well as
        # the MJD of the exposure.
        pos_tab = tb.Table.read(earthRef, hdu=1)
        pos_tab = pos_tab[pos_tab["expnum"] == self.expNum]
        ra0 = np.array(pos_tab["ra"])[0]*u.deg
        dec0 = np.array(pos_tab["dec"])[0]*u.deg
        DES_obs = Time(np.array(pos_tab["mjd_mid"])[0], format="mjd")

        #--------------------#

        # Use tileRef to find all of the tiles that our exposure is a part of.
        tiles_tab = tb.Table.read(tileRef)
        tiles = tiles_tab[np.array(tiles_tab["EXPNUM"]) == self.expNum]
        tiles = tiles["TILENAME"]
        #--------------------#

        # Create an empty astropy table with all of the necessary columns.
        DES_tab = tab0.copy()
        DES_tab.remove_rows(np.arange(len(DES_tab)))

        # Loop through each tile, open the table, and append (with tv.vstack)
        # the data to our empty table. Also check if the selected exposure is
        # in the Y band because we do not want to use those.
        for tile in tiles:
            try:
                tile = str(tile) + "_final.fits"
                file = os.path.join(zoneDir, tile)
                tab = tb.Table.read(file)
                tab = tab[tab["EXPNUM"] == self.expNum]
                DES_tab = tb.vstack([DES_tab, tab])
            except FileNotFoundError:
                print(f"File not found: {file}, continuing without it")
                continue

        self.band = np.unique(DES_tab["BAND"])[0]

        print(f"Exposure: {self.expNum}")
        print(f"Band: {self.band}")
        print(f"Number of objects: {len(DES_tab)}")

        # Initialize variables for the relevant columns.
        DES_ra = np.array(DES_tab["NEW_RA"])*u.deg
        DES_dec = np.array(DES_tab["NEW_DEC"])*u.deg
        DES_err = ((np.array(DES_tab["ERRAWIN_WORLD"])*u.deg).to(u.mas))**2

        #--------------------#

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

        # Circular error approximation
        GAIA_err = np.array(GAIA_tab["error"])*u.deg

        # Full covariance matrix
        GAIA_cov = np.array(GAIA_tab["cov"])*u.mas**2
        GAIA_cov = np.reshape(GAIA_cov, (GAIA_cov.shape[0], 5, 5))

        #--------------------#

        # Initialize astropy SkyCoord objects to take advantage of astropy's
        # `match_coordinates_sky`.
        X_DES = co.SkyCoord(DES_ra, DES_dec)
        X_GAIA = co.SkyCoord(GAIA_ra, GAIA_dec)

        # Match DES objects with Gaia counterparts based on how close together
        # they are on the sky. 
        idx, sep2d, dist3d = co.match_coordinates_sky(X_GAIA, X_DES)

        # slice that can index the Gaia catalog for only the stars that have a
        # match
        self.ind_GAIA = np.where(sep2d < tol)[0]

        # slice that can index the DES catalog for only the stars that have a
        # match. Will be in the same order as ind_GAIA
        self.ind_DES = idx[self.ind_GAIA]

        print(f"There were {self.ind_GAIA.size} matches within {tol}.")

        #--------------------#

        # Transformation matrix from ICRS to gnomonic projection about 
        # (ra0, dec0).
        M = np.array([
            [-np.sin(ra0),np.cos(ra0),0],
            [-np.cos(ra0)*np.sin(dec0),-np.sin(ra0)*np.sin(dec0),np.cos(dec0)],
            [np.cos(ra0)*np.cos(dec0),np.sin(ra0)*np.cos(dec0),np.sin(dec0)]
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

        #--------------------#

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
        
        #--------------------#
        
        # Find covariance matrix for X_gn_transf_GAIA.
        cov = np.dot(A, np.dot(GAIA_cov, A.T))

        # Swap axes to make the shape of the array more natural to humans.
        cov = np.swapaxes(cov, 1, 0)

        #--------------------#

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

        # Declare attribute for circular measurement error for DES stars.
        # There is not a (2, 2) covariance matrix for each detection because
        # DES only provides circular measurement errors (to my knowledge as a
        # lowly undergraduate).
        self.E_DES = DES_err

        #--------------------#

        # Load the TV set (training + validation) into an astropy table.
        x = tb.Column(data=self.X[self.ind_DES][:, 0], name=f"X")
        y = tb.Column(data=self.X[self.ind_DES][:, 1], name=f"Y")
        dx = tb.Column(data=self.Y[:, 0].to(u.mas), name=f"dX")
        dy = tb.Column(data=self.Y[:, 1].to(u.mas), name=f"dY")
        err_DES = tb.Column(
            data=self.E_DES[self.ind_DES], name=f"DES variance")
        err_GAIA = tb.Column(
            data=self.E_GAIA, name=f"GAIA covariance")
        self.TV = tb.QTable([x, y, dx, dy, err_DES, err_GAIA])

        #--------------------#

        # Sigma clip on the TV residuals. This usually removes about 100
        # objects.
        Y = np.vstack([self.TV["dX"], self.TV["dY"]]).T.value
        mask = stats.sigma_clip(Y, sigma=nSigma, axis=0).mask
        mask = tb.Column(
            data=~np.logical_or(*mask.T),
            name="Mask0",
            description="False if excluded in initial sigma clipping.")
        self.TV.add_column(mask)
        
        #--------------------#
        
        # Remove stars that have less variance than minDESErr and stars
        # that have more variance than maxDESErr. Fold this into Mask0.
        minMask = self.TV["DES variance"] > minDESErr
        maxMask = self.TV["DES variance"] < maxDESErr
        self.TV["Mask0"] = self.TV["Mask0"] & minMask & maxMask
        
        #--------------------#
        
        if useRMS:
            x = self.TV[self.TV["Mask0"]]["X"].value
            y = self.TV[self.TV["Mask0"]]["Y"].value
            dx = self.TV[self.TV["Mask0"]]["dX"].value
            dy = self.TV[self.TV["Mask0"]]["dY"].value
            err = self.TV[self.TV["Mask0"]]["DES variance"].value

            table_inds = np.arange(len(self.TV))

            # Index the table by increasing DES variance
            sort = np.argsort(err)
            sorted_inds = table_inds[self.TV["Mask0"]][sort]

            resid_x = dx[sort]
            resid_y = dy[sort]

            # Split arrays into groups of nStars
            nStars = 256
            sorted_inds = np.array_split(sorted_inds, len(sorted_inds)//nStars)

            resid_x = np.array_split(resid_x, len(dx)//nStars)
            resid_y = np.array_split(resid_y, len(dy)//nStars)

            # Find RMS and median values
            RMSx = np.array([np.std(arr) for arr in resid_x])
            RMSy = np.array([np.std(arr) for arr in resid_y])

            # Get average RMS
            RMSx = RMSx**2
            RMSy = RMSy**2
            RMSxy = 0.5 * (RMSx**2 + RMSy**2)
            
            if useCov:
                DESvar = tb.Column(data=np.zeros((len(self.TV), 2, 2)), name="DES variance", unit=u.mas**2)
                
                for i, (group, RMS) in enumerate(zip(sorted_inds, cov)):
                    N = len(group)
                    RMS = np.ones((N, 2, 2))*RMS*u.mas**2
                    DESvar[group] = RMS
                self.TV["DES variance"] = DESvar
            
            else:
                for i, (group, RMS) in enumerate(zip(sorted_inds, RMSxy)):
                    N = len(group)
                    RMS = np.ones(N)*RMS*(self.TV["DES variance"].unit)
                    self.TV["DES variance"][group] = RMS
        
        #--------------------#

        # Define some placeholder arrays for removing a polynomial fit.
        x = self.TV["X"][self.TV["Mask0"]]
        y = self.TV["Y"][self.TV["Mask0"]]
        dx = self.TV["dX"][self.TV["Mask0"]]
        dy = self.TV["dY"][self.TV["Mask0"]]

        # Remove a 3rd order polynomial fit from the residuals.
        poly = Poly2d(3)
        poly.fit(x, y, dx)
        self.TV["dX"][self.TV["Mask0"]] -= poly.evaluate(x, y)*self.TV["dX"].unit
        poly.fit(x, y, dy)
        self.TV["dY"][self.TV["Mask0"]] -= poly.evaluate(x, y)*self.TV["dY"].unit

        #--------------------#
        
        n = len(self.TV)
        n0 = len(self.TV[self.TV["Mask0"]])
        nTrue = int(np.floor(n0 * downselect))
        
        # Find the indices (relative to the entire table) of the rows
        # where Mask0 = True. Then take a random fraction of those
        # (specified by kwarg downselect) that will be true.
        inds = np.arange(n)
        inds_mask0 = inds[self.TV["Mask0"]]
        rng = np.random.RandomState(self.randomState)
        rng.shuffle(inds_mask0)
        inds_mask0_True = inds_mask0[:nTrue]
        
        # Create a False array of the length of the entire table. Use the
        # above indices to specify which of the rows (that already have Mask0 = True)
        # will be True in this new mask.
        downselectMask = np.zeros(n, dtype=bool)
        downselectMask[inds_mask0_True] = True
        
        self.TV["Mask0"] = self.TV["Mask0"] & downselectMask

        #--------------------#

        # Make an index array for the entire TV set.
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

        #--------------------#

        # Create the prediction dataset table that represents all DES objects
        # that don't have a Gaia counterpart.
        Xpred = np.delete(self.X, self.ind_DES, axis=0)
        Epred = np.delete(self.E_DES, self.ind_DES, axis=0)

        x = tb.Column(data=Xpred[:, 0], name="X")
        y = tb.Column(data=Xpred[:, 1], name="Y")
        e = tb.Column(data=Epred, name="DES variance")

        self.Pred = tb.QTable([x, y, e])

        #--------------------#

        # Make the training and validation sets into attributes of this object
        # like how vK2KGPR.py expects it.
        Train, Valid = makeSplit(self.TV, self.vSet)
        self.makeArrays(Train, Valid)

        #--------------------#

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
        
        #--------------------#
        
        self.nData = len(self.TV) + len(self.Pred)
        self.nPred = len(self.Pred)
        self.nTrain = len(Train)
        self.nValid = len(Valid)

    def makeArrays(self, Train, Valid):
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

    def postFitCorr_sigmaClip(self, GP):

        subsets = ["Subset A", "Subset B", "Subset C", "Subset D", "Subset E"]
        subsets.remove(self.vSet)
        train_mask = self.TV[subsets[0]] + \
                     self.TV[subsets[1]] + \
                     self.TV[subsets[2]] + \
                     self.TV[subsets[3]]
        valid_mask = self.TV[self.vSet]

        maskX = tb.Column(
            data=self.arr.copy(),
            name=f"MaskCorrFit")
        self.TV.add_column(maskX)
        
        # Sigma clip on the validation set.
        GP.predict(self.Xvalid)
        mask = stats.sigma_clip(
            self.Yvalid - self.fbar_s,
            sigma=self.nSigma, axis=0).mask
        mask = ~np.logical_or(*mask.T)
        self.TV["MaskCorrFit"][valid_mask] = mask

        # Sigma clip on the training set.
        GP. predict(self.Xtrain)
        mask = stats.sigma_clip(
            self.Ytrain - self.fbar_s,
            sigma=self.nSigma, axis=0).mask
        mask = ~np.logical_or(*mask.T)
        self.TV["MaskCorrFit"][train_mask] = mask
        
        Train, Valid = makeSplit(self.TV[self.TV["MaskCorrFit"]], self.vSet)
        self.makeArrays(Train, Valid)
        
    def JackKnife(self, GP):
        
        maskjk = tb.Column(data=self.arr.copy(), name=f"MaskJackKnife")
        self.TV.add_column(maskjk)
        
        subsets = ["Subset A", "Subset B", "Subset C", "Subset D", "Subset E"]
        for i in range(5):
            Train, Valid = makeSplit(self.TV[self.TV["MaskCorrFit"]], subsets[i])
            self.makeArrays(Train, Valid)
            
            GP.fit(self.params)
            GP.predict(self.Xvalid)
            index = np.logical_and(self.TV[subsets[i]], self.TV["MaskCorrFit"])
            self.TV["fbar_s dX"][index] = self.fbar_s[:, 0]*u.mas
            self.TV["fbar_s dY"][index] = self.fbar_s[:, 1]*u.mas
    
            mask = stats.sigma_clip(self.fbar_s, sigma=self.nSigma, axis=0).mask
            mask = ~np.logical_or(*mask.T)
            self.TV["MaskJackKnife"][index] = mask
            
        maskf = self.TV["Subset A"] + self.TV["Subset B"] + self.TV["Subset C"] + self.TV["Subset D"] + self.TV["Subset E"]
        maskf = np.logical_and(maskf, np.logical_and(self.TV["MaskJackKnife"], self.TV["MaskCorrFit"]))
        maskf = tb.Column(data=maskf, name="Maskf")
        self.TV.add_column(maskf)
        
    def JackKnifeXi(self, allPairs=False):
        if allPairs:
            x = self.TV["X"][self.TV["Maskf"]]
            y = self.TV["Y"][self.TV["Maskf"]]
            dx = self.TV["dX"][self.TV["Maskf"]]
            dy = self.TV["dY"][self.TV["Maskf"]]
            dx2 = dx - self.TV["fbar_s dX"][self.TV["Maskf"]]
            dy2 = dy - self.TV["fbar_s dY"][self.TV["Maskf"]]
            
            xi = getXi(np.vstack([x, y]).T, np.vstack([dx, dy]).T)
            xi2 = getXi(np.vstack([x, y]).T, np.vstack([dx2, dy2]).T)
            return xi, xi2
        
        elif allPairs == False:
            prs_list = []
            prs_list2 = []
            subsets = ["Subset A", "Subset B", "Subset C", "Subset D", "Subset E"]
            for subset in subsets:
                data = self.TV[self.TV[subset] & self.TV["Maskf"]]
                X = np.vstack([data["X"], data["Y"]]).T
                Y = np.vstack([data["dX"], data["dY"]]).T
                Y2 = Y - np.vstack([data["fbar_s dX"], data["fbar_s dY"]]).T

                xi, Uerr, Verr, prs = getXi(X, Y)
                xi2, Uerr2, Verr2, prs2 = getXi(X, Y2)

                prs_list.append(Y[prs])
                prs_list2.append(Y2[prs2])

            prs = np.vstack(prs_list)
            prs2 = np.vstack(prs_list2)

            X = np.vstack([self.TV["X"], self.TV["Y"]]).T
            Y = np.vstack([self.TV["dX"], self.TV["dY"]]).T
            Y2 = Y - np.vstack([self.TV["fbar_s dX"], self.TV["fbar_s dY"]]).T

            xiplus = np.mean(np.sum(prs[:, 0] * prs[:, 1], axis=1))
            xiplus2 = np.mean(np.sum(prs2[:, 0] * prs2[:, 1], axis=1))

            err = np.std(prs[:, 0] * prs[:, 1], axis=0) / np.sqrt(prs.shape[0])
            err2 = np.std(prs2[:, 0] * prs2[:, 1], axis=0) / np.sqrt(prs2.shape[0])
            
            xi = (xiplus, err[0], err[1])
            xi2 = (xiplus2, err2[0], err2[1])

            return xi, xi2

    def saveFITS(self, savePath, overwrite=True):
        hdr = fits.Header()
        hdr["expNum"] = self.expNum
        hdr["band"] = self.band
        hdr["zoneDir"] = self.zoneDir
        hdr["tile0"] = self.tile0
        hdr["earthRef"] = self.earthRef
        hdr["tileRef"] = self.tileRef
        hdr["tol"] = self.tol.value
        hdr["nSigma"] = self.nSigma
        hdr["vSet"] = self.vSet
        hdr["randomState"] = self.randomState
        
        xi, xi2 = self.JackKnifeXi(allPairs=True)
        hdr["allPairs_xi0"] = xi[0].value
        hdr["allPairs_xi0_Xerr"] = xi[1].value
        hdr["allPairs_xi0_Yerr"] = xi[2].value
        hdr["allPairs_xif"] = xi2[0].value
        hdr["allPairs_xif_Xerr"] = xi2[1].value
        hdr["allPairs_xif_Yerr"] = xi2[2].value
        
        xi, xi2 = self.JackKnifeXi(allPairs=False)
        hdr["xi0"] = xi[0].value
        hdr["xi0_Xerr"] = xi[1].value
        hdr["xi0_Yerr"] = xi[2].value
        hdr["xif"] = xi2[0].value
        hdr["xif_Xerr"] = xi2[1].value
        hdr["xif_Yerr"] = xi2[2].value
        
        hdr["var"] = self.params[0]
        hdr["outerScale"] = self.params[1]
        hdr["diameter"] = self.params[2]
        hdr["wind_x"] = self.params[3]
        hdr["wind_y"] = self.params[4]
        
        hdr["fcvar"] = self.fitCorrParams[0]
        hdr["fcouterScale"] = self.fitCorrParams[1]
        hdr["fcdiameter"] = self.fitCorrParams[2]
        hdr["fcwind_x"] = self.fitCorrParams[3]
        hdr["fcwind_y"] = self.fitCorrParams[4]
        
        prim_HDU = fits.PrimaryHDU(header=hdr)
        TV_HDU = fits.BinTableHDU(data=self.TV)
        Pred_HDU = fits.BinTableHDU(data=self.Pred)

        hdul = fits.HDUList([prim_HDU, TV_HDU, Pred_HDU])
        hdul.writeto(
            os.path.join(savePath, f"DES{self.expNum}_{self.band}.fits"),
            overwrite=overwrite)

def loadFITS(FITSfile):
    hdul = fits.open(FITSfile)

    dataC = dataContainer()
    dataC.header = hdul[0].header
    dataC.expNum = hdul[0].header["expNum"]
    dataC.band = hdul[0].header["band"]
    dataC.zoneDir = hdul[0].header["zoneDir"]
    dataC.tile0 = hdul[0].header["tile0"]
    dataC.earthRef = hdul[0].header["earthRef"]
    dataC.tileRef = hdul[0].header["tileRef"]
    dataC.tol = hdul[0].header["tol"]*u.arcsec
    dataC.nSigma = hdul[0].header["nSigma"]
    dataC.vSet = hdul[0].header["vSet"]
    dataC.randomState = hdul[0].header["randomState"]
    
    dataC.params = np.zeros(5)
    dataC.params[0] = hdul[0].header["var"]
    dataC.params[1] = hdul[0].header["outerScale"]
    dataC.params[2] = hdul[0].header["diameter"]
    dataC.params[3] = hdul[0].header["wind_x"]
    dataC.params[4] = hdul[0].header["wind_y"]
    
    dataC.fitCorrParams = np.zeros(5)
    dataC.fitCorrParams[0] = hdul[0].header["fcvar"]
    dataC.fitCorrParams[1] = hdul[0].header["fcouterScale"]
    dataC.fitCorrParams[2] = hdul[0].header["fcdiameter"]
    dataC.fitCorrParams[3] = hdul[0].header["fcwind_x"]
    dataC.fitCorrParams[4] = hdul[0].header["fcwind_y"]

    dataC.TV = tb.QTable(hdul[1].data)
    dataC.TV["X"].unit = u.deg
    dataC.TV["Y"].unit = u.deg
    dataC.TV["dX"].unit = u.mas
    dataC.TV["dY"].unit = u.mas
    dataC.TV["DES variance"].unit = u.mas**2
    dataC.TV["GAIA covariance"].unit = u.mas**2
    dataC.TV["fbar_s dX"].unit = u.mas
    dataC.TV["fbar_s dY"].unit = u.mas

    dataC.Pred = tb.QTable(hdul[2].data)
    dataC.Pred["X"].unit = u.deg
    dataC.Pred["Y"].unit = u.deg
    dataC.Pred["DES variance"].unit = u.mas**2
    
    return dataC

def loadNPZ(file):
    
    data = np.load(file, allow_pickle=True)
    
    expNum = data["expNum"].item()
    randomState = data["randomState"].item()
    ind_GAIA = data["ind_GAIA"]
    ind_DES = data["ind_DES"]
    nSigma = data["nSigma"].item()
    train_size = data["train_size"].item()
    subSample = data["subSample"].item()

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
        
    dataC.E_GAIA  = data["E_GAIA"]
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
        
def makeSplit(TV, vSet):
    subsets = ["Subset A", "Subset B", "Subset C", "Subset D", "Subset E"]
    subsets.remove(vSet)
    
    train_mask = TV[subsets[0]] + \
                 TV[subsets[1]] + \
                 TV[subsets[2]] + \
                 TV[subsets[3]]
    valid_mask = TV[vSet]
    
    Train = TV[train_mask]
    Valid = TV[valid_mask]
    
    return Train, Valid

def getXi(X, Y, rMax=0.02*u.deg, rMin=5*u.mas):
    res = Y
    kdt = cKDTree(X)

    rMax = rMax.to(u.deg).value
    rMin = rMin.to(u.deg).value

    prs_set = kdt.query_pairs(rMax, output_type='set')
    prs_set -= kdt.query_pairs(rMin, output_type='set')
    prs = np.array(list(prs_set))
    xiplus = np.mean(np.sum(res[prs[:, 0]] * res[prs[:, 1]], axis=1))

    err = np.std(res[prs[:, 0]] * res[prs[:, 1]], axis=0) / np.sqrt(prs.shape[0])
    Uerr, Verr = err

    return xiplus, Uerr, Verr, prs

def makeW(E_GAIA, E_DES):

    N = E_GAIA.shape[0]
    out = np.zeros((N, N, 2, 2))
    out[:, :, 0, 0] = np.diag(E_GAIA[:, 0, 0])
    out[:, :, 1, 1] = np.diag(E_GAIA[:, 1, 1])
    out[:, :, 1, 0] = np.diag(E_GAIA[:, 1, 0])
    out[:, :, 0, 1] = np.diag(E_GAIA[:, 0, 1])
    W_GAIA = np.swapaxes(out, 1, 2).reshape((2*N, 2*N))
    
    if E_DES.ndim == 3:
        ExEy = flat(np.vstack([E_DES[:, 0, 0], E_DES[:, 1, 1]]).T)
        Exy =  flat(np.vstack([E_DES[:, 0, 1], np.zeros(E_DES.shape[0])]).T)

        W_DES = np.diag(ExEy) + (np.diag(Exy, k=1) + np.diag(Exy, k=-1))[:-1, :-1]
        
    else:
        E_DES = np.vstack([E_DES, E_DES]).T
        W_DES = np.diag(flat(E_DES))

    return W_GAIA + W_DES

def getGrid(X1, X2):
    u1, u2 = X1[:, 0], X2[:, 0]
    v1, v2 = X1[:, 1], X2[:, 1]
    uu1, uu2 = np.meshgrid(u1, u2)
    vv1, vv2 = np.meshgrid(v1, v2)
    
    return uu1 - uu2, vv1 - vv2

def flat(arr):
    """
    Flattens 2d arrays in row-major order.
    
    Useful because the more natural way to process data is to use arrays of shape (N, 2), but for a curl-free kernel a shape (2N,) array is necessary.
    
    Parameters:
        arr -- (ndarray) 2d array to be flattened in row-major order (C order).
        
    Returns:
        arr -- (ndarray) 1d array.
    """
    
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    return arr.flatten(order='C')

def unflat(arr):
    """Unflattens a 1d array in row-major order.
    
    Turns a shape (2N,) array into a shape (N, 2) array.
    
    Parameters:
        arr -- (ndarray) 1d array to be unflattened in row-major order (C 
            order).
        
    Returns:
        arr -- (ndarray) 2d array.
    """
    
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    return arr.reshape((arr.shape[0] // 2, 2), order='C')

def calcPixelGrid(
    x, y, dx, dy, err,
    minPoints=100,
    pixelsPerBin=500,
    maxErr=50*u.mas
    ):
    """
    Calculates a pixel grid to make a weighted and binned 2d vector diagram.
    
    Return a 2d vector diagram (quiver plot) of weighted mean astrometric residuals in pixelized areas. Takes in positions (x, y) and vectors (dx, dy) to make a quiver plot, but the plot is binned (according to pixelsPerBin) and weighted (according to err).
    
    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err -- (astropy.units.quantity.Quantity) (N,) specifies error for each vector (dx, dy)
    
    Keyword Arguments:
        minPoints -- (int) Minimum number of max plots that will be plotted
        pixelsPerBin -- (int) number of pixels that are represented by one bin
        maxErr -- (astropy.units.quantity.Quantity) (scalar) largest error that a binned vector can have and still be plotted. Avoids cluttering the plot with noisy arrows.
        
    Returns:
        x -- (astropy.units.quantity.Quantity) (N,) binned x position
        y -- (astropy.units.quantity.Quantity) (N,) binned y position
        dx -- (astropy.units.quantity.Quantity) (N,) binned dx position
        dy -- (astropy.units.quantity.Quantity) (N,) binned dy position
        errors -- (tuple) (3,) RMS error of binned x and y, and noise
        cellSize -- (astropy.units.quantity.Quantity) (scalar) helpful for making the arrow scale on a quiver plot
    """

    # Check that all arrays (x, y, dx, dy, err) are astropy quantity objects
    if not np.all([isinstance(arr, u.quantity.Quantity) 
                   for arr in [x, y, dx, dy, err]]):
        raise TypeError("All input arrays must be of type astropy.units.quantity.Quantity.")

    # Check that x, y have the same units
    if x.unit != y.unit:
        raise u.UnitsError(f"x and y arrays must have the same units but are {x.unit} and {y.unit} respectively.")

    # Check that dx, dy, err have the same units
    if not (dx.unit == dy.unit and dx.unit == err.unit):
        raise u.UnitsError(f"dx, dy, and err arrays must have the same units but are {dx.unit}, {dy.unit}, and {err.unit} respectively.")
        
    # Check that all arrays (x, y, dx, dy, err) are of shape (N,)
    if not (np.all([arr.shape == x.shape for arr in [x, y, dx, dy, err]])
            and np.all([arr.ndim == 1 for arr in [x, y, dx, dy, err]])):
        raise TypeError(f"x, y, dx, dy, and err arrays must be 1 dimensional and the same shape but are {x.shape}, {y.shape}, {dx.shape}, {dy.shape}, and {err.shape} respectively.")
    
    # Check that there are enough data points to do this computation
    if x.shape[0] < minPoints:
        raise ValueError(f"There are not enough points to do this calculation. The minimum number of points is {minPoints} and the length of the dataset is {x.shape[0]}.")
        
    # Check that maxErr is the correct type
    if not isinstance(maxErr, u.quantity.Quantity):
        raise TypeError("maxErr must be of type astropy.units.quantity.Quantity.")
    if maxErr.unit != err.unit:
        raise u.UnitsError(f"maxErr has units of {maxErr.unit} but should have the same units as dx, dy and err ({err.unit}).")
        
    x = x.to(u.deg)
    y = y.to(u.deg)
    dx = dx.to(u.mas)
    dy = dy.to(u.mas)
    err = err.to(u.mas)
    
    # Find the min and max values of x and y in order calculate the bin grid.
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    # Calculate the size of each bin in degrees.
    pixelScale = 264.*u.mas # Nominal mas per pixel
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
    weight = np.where(err > 0., err**-2., 0.)
    
    # ???
    totalBinGridSize = binGridSize_x * binGridSize_y
    sumWeights_x = np.histogram(
        index,
        bins=totalBinGridSize,
        range=(-0.5, totalBinGridSize + 0.5),
        weights=(weight * dx))[0]
    sumWeights_y = np.histogram(
        index,
        bins=totalBinGridSize,
        range=(-0.5, totalBinGridSize + 0.5),
        weights=(weight * dy))[0]
    sumWeights = np.histogram(
        index,
        bins=totalBinGridSize,
        range=(-0.5, totalBinGridSize + 0.5),
        weights=weight)[0]
    
    # Smallest weight that we'd want to plot a point for
    minWeight = (maxErr**-2.).to(u.mas**-2).value
    
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
    x = np.arange(totalBinGridSize ,dtype=int)
    x = (((x % binGridSize_x) + 0.5) * cellSize) + min_x
    y = np.arange(totalBinGridSize ,dtype=int)
    y = (((y // binGridSize_x) + 0.5) * cellSize) + min_y

    # Finally take the relevant points to plot
    usefulIndices = np.logical_and(sumWeights != 0,
                                   sumWeights < (maxErr**2.).value)
    x = x[usefulIndices]
    y = y[usefulIndices]
    dx = sumWeights_x[usefulIndices]*u.mas
    dy = sumWeights_y[usefulIndices]*u.mas
    
    # Calculate rms and noise and print it.
    RMS_x = np.std(sumWeights_x[sumWeights > 0.])*u.mas
    RMS_y = np.std(sumWeights_y[sumWeights > 0.])*u.mas
    noise = np.sqrt(np.mean(sumWeights[sumWeights > 0.]))*u.mas
    
    return x, y, dx, dy, (RMS_x, RMS_y, noise), cellSize

def calcDivCurl(
    x, y, dx, dy
    ):
    """
    Calculate divergence and curl of the given vector field.
    
    Given vector displacement (dx, dy) defined on identical 2d grids, return arrays giving divergence and curl of the vector field. These will have NaN in pixels without useful info.
    
    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies weighted and binned x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies weighted and binned y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies weighted and binned dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies weighted and binned dy vector
    
    Return:
        div -- (np.ndarray) (2dim) curl of the vector field
        curl -- (np.ndarray) (2dim) divergence of the vector field
    """

    # This line has been replaced because sometimes this line happens to be zero and messes everything up. Removing all the points from np.diff(x) that are zero seems to have little to no effect on the resulting plot and helps get rid of this fairly frequent error.
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
    valid[iy,ix] = True
    use = np.logical_and(valid[1:-1, :-2], valid[1:-1, 2:])
    use = np.logical_and(use, valid[:-2, 1:-1])
    use = np.logical_and(use, valid[2:, 1:-1])
    
    dx2d[iy,ix] = dx
    dy2d[iy,ix] = dy
    
    # XXX These lines may be wrong?
    dxdx = dy2d[2:, 1:-1] - dy2d[:-2, 1:-1]
    dydx = dx2d[2:, 1:-1] - dx2d[:-2, 1:-1]
    dxdy = dy2d[1:-1, 2:] - dy2d[1:-1, :-2]
    dydy = dx2d[1:-1, 2:] - dx2d[1:-1, :-2]

    div = np.where(use, dxdx + dydy, np.nan)
    curl = np.where(use, dydx - dxdy, np.nan)

    vardiv = np.sqrt(gbutil.clippedMean(div[div == div], 5.)[1])
    varcurl = np.sqrt(gbutil.clippedMean(curl[div == div], 5.)[1])
    
    return div, curl, vardiv, varcurl

def calcCorrelation(
    x, y, dx, dy,
    rmin=5*u.arcsec, rmax=1.5*u.deg,
    dlogr=0.05
    ):
    """
    Produce angle-averaged 2-point correlation functions of astrometric error.

    Using bute-force pair counting, calculate the angle-averaged 2-point correlation functions (xi_+, xi_-, xi_z^2, xi_x) for the supplied sample of data. See appendix of Bernstein 2017 for more detailed explanation of calculations.

    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector

    Keyword Arguments:
        rmin -- (astropy.units.quantity.Quantity) (scalar) minimum separation radius bin
        rmax -- (astropy.units.quantity.Quantity) (scalar) maximum separation radius bun
        dlogr -- (float) step size between rmin and rmax

    Return:
        logr -- (np.ndarray) mean log(radius) in each bin (deg)
        xiplus -- (np.ndarray) <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2> (mas2)
        ximinus -- (np.ndarray) <vr1 vr2 - vt1 vt2> (mas2)
        xicross -- (np.ndarray) <vr1 vt2 + vt1 vr2> (mas2)
        xiz2 -- (np.ndarray) <vx1 vx2 - vy1 vy2 + 2 i vx1 vy2> (mas2)
        xiE -- (np.ndarray) E-mode correlation
        xiB -- (np.ndarray) B-mode correlation
    """

    # Check that all (x, y, dx, dy, rmin, rmax) are astropy quantity objects
    if not np.all([isinstance(arr, u.quantity.Quantity) 
                   for arr in [x, y, dx, dy, rmin, rmax]]):
        raise TypeError("All input arrays and rmax and rmin must be of type astropy.units.quantity.Quantity.")

    # Check that x, y have the same units
    if x.unit != y.unit:
        raise u.UnitsError(f"x and y arrays must have the same units but are {x.unit} and {y.unit} respectively.")

    # Check that dx, dy have the same units
    if not (dx.unit == dy.unit):
        raise u.UnitsError(f"dx, and dy arrays must have the same units but are {dx.unit} and {dy.unit} respectively.")
        
    # Check that all arrays (x, y, dx, dy, err) are of shape (N,)
    if not (np.all([arr.shape == x.shape for arr in [x, y, dx, dy]])
            and np.all([arr.ndim == 1 for arr in [x, y, dx, dy]])):
        raise TypeError(f"x, y, dx, and dy arrays must be 1 dimensional and the same shape but are {x.shape}, {y.shape}, {dx.shape}, and {dy.shape} respectively.")

    if not isinstance(dlogr, float):
        raise TypeError(f"dlogr must be a float but is {type(dlogr)}.")

    # Make sure everything is in the correct units and then just take the value. Don't feel like integrating units into the calculation.
    rmin = rmin.to(u.deg).value
    rmax = rmax.to(u.deg).value
    x = x.to(u.deg).value
    y = y.to(u.deg).value
    dx = dx.to(u.mas).value
    dy = dy.to(u.mas).value

    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(x))

    # Omit self-pairs
    use = i1!=i2
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
    vec =  dx + 1j*dy
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

def calcCorrelation2D(
    x, y, dx, dy,
    rmax=1*u.deg,
    nBins=250
    ):
    """
    Produce 2d 2-point correlation functions of astrometric error.

    Produce 2d 2-point correlation function of total displacement power for the supplied sample of data. Uses brute-force pair counting.

    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector

    Keyword Arguments:
        rmax -- (astropy.units.quantity.Quantity) (scalar) maximum separation radius bin. Will calculate from -rmax to rmax separation.
        nBins -- (int) final 2d array will be of shape (nBins, nBins)

    Return:
        xiplus -- (np.ndarray) (2dim) 2pt correlation function (mas2). Is symmetric about the origin. Calculated like so: xi_+ - <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2> 
        counts -- (np.ndarray) (1dim) number of pairs in each separation bin
    """
    
    # Check that all (x, y, dx, dy, rmin, rmax) are astropy quantity objects
    if not np.all([isinstance(arr, u.quantity.Quantity) 
                   for arr in [x, y, dx, dy, rmax]]):
        raise TypeError("All input arrays and rmax must be of type astropy.units.quantity.Quantity.")

    # Check that x, y have the same units
    if x.unit != y.unit:
        raise u.UnitsError(f"x and y arrays must have the same units but are {x.unit} and {y.unit} respectively.")

    # Check that dx, dy have the same units
    if not (dx.unit == dy.unit):
        raise u.UnitsError(f"dx, and dy arrays must have the same units but are {dx.unit} and {dy.unit} respectively.")
        
    # Check that all arrays (x, y, dx, dy) are of shape (N,)
    if not (np.all([arr.shape == x.shape for arr in [x, y, dx, dy]])
            and np.all([arr.ndim == 1 for arr in [x, y, dx, dy]])):
        raise TypeError(f"x, y, dx, and dy arrays must be 1 dimensional and the same shape but are {x.shape}, {y.shape}, {dx.shape}, and {dy.shape} respectively.")

    if not isinstance(nBins, int):
        raise TypeError(f"nBins must be int but is {type(nBins)}.")

    # Make sure everything is in the correct units and then just take the value. Don't feel like integrating units into the calculation.
    rmax = rmax.to(u.deg).value
    x = x.to(u.deg).value
    y = y.to(u.deg).value
    dx = dx.to(u.mas).value
    dy = dy.to(u.mas).value

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
    vec =  dx + 1j*dy
    vvec = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram2d(
        xshift,
        yshift,
        bins=nBins,
        range=hrange,
        weights=vvec)[0]
    xiplus /= counts

    # Combine pars
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

