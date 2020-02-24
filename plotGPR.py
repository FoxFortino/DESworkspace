import os

import GPRutils

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

def AstrometricResiduals(
    x, y, dx, dy, err,
    x2=None, y2=None, dx2=None, dy2=None, err2=None,
    title1="Observed", title2="GPR Applied",
    minPoints=100,
    pixelsPerBin=500,
    maxErr=50*u.mas,
    scale=350*u.mas,
    arrowScale=50*u.mas,
    savePath=None, saveExt="",
    plotShow=True,
    exposure=None
    ):
    """
    Plots the error on each star as a function of sky position.
    
    Makes a 2d vector diagram (quiver plot) where the astrometric errors/residuals for each measurement is plotted as a function of astrometric position. The input data is binned (according to pixelsPerBin) and weighted (according to err).
    
    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err -- (astropy.units.quantity.Quantity) (N,) specifies error for each vector (dx, dy)
        
    Keyword Arguments:
        x2 -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y2 -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx2 -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy2 -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err2 -- (astropy.units.quantity.Quantity) (N,) specifies error for each vector (dx, dy)
        title1 -- (str) Title of the first plot (x, y, etc.)
        title2 -- (str) TItle of the secnod plot (x2, y2, etc.)
        minPoints -- (int) Minimum number of max plots that will be plotted
        pixelsPerBin -- (int) number of pixels that are represented by one bin
        maxErr -- (astropy.units.quantity.Quantity) (scalar) largest error that a binned vector can have and still be plotted. Avoids cluttering the plot with noisy arrows.
        scale -- (astrpoy.units.quantity.Quantity) (scalar) specifies the scale parameter for the quiver plot
        savePath -- (str) specifies the path to save the plot pdf to
        printShow -- (bool) whether or not to print the plots. Useful when making and saving a bunch of plots in a row.

    Returns:
        None
    """
    
    # Check that scale is an astropy quantity object
    if not isinstance(scale, u.quantity.Quantity):
        raise u.TypeError(f"scale must be of type astropy.quantity.Quantity but is {type(scale)}.")
        
    nData = len(x)
    
    # Grab binned and weighted 2d vector diagram
    x, y, dx, dy, errors, cellSize = \
        GPRutils.calcPixelGrid(
            x, y, dx, dy, err,
            minPoints=minPoints,
            pixelsPerBin=pixelsPerBin,
            maxErr=maxErr)
    RMS_x = f"RMS x: {np.round(errors[0].value, 1)} {errors[0].unit}"
    RMS_y = f"RMS y: {np.round(errors[1].value, 1)} {errors[1].unit}"
    noise = f"Noise: {np.round(errors[2].value, 1)} {errors[2].unit}"
    
    if np.all([arr is not None for arr in [x2, y2, dx2, dy2, err2]]):
        x2, y2, dx2, dy2, errors2, cellSize2 = \
            GPRutils.calcPixelGrid(
                x2, y2, dx2, dy2, err2,
                minPoints=minPoints,
                pixelsPerBin=pixelsPerBin,
                maxErr=maxErr)
        RMS_x2 = f"RMS x: {np.round(errors2[0].value, 1)} {errors2[0].unit}"
        RMS_y2 = f"RMS y: {np.round(errors2[1].value, 1)} {errors2[1].unit}"
        noise2 = f"Noise: {np.round(errors2[2].value, 1)} {errors2[2].unit}"
        
        fig, axes = plt.subplots(
            nrows=1, ncols=2,
            sharex=True, sharey=True,
            figsize=(16, 8))
        fig.subplots_adjust(wspace=0)
        
        quiver = axes[0].quiver(
            x.to(u.deg).value,
            y.to(u.deg).value,
            dx.to(u.deg).value,
            dy.to(u.deg).value,
            pivot='middle',
            color='green',
            angles='xy',
            scale_units='xy',
            scale=scale.to(u.deg).value,
            units='x')
        axes[0].text(0.15, 0.92, RMS_x,
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure)
        axes[0].text(0.15, 0.9, RMS_y,
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure)
        axes[0].text(0.15, 0.88, noise,
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure)
        axes[0].set_xlabel("Sky Position (deg)", fontsize=14)
        axes[0].set_ylabel("Sky Position (deg)", fontsize=14)
        axes[0].set_aspect("equal")
        axes[0].grid()
        axes[0].set_title(title1)
        axes[0].quiverkey(
            quiver,
            0.45,
            0.75,
            arrowScale.to(u.deg).value,
            f"{arrowScale.to(u.mas).value} mas",
            coordinates='figure',
            color='red',
            labelpos='N',
            labelcolor='red')

        quiver = axes[1].quiver(
            x2.to(u.deg).value,
            y2.to(u.deg).value,
            dx2.to(u.deg).value,
            dy2.to(u.deg).value,
            pivot='middle',
            color='green',
            angles='xy',
            scale_units='xy',
            scale=scale.to(u.deg).value,
            units='x')
        axes[1].text(0.80, 0.92, RMS_x2,
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure)
        axes[1].text(0.80, 0.90, RMS_y2,
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure)
        axes[1].text(0.80, 0.88, noise2,
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure)
        axes[1].set_xlabel("Sky Position (deg)", fontsize=14)
        axes[1].set_aspect("equal")
        axes[1].grid()
        axes[1].set_title(title2)
        axes[1].quiverkey(
            quiver,
            0.85,
            0.75,
            arrowScale.to(u.deg).value,
            f"{arrowScale.to(u.mas).value} mas",
            coordinates='figure',
            color='red',
            labelpos='N',
            labelcolor='red')
        
        axes[0].text(0.50, 0.90, "Weighted Astrometric Residuals",
                     fontsize=20,
                     transform=fig.transFigure,
                     ha="center")
        
        if exposure is not None:
            axes[0].text(0.49, 0.87, f"Exposure {exposure}",
                         fontsize=10, **{"fontname": "monospace"},
                         transform=fig.transFigure,
                         ha="right")
        axes[0].text(0.51, 0.87, f"{nData} stars",
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure,
                     ha="left")

    else:
        # Make the quiver plot
        fig = plt.figure(figsize=(8, 8))
        quiver = plt.quiver(
            x.to(u.deg).value,
            y.to(u.deg).value,
            dx.to(u.deg).value,
            dy.to(u.deg).value,
            pivot='middle',
            color='green',
            angles='xy',
            scale_units='xy',
            scale=scale.to(u.deg).value,
            units='x')

        # Make the quiver key and label
        quiverkey = plt.quiverkey(
            quiver,
            0.85,
            0.75,
            arrowScale.to(u.deg).value,
            f"{arrowScale.to(u.mas).value} mas",
            coordinates='figure',
            color='red',
            labelpos='N',
            labelcolor='red')
        plt.text(0.15, 0.92, RMS_x,
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure)
        plt.text(0.15, 0.90, RMS_y,
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure)
        plt.text(0.15, 0.88, noise,
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure)
        
        plt.title("Weighted Astrometric Residuals", fontsize=14)
        plt.xlabel("Sky Position (deg)", fontsize=14)
        plt.ylabel("Sky Position (deg)", fontsize=14)

        if exposure is not None:
            plt.text(0.75, 0.92, f"Exposure {exposure}",
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure,
                     ha="left")
        plt.text(0.75, 0.90, f"{nData} stars",
                 fontsize=10, **{"fontname": "monospace"},
                 transform=fig.transFigure,
                 ha="left")
        
        plt.gca().set_aspect('equal')
        plt.grid()

    xyBuffer = 0.05*u.deg
    plt.xlim((np.min(x) - xyBuffer).value, (np.max(x) + xyBuffer).value)
    plt.ylim((np.min(y) - xyBuffer).value, (np.max(y) + xyBuffer).value)

    
    if savePath is not None:
        saveFile = []
        if saveExt is not None:
            saveFile.append(saveExt)
        if np.all([arr is not None for arr in [x2, y2, dx2, dy2, err2]]):
            saveFile.append("compare")
        saveFile.append("AstroRes.pdf")
        saveFile = "".join(saveFile)
        plt.savefig(os.path.join(savePath, saveFile))

    if plotShow:
        plt.show()
    
def DivCurl(
    x, y, dx, dy, err,
    x2=None, y2=None, dx2=None, dy2=None, err2=None,
    title1="Observed", title2="GPR Applied",
    minPoints=100,
    pixelsPerBin=1000,
    maxErr=50*u.mas,
    scale=50,
    savePath=None, saveExt=None,
    plotShow=True,
    exposure=None
    ):
    """ 
    Make 2d divergence and curl plots for the supplied vector fields.

    Vector field is assumed to be samples from a grid. Plots the divergence and curl of two different sets of data. This is often useful when plotting the observed data and also the data after the GPR has been applied.

    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err -- (astropy.units.quantity.Quantity) (N,) specifies error for each vector (dx, dy)
        
    Keyword Arguments:
        x2 -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y2 -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx2 -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy2 -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err2 -- (astropy.units.quantity.Quantity) (N,) specifies error for each vector (dx, dy)
        title1 -- (str) Title of the first plot (x, y, etc.)
        title2 -- (str) TItle of the secnod plot (x2, y2, etc.)
        minPoints -- (int) Minimum number of max plots that will be plotted
        pixelsPerBin -- (int) number of pixels that are represented by one bin
        maxErr -- (astropy.units.quantity.Quantity) (scalar) largest error that a binned vector can have and still be plotted. Avoids cluttering the plot with noisy points.
        scale -- (int) (scalar) dynamic range (-scale, scale) for the imshow plots
        savePath -- (str) specifies the path to save the plot pdf to
        printShow -- (bool) whether or not to print the plots. Useful when making and saving a bunch of plots in a row.

    Returns:
        None
    """

    nData = len(x)
    
    # Calculate pixel grid
    x, y, dx, dy, errors, cellSize = \
        GPRutils.calcPixelGrid(
            x, y, dx, dy, err,
            minPoints=minPoints,
            pixelsPerBin=pixelsPerBin,
            maxErr=maxErr)
    RMS_x = f"RMS x: {np.round(errors[0].value, 1)} {errors[0].unit}"
    RMS_y = f"RMS y: {np.round(errors[1].value, 1)} {errors[1].unit}"
    noise = f"Noise: {np.round(errors[2].value, 1)} {errors[2].unit}"

    # Calculate div and curl
    div, curl, vardiv, varcurl = GPRutils.calcDivCurl(
        x, y, dx, dy)
    
    if np.all([arr is not None for arr in [x2, y2, dx2, dy2, err2]]):
        
        # Calculate second pixel grid
        x2, y2, dx2, dy2, errors2, cellSize2 = \
            GPRutils.calcPixelGrid(
                x2, y2, dx2, dy2, err2,
                minPoints=minPoints,
                pixelsPerBin=pixelsPerBin,
                maxErr=maxErr)
        RMS_x2 = f"RMS x: {np.round(errors2[0].value, 1)} {errors2[0].unit}"
        RMS_y2 = f"RMS y: {np.round(errors2[1].value, 1)} {errors2[1].unit}"
        noise2 = f"Noise: {np.round(errors2[2].value, 1)} {errors2[2].unit}"
        
        # Calculate second div and curl
        div2, curl2, vardiv2, varcurl2 = GPRutils.calcDivCurl(
            x2, y2, dx2, dy2)
        
        # Create plot
        fig, axes = plt.subplots(
            nrows=2, ncols=2,
            sharex=True, sharey=True,
            figsize=(12, 12))
        fig.subplots_adjust(wspace=0.1)
        fig.subplots_adjust(hspace=0)
        
        
        # First divergence plot
        divplot = axes[0, 0].imshow(
            div,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[0, 0].set_title("Divergence", fontsize=14)
        axes[0, 0].text(0.35, 0.8, f"RMS\n{np.round(vardiv, 2)}",
                        fontsize=12,
                        **{"fontname": "monospace"},
                        transform=fig.transFigure)
        axes[0, 0].axis("off")
        
        # First curl plot
        curlplot = axes[0, 1].imshow(
            curl,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[0, 1].set_title("Curl", fontsize=14)
        axes[0, 1].text(0.7, 0.8, f"RMS\n{np.round(varcurl, 2)}",
                        fontsize=12,
                        **{"fontname": "monospace"},
                        transform=fig.transFigure)
        axes[0, 1].axis("off")
        
        # Second divergence plot
        divplot = axes[1, 0].imshow(
            div2,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[1, 0].set_title("Divergence", fontsize=14)
        axes[1, 0].text(0.35, 0.42, f"RMS\n{np.round(vardiv2, 2)}",
                        fontsize=12,
                        **{"fontname": "monospace"},
                        transform=fig.transFigure)
        axes[1, 0].axis("off")
        
        # Second curl plot
        curlplot = axes[1, 1].imshow(
            curl2,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[1, 1].set_title("Curl", fontsize=14)
        axes[1, 1].text(0.7, 0.42, f"RMS\n{np.round(varcurl2, 2)}",
                        fontsize=12,
                        **{"fontname": "monospace"},
                        transform=fig.transFigure)
        axes[1, 1].axis("off")
        
        # Titles and colorbar
        axes[0, 0].text(0.45, 0.85, title1, fontsize=20,
                        transform=fig.transFigure,
                        ha="center")
        axes[1, 0].text(0.45, 0.50, title2, fontsize=20,
                        transform=fig.transFigure,
                        ha="center")
        
        axes[0, 0].text(0.45, 0.90, "Divergence and Curl Fields",
                     fontsize=20,
                     transform=fig.transFigure,
                     ha="center")

        fig.colorbar(divplot, ax=fig.get_axes())
        
        if exposure is not None:
            axes[0, 0].text(0.15, 0.92, f"Exposure {exposure}",
                            fontsize=10, **{"fontname": "monospace"},
                            transform=fig.transFigure,
                            ha="left")
        axes[0, 0].text(0.15, 0.90, f"{nData} stars",
                        fontsize=10, **{"fontname": "monospace"},
                        transform=fig.transFigure,
                        ha="left")
        
    else:
        # Create plot
        fig, axes = plt.subplots(
            nrows=1, ncols=2,
            sharex=True, sharey=True,
            figsize=(12, 6))

        # Divergence plot
        divplot = axes[0].imshow(
            div,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[0].set_title("Divergence", fontsize=14)
        axes[0].text(0.35, 0.7, f"RMS\n{np.round(vardiv, 2)}",
                     fontsize=12,
                     **{"fontname": "monospace"},
                     transform=fig.transFigure)
        axes[0].axis("off")
        
        # Curl plot
        curlplot = axes[1].imshow(
            curl,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[1].set_title("Curl", fontsize=14)
        axes[1].text(0.7, 0.7, f"RMS\n{np.round(varcurl, 2)}",
                     fontsize=12,
                     **{"fontname": "monospace"},
                     transform=fig.transFigure)
        axes[1].axis("off")
        
        fig.colorbar(divplot, ax=fig.get_axes())
        axes[0].text(0.45, 0.80, "Divergence and Curl Fields",
                     fontsize=20,
                     transform=fig.transFigure,
                     ha="center")
        
        if exposure is not None:
            plt.text(0.15, 0.85, f"Exposure {exposure}",
                     fontsize=10, **{"fontname": "monospace"},
                     transform=fig.transFigure,
                     ha="left")
        plt.text(0.15, 0.82, f"{nData} stars",
                 fontsize=10, **{"fontname": "monospace"},
                 transform=fig.transFigure,
                 ha="left")

    if savePath is not None:
        saveFile = []
        if saveExt is not None:
            saveFile.append(saveExt)
        if np.all([arr is not None for arr in [x2, y2, dx2, dy2, err2]]):
            saveFile.append("compare")
        saveFile.append("DivCurl.pdf")
        saveFile = "".join(saveFile)
        plt.savefig(os.path.join(savePath, saveFile))

    if plotShow:
        plt.show()

def Correlation(
    x, y, dx, dy,
    x2=None, y2=None, dx2=None, dy2=None,
    title1="Observed", title2="GPR Applied",
    xiE_ON=True, xiB_ON=True,
    xiplus_ON=False, ximinus_ON=False,
    xicross_ON=False, xiz2_ON=False,
    rmin=5*u.arcsec, rmax=1.5*u.deg, dlogr=0.05,
    ylim=(-50, 500),
    sep=1e-2*u.deg, avgLine=True,
    savePath=None, saveExt=None,
    plotShow=True,
    exposure=None,
    showInfo=True
    ):

    # Calculate correlation functions
    correlations = GPRutils.calcCorrelation(x, y, dx, dy,
                                            rmin=rmin, rmax=rmax,
                                            dlogr=dlogr)
    logr, xiplus, ximinus, xicross, xiz2, xiE, xiB = correlations
    r = np.exp(logr)

    # Calculate the indices to average together for each correlation function
    ind = np.where(r <= sep.to(u.deg).value)[0]

    fig = plt.figure(figsize=(8, 10))
    plt.title("Angle Averaged 2-Point Correlation Function of Astrometric Residuals")
    plt.xlabel("Separation (deg)")
    plt.ylabel("Correlation (mas2)")

    avgs = {}
    stds = {}
    
    # Check for second set of data:
    if np.all([arr is not None for arr in [x2, y2, dx2, dy2]]):
        data2 = True
        correlations2 = GPRutils.calcCorrelation(x2, y2, dx2, dy2,
                                                 rmin=rmin, rmax=rmax,
                                                 dlogr=dlogr)
        logr2, xiplus2, ximinus2, xicross2, xiz22, xiE2, xiB2 = correlations2
        r2 = np.exp(logr2)
        assert np.allclose(r, r2, equal_nan=True), "r and r2 are not the same"
        
        avgs2 = {}
        stds2 = {}
        ratios = {}
    else:
        data2 = False
        
    corrData = {}
    corrTypes = []
    if xiE_ON:
        corrType = "E-Mode"
        corrTypes.append(corrType)
        corrData[corrType] = [xiE, "r"]
        if data2:
            corrData[corrType].append(xiE2)
    if xiB_ON:
        corrType = "B-Mode"
        corrTypes.append(corrType)
        corrData[corrType] = [xiB, "b"]
        if data2:
            corrData[corrType].append(xiB2)
    if xiplus_ON:
        corrType = "xi_+"
        corrTypes.append(corrType)
        corrData[corrType] = [xiplus, "g"]
        if data2:
            corrData[corrType].append(xiplus2)
    if ximinus_ON:
        corrType = "xi_-"
        corrTypes.append(corrType)
        corrData[corrType] = [ximinus, "c"]
        if data2:
            corrData[corrType].append(ximinus2)
    if xicross_ON:
        corrType = "xi_x"
        corrTypes.append(corrType)
        corrData[corrType] = [xicross, "m"]
        if data2:
            corrData[corrType].append(xicross2)
    if xiz2_ON:
        corrType = "xi_z^2"
        corrTypes.append(corrType)
        corrData[corrType] = [xiz2, "y"]
        if data2:
            corrData[corrType].append(xiz22)
            
            
    def plotCorr(r, data, color, marker, title, corr):
        plt.semilogx(
            r,
            data,
            f"{color}{marker}",
            label=f"{corr} {title}",
            alpha=0.5)
            
    for i, corr in enumerate(corrTypes):
        plotCorr(
            r,
            corrData[corr][0],
            corrData[corr][1],
            ".",
            title1,
            corr)
        avg = np.nanmean(corrData[corr][0][ind])
        std = np.nanstd(corrData[corr][0][ind])
        corrData[corr].append(avg)
        corrData[corr].append(std)
        
        if data2:
            plotCorr(
                r,
                corrData[corr][2],
                corrData[corr][1],
                "o",
                title2,
                corr)
            avg2 = np.nanmean(corrData[corr][2][ind])
            std2 = np.nanstd(corrData[corr][2][ind])
            ratio = np.abs(avg / avg2)
            corrData[corr].append(avg2)
            corrData[corr].append(std2)
            corrData[corr].append(ratio)


    corrTypes.insert(0, " "*20)
    data2_int = int(np.logical_not(data2))
    means = [f"{np.round(corrData[corr][3-data2_int], 3):<10}"
             for corr in corrTypes[1:]]
    stds = [f"{np.round(corrData[corr][4-data2_int], 3):<10}"
            for corr in corrTypes[1:]]
    
    meantitle1 = "Mean " + title1
    means.insert(0, f"{meantitle1:<20}")
    
    stdtitle1 = "Std  " + title1
    stds.insert(0, f"{stdtitle1:<20}")
    
    if data2:
        means2 = [f"{np.round(corrData[corr][5], 3):<10}" 
                  for corr in corrTypes[1:]]
        stds2 = [f"{np.round(corrData[corr][6], 3):<10}"
                 for corr in corrTypes[1:]]
        
        meantitle2 = "Mean " + title2
        means2.insert(0, f"{meantitle2:<20}")
        
        stdtitle2 = "Std  " + title2
        stds2.insert(0, f"{stdtitle2:<20}")
        
        ratios = [f"{np.round(corrData[corr][7], 3):<10}"
                for corr in corrTypes[1:]]
        ratiotitle = "Mean Ratio"
        ratios.insert(0, f"{ratiotitle:<20}")
        
    corrInfo = f"Exposure {exposure}" + "\n"
    corrInfo += f"{len(x)} Stars Evaluated" + "\n"
    corrInfo += f"Stars with separation <= {sep}..." + "\n"
    corrInfo += "".join([f"{corr:<10}" for corr in corrTypes]) + "\n"
    if data2:
        corrInfo += "".join(means) + "\n"
        corrInfo += "".join(means2) + "\n\n"
        corrInfo += "".join(stds) + "\n"
        corrInfo += "".join(stds2) + "\n\n"
        corrInfo += "".join(ratios)
    else:
        corrInfo += "".join(means) + "\n"
        corrInfo += "".join(stds) + "\n"

    if showInfo:
        plt.text(0.05, 0.02, corrInfo,
                 fontsize=10, **{"fontname": "monospace"},
                 transform=fig.transFigure,
                 ha="left")
    

    plt.xlim((rmin.to(u.deg).value, rmax.to(u.deg).value))
    plt.ylim((ylim[0], ylim[1]))
            
    plt.axhline(y=0, c="k")
    if avgLine:
        plt.axvline(x=r[ind[-1]+1], c="k", ls=":")

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    plt.grid()
    plt.legend()
    
    if savePath is not None:
        saveFile = []
        if saveExt is not None:
            saveFile.append(saveExt)
        if np.all([arr is not None for arr in [x2, y2, dx2, dy2]]):
            saveFile.append("compare")
        saveFile.append("2ptCorr.pdf")
        saveFile = "".join(saveFile)
        plt.savefig(os.path.join(savePath, saveFile))

    if plotShow:
        plt.show()

def Correlation2D(
    x, y, dx, dy,
    x2=None, y2=None, dx2=None, dy2=None,
    title1="Observed", title2="GPR Applied", title3="GPR Interpolation",
    rmax=0.3*u.deg, nBins=250,
    vmin=-100*u.mas**2, vmax=450*u.mas**2,
    savePath=None, saveExt=None,
    plotShow=True,
    exposure=None
    ):
    
    nData = len(x)
    if nBins % 2 == 0:
        nBins += 1
    bins = np.around(np.linspace(-rmax, rmax, nBins).to(u.arcmin), 1).value
    ticks = np.array([
        nBins//7, nBins//3,
        nBins//2,
        nBins-nBins//3-1, nBins-nBins//7-1
    ])
    ticklabels = bins[ticks]
    
    xiplus = GPRutils.calcCorrelation2D(
        x, y, dx, dy, rmax=rmax, nBins=nBins)[0]
    
    if np.all([arr is not None for arr in [x2, y2, dx2, dy2]]):
        xiplus2 = GPRutils.calcCorrelation2D(
            x2, y2, dx-dx2, dy-dy2, rmax=rmax, nBins=nBins)[0]
        
        xiplus3 = GPRutils.calcCorrelation2D(
            x2, y2, dx2, dy2, rmax=rmax, nBins=nBins)[0]
        
        # Create plot
        fig, axes = plt.subplots(
            nrows=1, ncols=3,
            sharex=True, sharey=True,
            figsize=(16, 6))
        fig.subplots_adjust(wspace=0)
        axes[0].set_ylabel("Separation (arcmin)")
        axes[0].set_xlabel("Separation (arcmin)")
        axes[1].set_xlabel("Separation (arcmin)")
        axes[2].set_xlabel("Separation (arcmin)")
        
        im1 = axes[0].imshow(
            xiplus,
            origin="Lower",
            cmap="Spectral",
            interpolation="nearest",
            vmin=vmin.to(u.mas**2).value,
            vmax=vmax.to(u.mas**2).value)
        axes[0].set_title(title1)

        
        im2 = axes[1].imshow(
            xiplus2,
            origin="Lower",
            cmap="Spectral",
            interpolation="nearest",
            vmin=vmin.to(u.mas**2).value,
            vmax=vmax.to(u.mas**2).value)
        axes[1].set_title(title3)
        
        im3 = axes[2].imshow(
            xiplus3,
            origin="Lower",
            cmap="Spectral",
            interpolation="nearest",
            vmin=vmin.to(u.mas**2).value,
            vmax=vmax.to(u.mas**2).value)
        axes[2].set_title(title2)
        
        axes[0].set_yticklabels(ticklabels)
        axes[0].set_xticklabels(ticklabels)
        axes[0].set_yticks(ticks)
        axes[0].set_xticks(ticks)
        
    else:
        fig, ax = plt.subplots(
            nrows=1, ncols=1,
            sharex=True, sharey=True,
            figsize=(6, 6))
        fig.subplots_adjust(wspace=0)
        ax.set_ylabel("Separation (arcmin)")
        ax.set_xlabel("Separation (arcmin)")
        
        im1 = ax.imshow(
            xiplus,
            origin="Lower",
            cmap="Spectral",
            interpolation="nearest",
            vmin=vmin.to(u.mas**2).value,
            vmax=vmax.to(u.mas**2).value)
        ax.set_title(title1)
        
        ax.set_yticklabels(ticklabels)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)

    cbar = fig.colorbar(im1, ax=fig.get_axes())
    cbar.set_label("xi_+ Correlation (mas2)", rotation=270)
    

    
    if exposure is not None:
        plt.text(0.1, 0.86, f"Exposure {exposure}",
                 fontsize=10, **{"fontname": "monospace"},
                 transform=fig.transFigure,
                 ha="left")
    plt.text(0.1, 0.83, f"{nData} stars",
             fontsize=10, **{"fontname": "monospace"},
             transform=fig.transFigure,
             ha="left")
    plt.text(0.77, 0.83, f"Max Separation:\n{rmax}",
             fontsize=10, **{"fontname": "monospace"},
             transform=fig.transFigure,
             ha="right")

    if savePath is not None:
        saveFile = []
        if saveExt is not None:
            saveFile.append(saveExt)
        if np.all([arr is not None for arr in [x2, y2, dx2, dy2]]):
            saveFile.append("compare")
        saveFile.append("2ptCorr2D.pdf")
        saveFile = "".join(saveFile)
        plt.savefig(os.path.join(savePath, saveFile))

    if plotShow:
        plt.show()
