import os
import glob

import numpy as np
import astropy.table as tb

from IPython import embed

def findExpNums(
    zoneDir="/media/pedro/Data/austinfortino/zone134/",
    tileRef="/home/austinfortino/DESworkspace/data/expnum_tile.fits.gz"
    ):
    
    if zoneDir == "/media/pedro/Data/austinfortino/zone134/":
        complete_exposures = np.array([
            348819, 355303, 361577, 361580, 361582, 362365, 362366, 364209,
            364210, 364213, 364215, 367482, 367483, 367484, 367488, 369801,
            369802, 369803, 369804, 370199, 370200, 370204, 370600, 370601,
            370602, 370609, 371367, 371368, 371369, 372006, 372064, 372437,
            372522, 373245, 374797, 474260, 474261, 474262, 474263, 474264,
            474265, 476846, 484481, 484482, 484483, 484490, 484491, 484499,
            573396, 573398, 576861, 576862, 576863, 576864, 576865, 576866,
            579815, 579816, 586534, 592152, 674340, 675645, 676791, 676792,
            676799, 676800, 676801, 680497, 681166, 686427, 686457, 686459,
            689611, 689612, 689613, 691478, 696547, 696552, 784503, 788112,
            788113, 788116, 788117, 791184, 791186, 791215, 791229, 791593,
            791640
        ])

    else:
        files = sorted(glob.glob(os.path.join(zoneDir, "*fits")))

        exposures = []
        for file in files:
            fits = tb.Table.read(file)
            exposures.append(np.unique(np.array([fits["EXPNUM"]])))
        exposures = np.sort(np.unique(np.hstack(exposures)))

        tiles_tab = tb.Table.read(tileRef)

        complete_exposures = []
        for exposure in exposures:
            tiles = np.array(tiles_tab[tiles_tab["EXPNUM"] == exposure]["TILENAME"], dtype=str)
            if not len(tiles):
                continue

            for tile in tiles:
                file = str(tile) + "_final.fits"
                file = os.path.join(zoneDir, file)

                if not os.path.isfile(file):
                    break

            else:
                tab0 = tb.Table.read(os.path.join(zoneDir, tiles[0] + "_final.fits"))
                band = np.unique(tab0[tab0["EXPNUM"] == exposure]["BAND"])[0]
                if band == "Y":
                    continue
                complete_exposures.append(exposure)

        complete_exposures = np.array(complete_exposures, dtype=int)
        complete_exposures = np.delete(complete_exposures, np.argmax(complete_exposures == 999999))

    return complete_exposures

def getBand(
    expNum,
    zoneDir="/data3/garyb/tno/y6/zone134",
    #zoneDir="/media/pedro/Data/austinfortino/zone134/",
    tileRef="/home/fortino/DESworkspace/data/expnum_tile.fits.gz",
    #tileRef="/home/austinfortino/DESworkspace/data/expnum_tile.fits.gz"
    ):
    
    tiles_tab = tb.Table.read(tileRef)
    tile0 = tiles_tab[tiles_tab["EXPNUM"] == expNum]["TILENAME"][0]
    
    file = os.path.join(zoneDir, tile0 + "_final.fits")
    tab0 = tb.Table.read(file)
    band = np.unique(tab0[tab0["EXPNUM"] == expNum]["BAND"])[0]
    
    return band