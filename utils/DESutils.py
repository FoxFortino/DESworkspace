import os
import glob

import numpy as np
import astropy.table as tb


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
            tiles = tiles_tab[tiles_tab["EXPNUM"] == exposure]["TILENAME"]
            tiles = np.array(tiles, dtype=str)
            if not len(tiles):
                continue

            for tile in tiles:
                file = str(tile) + "_final.fits"
                file = os.path.join(zoneDir, file)

                if not os.path.isfile(file):
                    break

            else:
                tab0 = os.path.join(zoneDir, tiles[0] + "_final.fits")
                tab0 = tb.Table.read(tab0)
                band = np.unique(tab0[tab0["EXPNUM"] == exposure]["BAND"])[0]
                if band == "Y":
                    continue
                complete_exposures.append(exposure)

        complete_exposures = np.array(complete_exposures, dtype=int)
        complete_exposures = np.delete(
            complete_exposures, np.argmax(complete_exposures == 999999))

    return complete_exposures


def getBand(
    expNum,
    zoneDir="/data3/garyb/tno/y6/zone134",
    # zoneDir="/media/pedro/Data/austinfortino/zone134/",
    tileRef="/home/fortino/DESworkspace/data/expnum_tile.fits.gz",
    # tileRef="/home/austinfortino/DESworkspace/data/expnum_tile.fits.gz"
        ):

    tiles_tab = tb.Table.read(tileRef)
    tile0 = tiles_tab[tiles_tab["EXPNUM"] == expNum]["TILENAME"][0]

    file = os.path.join(zoneDir, tile0 + "_final.fits")
    tab0 = tb.Table.read(file)
    band = np.unique(tab0[tab0["EXPNUM"] == expNum]["BAND"])[0]

    return band


bandDict = {
    'g': [
        361580, 367484, 369801, 369803, 370204,
        370601, 371369, 372064, 372522, 474262,
        474265, 484483, 576863, 576866, 579816,
        676792, 676800, 696547, 791229, 791593,
        791640],
    'r': [
        361577, 362366, 367482, 367488, 369802,
        370200, 370600, 370602, 371368, 474261,
        474264, 484481, 576861, 576865, 579815,
        676791, 676799, 676801, 681166, 689612,
        791184, 791186, 791215],
    'i': [
        361582, 362365, 364210, 364213, 367483,
        369804, 370199, 370609, 371367, 474260,
        474263, 484499, 576862, 576864, 586534,
        680497, 686427, 686457, 689611, 696552,
        788113, 788116, 788117],
    'z': [
        348819, 355303, 364209, 364215, 372006,
        372437, 373245, 374797, 476846, 484482,
        484490, 484491, 573396, 573398, 592152,
        674340, 675645, 686459, 689613, 691478,
        784503, 788112]}
