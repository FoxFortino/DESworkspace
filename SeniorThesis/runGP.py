from time import time
import os
import shutil
import sys
import traceback

import GPRutils
import vK2KGPR

import numpy as np
import astropy.units as u

def main(expNum, expFile):
    t0 = time()*u.s

    dataC = GPRutils.dataContainer()
    dataC.load(expNum=expNum)
    dataC.splitData()
    t1 = time()*u.s

    GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=expFile)
    GP.fitCorr()
    t2 = time()*u.s

    GP.fit(GP.opt_result[0])
    t3 = time()*u.s

    dataC.makeMasks(GP)
    t4 = time()*u.s
    
    try:
        GP.optimize()
    except np.linalg.LinAlgError:
        with open(GP.paramFile, mode="a+") as file:
            file.write("LinAlgError:" + "\n")
            file.write(str(GP.dC.params) + "\n")
        return
    finally:
        t5 = time()*u.s
        
    GP.fit(GP.opt_result_GP[0])
    t6 = time()*u.s

    GP.predict(dataC.Xvalid)
    t7 = time()*u.s

    dataC.saveNPZ(expFile)
    tf = time()*u.s

    print(f"Loading and splitting data: {(t1-t0).to(u.hr)}")
    print(f"fitCorr: {(t2-t1).to(u.hr)}")
    print(f"Fitting: {(t3-t2).to(u.hr)}")
    print(f"Sigma clipping: {(t4-t3).to(u.hr)}")
    print(f"Optimizing: {(t5-t4).to(u.hr)}")
    print(f"Fitting 2: {(t6-t5).to(u.hr)}")
    print(f"Final prediction: {(t7-t6).to(u.hr)}")
    print(f"Total modeling time: {(tf-t0).to(u.hr)}")


if __name__=='__main__':
    exps = np.array([
        # 248717,
#         348819, 355303, 361577, 361580, 361582, 362365, 362366, 364209,
#         364210, 364213,
        364215, 367482, 367483, 367484, 367488, 369801,
        369802, 369804, 370199, 370200, 370204, 370601, 370602, 370609,
        371367, 371368, 371369, 372006, 372064, 372437, 372522, 373245,
        # 374797, 474260, 474261, 474262, 474263, 474264, 474265, 476846,
        # 484481, 484482, 484483, 484490, 484491, 484499, 573396, 573398,
        # 576861, 576862, 576863, 576864, 576865, 576866, 579815, 579816,
        # 586534, 592152, 674340, 675645, 676791, 676792, 676799, 676800,
        # 676801, 680497, 681166, 686427, 686457, 686459, 689611, 689612,
        # 689613, 691478, 696547, 696552, 784503, 788112, 788113, 788116,
        # 788117, 791184, 791186, 791215, 791229, 791593, 791640
    ])

    for expNum in exps:    
        expFile = os.path.join("/home/fortino/thesis", str(expNum))
        try:
            os.mkdir(expFile)
        except FileExistsError:
            shutil.rmtree(expFile)
            os.mkdir(expFile)
            
        sys.stderr = open(os.path.join(expFile, "err.err"), "a+")
        sys.stdout = open(os.path.join(expFile, "out.out"), "a+")
        
        try:
            main(expNum, expFile)
        except:
            traceback.print_exc()


