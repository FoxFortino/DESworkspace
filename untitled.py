import gpr

import numpy as np
bad_solns = [467, 473, 477, 478, 480, 483, 486, 488, 496, 497]
exposures = [exp for exp in np.arange(450, 490) if exp not in bad_solns]

total_points = 0
for exp in exposures:
    GP = gpr.GPR('dxdy', npz=f"../exposures/{exp}.npz")
    total_points += GP.nData
    print(GP.nData, total_points)
    
print(total_points)