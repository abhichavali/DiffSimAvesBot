import numpy as np

#Computes the drag coefficient CD in hovering flight using a cosine-based
#empirical model. Takes angle of attack (radians) and parameters a, b, c, d,
#converts the angle to degrees, and returns the resulting dimensionless CD.
#Used in blade-element aerodynamic force calculations.

def CD_hu(alpha, a, b, c, d):
    # convert radians to degrees
    alphadeg = alpha * 180 / np.pi
    
    # compute drag coefficient
    CD = a - b * np.cos((c * alphadeg - d) * np.pi / 180)
    
    return CD
