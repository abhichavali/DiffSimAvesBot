# Computes the lift coefficient CL in hovering flight using a sine-based model.
# Takes angle of attack (radians) and parameters e, f, g, h, converts the angle 
# to degrees, and returns the resulting dimensionless CL. Used in blade-element 
# aerodynamic force calculations.
import numpy as np

def CL_hu(alpha, e, f, g, h):
    alphadeg = alpha * 180 / np.pi
    CL = e + f * np.sin((g * alphadeg - h) * np.pi / 180)
    return CL
