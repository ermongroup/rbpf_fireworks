import numpy as np
CAMERA_PIXEL_WIDTH = 1242
CAMERA_PIXEL_HEIGHT = 375
#Entries in assignment cost matrices that cannot be chosen as associations are set
#to this value or greatershould be big but don't want overflow issues
# when further transforming the matrix, more principled number?
INFEASIBLE_COST = 9999999999999999
USE_POISSON_DEATH_MODEL = False

#measurement area (e.g. image) borders
X_MIN = 0
X_MAX = 1241
Y_MIN = 0
Y_MAX = 374

DEFAULT_TIME_STEP = 0.1 

#measurement function matrix
H = np.array([[1.0,  0.0, 0.0, 0.0],
              [0.0,  0.0, 1.0, 0.0]])  

Q_DEFAULT = np.array([[  60.33442497,  102.95992102,   -5.50458177,   -0.22813535],
                      [ 102.95992102,  179.84877761,  -13.37640528,   -9.70601621],
                      [  -5.50458177,  -13.37640528,    4.56034398,    9.48945108],
                      [  -0.22813535,   -9.70601621,    9.48945108,   22.32984314]])
