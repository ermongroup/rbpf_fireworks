#This class is irritatingly unintuitive, but current code uses it.
#Would be nice to fix sometime
class Measurement:
    #a collection of measurements at a single time instance
    def __init__(self, time = -1):
        #self.val is a list of numpy arrays of measurement x, y locations
        #each numpy array has shape (2,)
        self.val = []
        #list of widths of each bounding box
        self.widths = []
        #list of widths of each bounding box        
        self.heights = []
        #list of scores for each individual measurement
        self.scores = []
        self.time = time

        self.ids = [] #save target id's for generating ground truth
#Transition to an idea like this sometime
class BoundingBox:
    #a single bounding box and time stamp, may be a measurement
    #or the estimated position of a target at a certain time
    def __init__(self, x, y, width, height, time, score = None):
        #floats:
        self.x = x #x location of bounding box center
        self.y = y #y location of bounding box center
        self.width = width #width of bounding box
        self.height = height #height of bounding box

        #float
        self.time = time

        #float, score if this bounding box represents a measurement
        #(applicable to object detections)
        self.score = score

class MeasSet:
    #a set of measurements from a single source that occur
    #at the same time
    def __init__(self, time, measurements = []):
        #float, the time when this MeasSet occured
        self.time = time
        #a list of BoundingBoxes, not we are storing a REFERENCE
        #to the list we are given and are not making a copy
        self.measurements = measurements
    def add_measurement(self, measurement):
        self.measurements.append(measurement)
