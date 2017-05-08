from collections import defaultdict
from collections import deque
from global_params import *

class TargetState:
    #everything that uniquely defines a target at a single instance in time
    def __init__(self, cur_time, id_, measurement):
        '''
        Inputs:
NOT NOW         -SPEC: dictionary, should contain keys:
NOT NOW            *'P'
NOT NOW            *'theta_death'
NOT NOW            *'alpha_death'
NOT NOW            *'BORDER_DEATH_PROBABILITIES'
NOT NOW            *'NOT_BORDER_DEATH_PROBABILITIES'
NOT NOW            *'USE_CONSTANT_R'
NOT NOW            *'R'
NOT NOW            *'Q'
        '''
        assert(measurement != None)
        #target state, [x, x_vel, y, y_vel].T
        #apologies for using x as the traditional Kalman filter state
        #and also as one of the coordinates in image space
        self.x = np.array([[measurement.x], [0], [measurement.y], [0]])

        #error covariance matrix of our estimated target state, self.x
#        self.P = SPEC['P']

        self.width = measurement.width
        self.height = measurement.height

        assert(self.x.shape == (4, 1))
        self.birth_time = cur_time
        #Time of the last measurement data association with this target
        #Or the last time this target produced a measurement for data generation
        self.last_measurement_association = cur_time
        self.id_ = id_ #named id_ to avoid clash with built in id
        self.death_prob = -1 #calculate at every time instance

        #if target's predicted location is offscreen, set to True and then kill
        self.offscreen = False
        self.updated_this_time_instance = True

        #set to false when death is sampled during data generation
        self.alive = True
    def near_border(self):
        near_border = False
        x1 = self.x[0][0] - self.width/2.0 #left edge of bounding box
        x2 = self.x[0][0] + self.width/2.0 #right edge of bounding box
        y1 = self.x[2][0] - self.height/2.0 #top of bounding box, (I think, assuming images are 0 at top)
        y2 = self.x[2][0] + self.height/2.0 #bottom of bounding box, (I think, assuming images are 0 at top)
        if(x1 < 10 or x2 > (CAMERA_PIXEL_WIDTH - 15) or y1 < 10 or y2 > (CAMERA_PIXEL_HEIGHT - 15)):
            near_border = True
        return near_border

    def is_offscreen(self):
        is_offscreen = False
        x1 = self.x[0][0] - self.width/2.0 #left edge of bounding box
        x2 = self.x[0][0] + self.width/2.0 #right edge of bounding box
        y1 = self.x[2][0] - self.height/2.0 #top of bounding box, (I think, assuming images are 0 at top)
        y2 = self.x[2][0] + self.height/2.0 #bottom of bounding box, (I think, assuming images are 0 at top)
        if(x2 < 10 or x1 > (CAMERA_PIXEL_WIDTH - 15) or y2 < 10 or y1 > (CAMERA_PIXEL_HEIGHT - 15)):
            is_offscreen = True
        return is_offscreen


    def target_death_prob(self, cur_time, prev_time, SPEC):
        """ Calculate death probability for this target.

        Input:
        - cur_time: The current measurement time (float)
        - prev_time: The previous time step when a measurement was received (float)
        - SPEC: firework spec, for death probabilities

        Return:
        - death_prob: Probability of target death if this is the only target (float)
        """

        if USE_POISSON_DEATH_MODEL:
            #scipy.special.gdtrc(b, a, x) calculates 
            #integral(gamma_dist(k = a, theta = b))from x to infinity
            last_assoc = self.last_measurement_association
#            if USE_GENERATED_DATA:
            cur_time = cur_time/10.0
            prev_time = prev_time/10.0
            last_assoc = self.last_measurement_association/10.0

#            #I think this is correct
#            death_prob = gdtrc(SPEC['theta_death'], SPEC['alpha_death'], prev_time - last_assoc) \
#                     - gdtrc(SPEC['theta_death'], SPEC['alpha_death'], cur_time - last_assoc)
#            death_prob /= gdtrc(SPEC['theta_death'], SPEC['alpha_death'], prev_time - last_assoc)
#            return death_prob

            #this is used in paper's code
            #Basically this is predicting death over the next time step, as opposed
            #to over the previous time step, which is what I wrote above
            time_step = cur_time - prev_time
            death_prob = gdtrc(SPEC['theta_death'], SPEC['alpha_death'], cur_time - last_assoc) \
                       - gdtrc(SPEC['theta_death'], SPEC['alpha_death'], cur_time - last_assoc + time_step)
            death_prob /= gdtrc(SPEC['theta_death'], SPEC['alpha_death'], cur_time - last_assoc)

            assert(death_prob >= 0.0 and death_prob <= 1.0), (death_prob, cur_time, prev_time)

            return death_prob
        else:
            if(self.offscreen == True):
                cur_death_prob = 1.0
            else:
                frames_since_last_assoc = int(round((cur_time - self.last_measurement_association)/SPEC['time_per_time_step']))
                assert(abs(float(frames_since_last_assoc) - (cur_time - self.last_measurement_association)/SPEC['time_per_time_step']) < .00000001)
                if(self.near_border()):
                    if frames_since_last_assoc < len(SPEC['BORDER_DEATH_PROBABILITIES']):
                        cur_death_prob = SPEC['BORDER_DEATH_PROBABILITIES'][frames_since_last_assoc]
                    else:
                        cur_death_prob = SPEC['BORDER_DEATH_PROBABILITIES'][-1]
    #                   cur_death_prob = 1.0
                else:
                    if frames_since_last_assoc < len(SPEC['NOT_BORDER_DEATH_PROBABILITIES']):
                        cur_death_prob = SPEC['NOT_BORDER_DEATH_PROBABILITIES'][frames_since_last_assoc]
                    else:
                        cur_death_prob = SPEC['NOT_BORDER_DEATH_PROBABILITIES'][-1]
    #                   cur_death_prob = 1.0

            assert(cur_death_prob >= 0.0 and cur_death_prob <= 1.0), cur_death_prob
            return cur_death_prob

    #################### Inference Methods ####################
#########    def kf_update(self, measurement, meas_noise_cov, SPEC):
#########        """ Perform Kalman filter update step and replace predicted position for the current time step
#########        with the updated position in self.all_states
#########        Input:
#########            - measurement: the measurement (numpy array)
#########            - cur_time: time when the measurement was taken (float)
#########            - SPEC: fireworks spec with any extras we need, clean this up sometime
#########        Output:
#########            -updated_x: updated state, numpy array with dimensions (4,1)
#########            -updated_P: updated covariance, numpy array with dimensions (4,4)
#########
#########!!!!!!!!!PREDICTION HAS BEEN RUN AT THE BEGINNING OF TIME STEP FOR EVERY TARGET!!!!!!!!!
#########        """
#########        if SPEC['USE_CONSTANT_R']:
#########            S = np.dot(np.dot(H, self.P), H.T) + SPEC['R']
#########        else:
#########            S = np.dot(np.dot(H, self.P), H.T) + meas_noise_cov
#########        K = np.dot(np.dot(self.P, H.T), inv(S))
#########        residual = measurement - np.dot(H, self.x)
#########        updated_x = self.x + np.dot(K, residual)
#########    #   updated_self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, H)), self.P) #NUMERICALLY UNSTABLE!!!!!!!!
#########        updated_P = self.P - np.dot(np.dot(K, S), K.T) #not sure if this is numerically stable!!
#########        assert(updated_P[0][0] > 0 and
#########               updated_P[1][1] > 0 and
#########               updated_P[2][2] > 0 and
#########               updated_P[3][3] > 0), (self.P, SPEC['R'], SPEC['USE_CONSTANT_R'], meas_noise_cov, K, updated_P)
##########        print "kf_update called :)"
#########        return (updated_x, updated_P)
#########
#########
#########    def update(self, measurement, cur_time, meas_noise_cov):
#########        """ Perform update step and replace predicted position for the current time step
#########        with the updated position in self.all_states
#########        Input:
#########        - measurement: the measurement (numpy array)
#########        - cur_time: time when the measurement was taken (float)
#########!!!!!!!!!PREDICTION HAS BEEN RUN AT THE BEGINNING OF TIME STEP FOR EVERY TARGET!!!!!!!!!
#########        """        
#########        reformat_meas = np.array([[measurement.x],
#########                                  [measurement.y]])
#########        assert(self.x.shape == (4, 1))
#########
#########        (self.x, self.P) = self.kf_update(reformat_meas, meas_noise_cov)
#########
#########        assert(self.x.shape == (4, 1))
#########        assert(self.P.shape == (4, 4))
#########
#########        self.width = measurement.width
#########        self.height = measurement.height
#########        assert(self.all_time_stamps[-1] == round(cur_time, 2) and self.all_time_stamps[-2] != round(cur_time, 2))
#########        assert(self.x.shape == (4, 1)), (self.x.shape, np.dot(K, residual).shape)
#########
##########        self.all_states[-1] = (self.x, self.width, self.height)
#########        self.updated_this_time_instance = True
#########        self.last_measurement_association = cur_time        
#########
#########
#########
#########    def kf_predict(self, dt):
#########        """
#########        Run kalman filter prediction on this target
#########        Inputs:
#########            -dt: time step to run prediction on
#########        Output:
#########            -x_predict: predicted state, numpy array with dimensions (4,1)
#########            -P_predict: predicted covariance, numpy array with dimensions (4,4)
#########
#########        """
#########        F = np.array([[1.0,  dt, 0.0, 0.0],
#########                      [0.0, 1.0, 0.0, 0.0],
#########                      [0.0, 0.0, 1.0,  dt],
#########                      [0.0, 0.0, 0.0, 1.0]])
#########        x_predict = np.dot(F, self.x)
#########        P_predict = np.dot(np.dot(F, self.P), F.T) + SPEC['Q']
#########        assert(P_predict[0][0] > 0 and
#########               P_predict[1][1] > 0 and
#########               P_predict[2][2] > 0 and
#########               P_predict[3][3] > 0), (self.P, SPEC['Q'], P_predict[0][0])
##########        print "kf_predict called :)"
#########
#########        return (x_predict, P_predict)
#########
#########    def predict(self, dt, cur_time):
#########        """
#########        Run prediction on this target
#########        Inputs:
#########            -dt: time step to run prediction on
#########            -cur_time: the time the prediction is made for
#########        """
#########        assert(self.all_time_stamps[-1] == round((cur_time - dt), 2))
#########        (self.x, self.P) = self.kf_predict(dt)
#########
#########        assert(self.x.shape == (4, 1))
#########        assert(self.P.shape == (4, 4))
#########
##########        self.all_states.append((self.x, self.width, self.height))
##########        self.all_time_stamps.append(round(cur_time, 2))
#########
##########        if(self.x[0][0]<0 or self.x[0][0]>=CAMERA_PIXEL_WIDTH or \
##########           self.x[2][0]<0 or self.x[2][0]>=CAMERA_PIXEL_HEIGHT):
#########        self.offscreen = self.is_offscreen()
#########
#########        self.updated_this_time_instance = False
#########
#########
    #################### Data Generation Methods ####################

    def move(self, dt, process_noise):
        """
        Update target state according to the linear motion model of the
        target's state over the specified time interval plus noise

        Leaves bounding box size unchanged, consider changing in the future
        Inputs:
            - dt: float, time step movement corresponds to
            - process_noise: numpy array (4x4), add noise to the new target state drawn
                from a Gaussian with this covariance
        Output:
            - none, but update the target's state and whether the target is offscreen
        """
        F = np.array([[1.0,  dt, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0,  dt],
                      [0.0, 0.0, 0.0, 1.0]])
        noise = np.random.multivariate_normal([0,0,0,0], process_noise)
        noise.shape = (-1, 1) #reshape to be a column vector
        assert(noise.shape == self.x.shape)
        self.x = np.dot(F, self.x) + noise
        self.offscreen = self.is_offscreen()

    def sample_measurement(self, meas_noise, cur_time):
        """
        Sample a measurement for this target and update last_measurement_association to
        the current time

        Inputs:
            - cur_time: float, time measurement is produced
            - meas_noise: numpy array (2x2), add noise to the measurement drawn
                from a Gaussian with this covariance

        Output:
            - sampled measurement (numpy array with shape (2,))
        """
        self.last_measurement_association = cur_time
        sampled_noise = np.random.multivariate_normal([0,0], meas_noise)
        true_position = np.dot(H, self.x).reshape(-1)
        measurement = true_position + sampled_noise
        return measurement

class TargetTrack:
    #A sequence of positions for a single target

    def __init__(self, parent_track = None):
        #parent_track has type TargetTrack.  This target track
        #really has parent_track's positions prepended to
        #it's positions, but we don't actually copy them for efficiency.
        #This is a REFERENCE to the parent_track
        self.parent_track = parent_track
        #list of type BoundingBox
        self.bounding_boxes = []

    def add_bb(self, bounding_box):
        '''
        Add a bounding box to this TargetTrack
        Inputs:
        - bounding_box: type BoundingBox, the bounding box we are adding
            to this TargetTrack
        '''
        #note this is a REFERENCE to the bounding_box we pass in
        self.bounding_boxes.append(bounding_box)

#class TargetSet:
#    #A sequence of positions for a single target
#
#    def __init__(self, parent_track = None):    
class TargetSet:
    """
    Contains ground truth states for all targets.  Also contains all generated measurements.
    """

    def __init__(self):
        '''

        '''

        #list of type Target containing targets currently alive
        self.living_targets = []
        self.measurements = [] #generated measurements for a generative TargetSet 


    def plot_all_target_locations(self, title):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(self.total_count):
            life = len(self.all_targets[i].all_states) #length of current targets life 
            locations_1D =  [self.all_targets[i].all_states[j][0] for j in range(life)]
            ax.plot(self.all_targets[i].all_time_stamps, locations_1D,
                    '-o', label='Target %d' % i)

        legend = ax.legend(loc='lower left', shadow=True)
        plt.title('%s, unique targets = %d, #targets alive = %d' % \
            (title, self.total_count, self.living_count)) # subplot 211 title

    def plot_generated_measurements(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        time_stamps = [self.measurements[i].time for i in range(len(self.measurements))
                                                 for j in range(len(self.measurements[i].val))]
        locations = [self.measurements[i].val[j][0] for i in range(len(self.measurements))
                                                    for j in range(len(self.measurements[i].val))]
        ax.plot(time_stamps, locations,'o')
        plt.title('Generated Measurements') 


    def write_measurements_to_KITTI_format(self, results_filename, SPEC, gt = False, plot_filename = None, plot_target_locations = False):
        '''
        Inputs:
        - gt: boolean, if true write ground truth target id's for each bounding box
        '''

        x_locations_all_targets = defaultdict(list)
        y_locations_all_targets = defaultdict(list)

        f = open(results_filename, "w")
        for measurement in self.measurements:
            frame_idx = int(round(measurement.time/SPEC['time_per_time_step']))
            for idx in range(len(measurement.val)):
                x_pos = measurement.val[idx][0]
                y_pos = measurement.val[idx][1]
                width = measurement.widths[idx]
                height = measurement.heights[idx]
                if gt:
                    cur_id = measurement.ids[idx]
                else:
                    cur_id = -1
                left = x_pos - width/2.0
                top = y_pos - height/2.0
                right = x_pos + width/2.0
                bottom = y_pos + height/2.0      
                f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                    (frame_idx, cur_id, left, top, right, bottom))

                x_locations_all_targets[cur_id].append(x_pos)
                y_locations_all_targets[cur_id].append(y_pos)

        f.close()

        #plot target locations
        if(plot_target_locations):
            assert(len(x_locations_all_targets) == len(y_locations_all_targets))

            print "plotting target locations, ", len(x_locations_all_targets), " targets"

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            for target_id, x_locations in x_locations_all_targets.iteritems():
                print "target", target_id, "is alive for", len(x_locations), "time instances"
                y_locations = y_locations_all_targets[target_id]
                ax.plot(x_locations, y_locations,
                        '-o', label='Target %d' % target_id)

#           legend = ax.legend(loc='lower left', shadow=True)
#           plt.title('%s, unique targets = %d, #targets alive = %d' % \
#               (title, self.total_count, self.living_count)) # subplot 211 title   
            print "ABOUT TO TRY TO SAVE FIG!!!!!!!!!"   
            fig.savefig(plot_filename)  
            print "CALL MADE TO TRY TO SAVE FIG!!!!!!!!!"   
