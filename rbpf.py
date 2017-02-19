import numpy as np
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
#from fireworks.core.firework import FiretaskBase
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.monte_carlo import stratified_resample
import filterpy

#import matplotlib
#matplotlib.use('Agg')
#uncomment for plotting
#import matplotlib.pyplot as plt


#import matplotlib.cm as cmx
#import matplotlib.colors as colors
from scipy.stats import multivariate_normal
from scipy.stats import gamma
from scipy.special import gdtrc
import random
import copy 
import math
from numpy.linalg import inv
import pickle
import sys
import resource
import errno
from munkres import Munkres
from collections import deque
from collections import defaultdict

from cluster_config import RBPF_HOME_DIRECTORY
sys.path.insert(0, "%sKITTI_helpers" % RBPF_HOME_DIRECTORY)
from learn_params1_local import get_meas_target_set
from learn_params1_local import get_meas_target_sets_lsvm_and_regionlets
from learn_params1_local import get_meas_target_sets_regionlets_general_format
from learn_params1_local import get_meas_target_sets_mscnn_general_format
from learn_params1_local import get_meas_target_sets_mscnn_and_regionlets
from learn_params1_local import get_meas_target_sets_2sources_general
from learn_params1_local import get_meas_target_sets_1sources_general

from learn_params1_local import get_meas_target_sets_general

import cProfile
import time
import os
#from run_experiment import DIRECTORY_OF_ALL_RESULTS
#from run_experiment import CUR_EXPERIMENT_BATCH_NAME
#from run_experiment import SEQUENCES_TO_PROCESS
#from run_experiment import get_description_of_run

#from rbpf_ORIGINAL_sampling import sample_and_reweight
#from rbpf_ORIGINAL_sampling import Parameters
#from rbpf_ORIGINAL_sampling import SCALED



#from gen_data import gen_data
#from gen_data import NUM_GEN_FRAMES
#from gen_data import NOISE_SD

#MOTION options
KF_MOTION = True
LSTM_MOTION = False
KNN_MOTION = False
#one of the above should be true, others false
assert([KF_MOTION, LSTM_MOTION, KNN_MOTION].count(True)==1)
assert([KF_MOTION, LSTM_MOTION, KNN_MOTION].count(False)==2)
LSTM_WINDOW = 3 #number of frames used to make LSTM prediction
KNN_WINDOW = 5 #number of frames used to make KNN prediction
#MIN_LSTM_X_VAR = 40.0/2.0 #if the LSTM predicts x variance less than this value, set to this value
#MIN_LSTM_Y_VAR = 5.0/2.0 #if the LSTM predicts y variance less than this value, set to this value
#MIN_LSTM_X_VAR = .01 #if the LSTM predicts x variance less than this value, set to this value
#MIN_LSTM_Y_VAR = .01 #if the LSTM predicts y variance less than this value, set to this value
#uncomment for using LSTM
if LSTM_MOTION:
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler



DATA_PATH = "%sKITTI_helpers/data" % RBPF_HOME_DIRECTORY


PROFILE = False
USE_GENERATED_DATA = False

PLOT_TARGET_LOCATIONS = False

USE_POISSON_DEATH_MODEL = False
USE_CREATE_CHILD = True #speed up copying during resampling
#Write results of the particle with the largest importance
#weight times current likelihood, double check doing this correctly
FIND_MAX_IMPRT_TIMES_LIKELIHOOD = False 
#if true only update a target with at most one measurement
#(i.e. not regionlets and then lsvm)

RESAMPLE_RATIO = 4.0 #resample when get_eff_num_particles < N_PARTICLES/RESAMPLE_RATIO

DEBUG = False

#if True, save the current max importance weight, whether this is the particles first time as
#the max importance weight particle, and the number of living targets along with every
#line of the results file
SAVE_EXTRA_INFO = True

#(if False bug, using R_default instead of S, check SPEC['USE_CONSTANT_R']
#I'm pretty sure this is actually FIXED, but check out some time)

USE_PYTHON_GAUSSIAN = False 
#default time between succesive measurement time instances (in seconds)
default_time_step = .1 

#For testing why score interval for R are slow
CACHED_LIKELIHOODS = 0
NOT_CACHED_LIKELIHOODS = 0

p_clutter_likelihood = 1.0/float(1242*375)
p_birth_likelihood = 1.0/float(1242*375)

#measurement function matrix
H = np.array([[1.0,  0.0, 0.0, 0.0],
              [0.0,  0.0, 1.0, 0.0]])   

#Gamma distribution parameters for calculating target death probabilities
alpha_death = 2.0
#beta_death = 0.5
beta_death = 1.0
theta_death = 1.0/beta_death

#for only displaying targets older than this
min_target_age = .2

#state parameters, during data generation uniformly sample new targets from range:
min_pos = -5.0
max_pos = 5.0
min_vel = -1.0
max_vel = 1.0

#The maximum allowed distance for a ground truth target and estimated target
#to be associated with each other when calculating MOTA and MOTP
MAX_ASSOCIATION_DIST = 1

CAMERA_PIXEL_WIDTH = 1242
CAMERA_PIXEL_HEIGHT = 375

def get_score_index(score_intervals, score):
    """
    Inputs:
    - score_intervals: a list specifying detection score ranges for which parameters have been specified
    - score: the score of a detection

    Output:
    - index: output the 0 indexed score interval this score falls into
    """

    index = 0
    for i in range(1, len(score_intervals)):
        if(score > score_intervals[i]):
            index += 1
        else:
            break
    assert(score > score_intervals[index]), (score, score_intervals[index], score_intervals[index+1]) 
    if(index < len(score_intervals) - 1):
        assert(score <= score_intervals[index+1]), (score, score_intervals[index], score_intervals[index+1])
    return index


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color




class Target:
    def __init__(self, cur_time, id_, measurement = None, width=-1, height=-1):
#       if measurement is None: #for data generation
#           position = np.random.uniform(min_pos,max_pos)
#           velocity = np.random.uniform(min_vel,max_vel)
#           self.x = np.array([[position], [velocity]])
#           self.P = P_default
#       else:
        assert(measurement != None)
        self.x = np.array([[measurement[0]], [0], [measurement[1]], [0]])
        self.P = SPEC['P']

        self.width = width
        self.height = height

        assert(self.x.shape == (4, 1))
        self.birth_time = cur_time
        #Time of the last measurement data association with this target
        self.last_measurement_association = cur_time
        self.id_ = id_ #named id_ to avoid clash with built in id
        self.death_prob = -1 #calculate at every time instance

        self.all_states = [(self.x, self.width, self.height)]
        self.all_time_stamps = [round(cur_time, 2)]

        self.measurements = []
        self.measurement_time_stamps = []

        #if target's predicted location is offscreen, set to True and then kill
        self.offscreen = False

        self.updated_this_time_instance = True

        #used when SPEC['UPDATE_MULT_MEAS_SIMUL'] = True
        self.associated_measurements = []

    def near_border(self):
        near_border = False
        x1 = self.x[0][0] - self.width/2.0
        x2 = self.x[0][0] + self.width/2.0
        y1 = self.x[2][0] - self.height/2.0
        y2 = self.x[2][0] + self.height/2.0
        if(x1 < 10 or x2 > (CAMERA_PIXEL_WIDTH - 15) or y1 < 10 or y2 > (CAMERA_PIXEL_HEIGHT - 15)):
            near_border = True
        return near_border


    def kf_update(self, measurement, meas_noise_cov):
        """ Perform Kalman filter update step and replace predicted position for the current time step
        with the updated position in self.all_states
        Input:
            - measurement: the measurement (numpy array)
            - cur_time: time when the measurement was taken (float)
        Output:
            -updated_x: updated state, numpy array with dimensions (4,1)
            -updated_P: updated covariance, numpy array with dimensions (4,4)

!!!!!!!!!PREDICTION HAS BEEN RUN AT THE BEGINNING OF TIME STEP FOR EVERY TARGET!!!!!!!!!
        """
        if SPEC['USE_CONSTANT_R']:
            S = np.dot(np.dot(H, self.P), H.T) + SPEC['R']
        else:
            S = np.dot(np.dot(H, self.P), H.T) + meas_noise_cov
        K = np.dot(np.dot(self.P, H.T), inv(S))
        residual = measurement - np.dot(H, self.x)
        updated_x = self.x + np.dot(K, residual)
    #   updated_self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, H)), self.P) #NUMERICALLY UNSTABLE!!!!!!!!
        updated_P = self.P - np.dot(np.dot(K, S), K.T) #not sure if this is numerically stable!!
        assert(updated_P[0][0] > 0 and
               updated_P[1][1] > 0 and
               updated_P[2][2] > 0 and
               updated_P[3][3] > 0), (self.P, SPEC['R'], SPEC['USE_CONSTANT_R'], meas_noise_cov, K)
        print "kf_update called :)"
        return (updated_x, updated_P)

    def update_2meas_simul(self):
        assert(SPEC['UPDATE_MULT_MEAS_SIMUL'] and KF_MOTION)
        assert(len(self.associated_measurements) == 2)
        assert(self.associated_measurements[0]['cur_time'] == self.associated_measurements[1]['cur_time'])

        if SPEC['TREAT_MEAS_INDEP_2']:
            reformat_meas1 = np.array([[self.associated_measurements[0]['meas_loc'][0]],
                                      [self.associated_measurements[0]['meas_loc'][1]]])            
            (self.x, self.P) = self.kf_update(reformat_meas1, JOINT_MEAS_NOISE_COV[0:2, 0:2])

            reformat_meas2 = np.array([[self.associated_measurements[1]['meas_loc'][0]],
                                      [self.associated_measurements[1]['meas_loc'][1]]])            
            (self.x, self.P) = self.kf_update(reformat_meas2, JOINT_MEAS_NOISE_COV[2:4, 2:4])

        else:    
            if SPEC['TREAT_MEAS_INDEP']:
                JOINT_MEAS_NOISE_COV[0:2, 2:4] = np.array([[0,0],[0,0]])
                JOINT_MEAS_NOISE_COV[2:4, 0:2] = np.array([[0,0],[0,0]])


            R_inv = inv(JOINT_MEAS_NOISE_COV)
            R_inv_11 = R_inv[0:2, 0:2]
            R_inv_12 = R_inv[0:2, 2:4]
            R_inv_12_T = R_inv[2:4, 0:2]
            R_inv_22 = R_inv[2:4, 2:4]
            #double check R_inv_12_T is the transpose of R_inv_12
            assert((R_inv_12[0,0] - R_inv_12_T[0,0] < .0000001) and
                   (R_inv_12[0,1] - R_inv_12_T[1,0] < .0000001) and
                   (R_inv_12[1,0] - R_inv_12_T[0,1] < .0000001) and
                   (R_inv_12[1,1] - R_inv_12_T[1,1] < .0000001))

            z_1 = self.associated_measurements[0]['meas_loc']
            z_2 = self.associated_measurements[1]['meas_loc']
#            print z_1.shape
#            print z_2.shape
#            sleep(4)
            A = R_inv_11 + R_inv_12 + R_inv_12_T + R_inv_22
            b = np.dot(z_1, R_inv_11) + np.dot(z_1, R_inv_12) + np.dot(z_2, R_inv_12_T) + np.dot(z_2, R_inv_22)
            combined_z = np.dot(inv(A), b)
            combined_R = inv(A)


            reformat_combined_z = np.array([[combined_z[0]],
                                      [combined_z[1]]])
            assert(self.x.shape == (4, 1))

    #        (self.x, self.P) = self.kf_update(reformat_meas, combined_R)
            (self.x, self.P) = self.kf_update(reformat_combined_z, combined_R)

        assert(self.x.shape == (4, 1))
        assert(self.P.shape == (4, 4))

        self.width = self.associated_measurements[0]['width']
        self.height = self.associated_measurements[0]['height']
        cur_time = self.associated_measurements[0]['cur_time']
        assert(self.all_time_stamps[-1] == round(cur_time, 2) and self.all_time_stamps[-2] != round(cur_time, 2))
        assert(self.x.shape == (4, 1)), (self.x.shape, np.dot(K, residual).shape)

        self.all_states[-1] = (self.x, self.width, self.height)
        self.updated_this_time_instance = True
        self.last_measurement_association = cur_time        


    def update(self, measurement, width, height, cur_time, meas_noise_cov):
        """ Perform update step and replace predicted position for the current time step
        with the updated position in self.all_states
        Input:
        - measurement: the measurement (numpy array)
        - cur_time: time when the measurement was taken (float)
!!!!!!!!!PREDICTION HAS BEEN RUN AT THE BEGINNING OF TIME STEP FOR EVERY TARGET!!!!!!!!!
        """        
        reformat_meas = np.array([[measurement[0]],
                                  [measurement[1]]])
        assert(self.x.shape == (4, 1))

        if KF_MOTION:
            (self.x, self.P) = self.kf_update(reformat_meas, meas_noise_cov)
        elif LSTM_MOTION:
            if(len(self.all_states) <= LSTM_WINDOW):
                (self.x, self.P) = self.kf_update(reformat_meas, meas_noise_cov)
            else:
                self.x = np.array([[measurement[0]],
                                              [-99],
                                   [measurement[1]],
                                              [-99]])
                self.P = np.array([[-99, -99, -99, -99],
                                   [-99, -99, -99, -99],
                                   [-99, -99, -99, -99],
                                   [-99, -99, -99, -99]])
        else:
            assert(KNN_MOTION)
            if(len(self.all_states) <= KNN_WINDOW):
                (self.x, self.P) = self.kf_update(reformat_meas, meas_noise_cov)
            else:
                self.x = np.array([[measurement[0]],
                                              [-99],
                                   [measurement[1]],
                                              [-99]])
                self.P = np.array([[-99, -99, -99, -99],
                                   [-99, -99, -99, -99],
                                   [-99, -99, -99, -99],
                                   [-99, -99, -99, -99]])

        assert(self.x.shape == (4, 1))
        assert(self.P.shape == (4, 4))

        self.width = width
        self.height = height
        assert(self.all_time_stamps[-1] == round(cur_time, 2) and self.all_time_stamps[-2] != round(cur_time, 2))
        assert(self.x.shape == (4, 1)), (self.x.shape, np.dot(K, residual).shape)

        self.all_states[-1] = (self.x, self.width, self.height)
        self.updated_this_time_instance = True
        self.last_measurement_association = cur_time        



    def kf_predict(self, dt):
        """
        Run kalman filter prediction on this target
        Inputs:
            -dt: time step to run prediction on
        Output:
            -x_predict: predicted state, numpy array with dimensions (4,1)
            -P_predict: predicted covariance, numpy array with dimensions (4,4)

        """
        F = np.array([[1.0,  dt, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0,  dt],
                      [0.0, 0.0, 0.0, 1.0]])
        x_predict = np.dot(F, self.x)
        P_predict = np.dot(np.dot(F, self.P), F.T) + SPEC['Q']
        assert(P_predict[0][0] > 0 and
               P_predict[1][1] > 0 and
               P_predict[2][2] > 0 and
               P_predict[3][3] > 0), (self.P, SPEC['Q'], P_predict[0][0])
        print "kf_predict called :)"

        return (x_predict, P_predict)


    def lstm_predict(self):
        """
        Output:
            -state_predict: predicted state, numpy array with dimensions (4,1)
            -P_predict: predicted covariance, numpy array with dimensions (4,4)

        """
        # array of locations:
        #[[x_t,     y_t],
        # [x_t-1, y_t-1],
        # ...
        # [x_t-windowsize+1, y_t-windowsize+1]]
        past_locations = np.zeros(LSTM_WINDOW*2)
        for i in range(LSTM_WINDOW):
            pos = -1 - LSTM_WINDOW +1 
            past_locations[2*i] = self.all_states[pos+i][0][0,0]
            past_locations[2*i+1] = self.all_states[pos+i][0][2,0]

        ##########DAN Begin
        cat = np.concatenate((past_locations, past_locations[:2]))
        past_locations_scaled = scaler.transform(np.matrix(cat).reshape(1,8))
        past_locations_scaled = past_locations_scaled.reshape((8,))[:6]
        prediction = model.predict(past_locations_scaled.reshape(1,3,2))
        unscaled_set = scaler.inverse_transform(np.concatenate((past_locations_scaled, prediction.ravel())).reshape(1,8))
        unscaled_prediction = unscaled_set.ravel()[6:]

        var_prediction = varmodel.predict(np.concatenate((past_locations_scaled,prediction.ravel())).reshape(1,4,2))
        oldvals = np.concatenate((past_locations_scaled,prediction.ravel()))
        newval = np.concatenate((oldvals, var_prediction.ravel()))
        variance_set = varscaler.inverse_transform(newval.reshape(1,10))
        variance_prediction_unscaled = variance_set.ravel()[8:]
        #Fill me in here
        x_predict = unscaled_prediction[0]
        y_predict = unscaled_prediction[1]
        x_var = variance_prediction_unscaled[0]
#        if(x_var < MIN_LSTM_X_VAR):
#            x_var = MIN_LSTM_X_VAR
        y_var = variance_prediction_unscaled[1]
#        if(y_var < MIN_LSTM_Y_VAR):
#            y_var = MIN_LSTM_Y_VAR
        #Dan's edit
        if x_var < 0:
            x_var = 1.2
        if x_var > 100:
            x_var = 100
        if y_var < 0:
            y_var = 2.8
        if y_var > 50:
            y_var = 50

        xy_cov = 0
        ##########DAN End

        state_predict = np.array([[x_predict],
                              [-99],
                              [y_predict],
                              [-99]])
        P_predict = np.array([[x_var, -99, xy_cov, -99],
                              [-99, -99, -99, -99],
                              [xy_cov, -99, y_var, -99],
                              [-99, -99, -99, -99]])
        return (state_predict, P_predict)


    def knn_predict(self):
        """
        Output:
            -state_predict: predicted state, numpy array with dimensions (4,1)
            -P_predict: predicted covariance, numpy array with dimensions (4,4)        
        """
        # array of locations:
        #[[x_t,     y_t],
        # [x_t-1, y_t-1],
        # ...
        # [x_t-windowsize+1, y_t-windowsize+1]]
        past_locations = np.zeros((LSTM_WINDOW,2))
        for i in range(LSTM_WINDOW):
            past_locations[i, 0] = self.all_states[-1-i][0][0,0]
            past_locations[i, 1] = self.all_states[-1-i][0][2,0]

        ##########Philip Begin


        #Fill me in here
        x_predict = -99
        y_predict = -99
        x_var = -99
        y_var = -99
        xy_cov = -99
        ##########Philip End

        state_predict = np.array([[x_predict],
                              [-99],
                              [y_predict],
                              [-99]])
        P_predict = np.array([[x_var, -99, xy_cov, -99],
                              [-99, -99, -99, -99],
                              [xy_cov, -99, y_var, -99],
                              [-99, -99, -99, -99]])
        return (state_predict, P_predict)


    def predict(self, dt, cur_time):
        """
        Run prediction on this target
        Inputs:
            -dt: time step to run prediction on
            -cur_time: the time the prediction is made for
        """
        assert(self.all_time_stamps[-1] == round((cur_time - dt), 2))

        if KF_MOTION:
            (self.x, self.P) = self.kf_predict(dt)
        elif LSTM_MOTION:
            if(len(self.all_states) < LSTM_WINDOW):
                (self.x, self.P) = self.kf_predict(dt)
            else:
                (self.x, self.P) = self.lstm_predict()
        else:
            assert(KNN_MOTION)
            if(len(self.all_states) < KNN_WINDOW):
                (self.x, self.P) = self.kf_predict(dt)
            else:
                (self.x, self.P) = self.knn_predict()

        assert(self.x.shape == (4, 1))
        assert(self.P.shape == (4, 4))

        self.all_states.append((self.x, self.width, self.height))
        self.all_time_stamps.append(round(cur_time, 2))

        if(self.x[0][0]<0 or self.x[0][0]>=CAMERA_PIXEL_WIDTH or \
           self.x[2][0]<0 or self.x[2][0]>=CAMERA_PIXEL_HEIGHT):
#           print '!'*40, "TARGET IS OFFSCREEN", '!'*40
            self.offscreen = True
            if USE_GENERATED_DATA:
                self.offscreen = False

        self.updated_this_time_instance = False
        self.associated_measurements = []


################### def target_death_prob(self, cur_time, prev_time):
###################     """ Calculate the target death probability if this was the only target.
###################     Actual target death probability will be (return_val/number_of_targets)
###################     because we limit ourselves to killing a max of one target per measurement.
###################
###################     Input:
###################     - cur_time: The current measurement time (float)
###################     - prev_time: The previous time step when a measurement was received (float)
###################
###################     Return:
###################     - death_prob: Probability of target death if this is the only target (float)
###################     """
###################
###################     #scipy.special.gdtrc(b, a, x) calculates 
###################     #integral(gamma_dist(k = a, theta = b))from x to infinity
###################     last_assoc = self.last_measurement_association
###################
###################     #I think this is correct
###################     death_prob = gdtrc(theta_death, alpha_death, prev_time - last_assoc) \
###################                - gdtrc(theta_death, alpha_death, cur_time - last_assoc)
###################     death_prob /= gdtrc(theta_death, alpha_death, prev_time - last_assoc)
###################     return death_prob
###################
####################        #this is used in paper's code
####################        time_step = cur_time - prev_time
####################    
####################        death_prob = gdtrc(theta_death, alpha_death, cur_time - last_assoc) \
####################                   - gdtrc(theta_death, alpha_death, cur_time - last_assoc + time_step)
####################        death_prob /= gdtrc(theta_death, alpha_death, cur_time - last_assoc)
####################        return death_prob
    def target_death_prob(self, cur_time, prev_time):
        """ Calculate the target death probability if this was the only target.
        Actual target death probability will be (return_val/number_of_targets)
        because we limit ourselves to killing a max of one target per measurement.

        Input:
        - cur_time: The current measurement time (float)
        - prev_time: The previous time step when a measurement was received (float)

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
#            death_prob = gdtrc(theta_death, alpha_death, prev_time - last_assoc) \
#                     - gdtrc(theta_death, alpha_death, cur_time - last_assoc)
#            death_prob /= gdtrc(theta_death, alpha_death, prev_time - last_assoc)
#            return death_prob

            #this is used in paper's code
            #Basically this is predicting death over the next time step, as opposed
            #to over the previous time step, which is what I wrote above
            time_step = cur_time - prev_time
            death_prob = gdtrc(theta_death, alpha_death, cur_time - last_assoc) \
                       - gdtrc(theta_death, alpha_death, cur_time - last_assoc + time_step)
            death_prob /= gdtrc(theta_death, alpha_death, cur_time - last_assoc)

            assert(death_prob >= 0.0 and death_prob <= 1.0), (death_prob, cur_time, prev_time)

            return death_prob
        else:
            if(self.offscreen == True):
                cur_death_prob = 1.0
            else:
                frames_since_last_assoc = int(round((cur_time - self.last_measurement_association)/default_time_step))
                assert(abs(float(frames_since_last_assoc) - (cur_time - self.last_measurement_association)/default_time_step) < .00000001)
                if(self.near_border()):
                    if frames_since_last_assoc < len(BORDER_DEATH_PROBABILITIES):
                        cur_death_prob = BORDER_DEATH_PROBABILITIES[frames_since_last_assoc]
                    else:
                        cur_death_prob = BORDER_DEATH_PROBABILITIES[-1]
    #                   cur_death_prob = 1.0
                else:
                    if frames_since_last_assoc < len(NOT_BORDER_DEATH_PROBABILITIES):
                        cur_death_prob = NOT_BORDER_DEATH_PROBABILITIES[frames_since_last_assoc]
                    else:
                        cur_death_prob = NOT_BORDER_DEATH_PROBABILITIES[-1]
    #                   cur_death_prob = 1.0

            assert(cur_death_prob >= 0.0 and cur_death_prob <= 1.0), cur_death_prob
            return cur_death_prob

class Measurement:
    #a collection of measurements at a single time instance
    def __init__(self, time = -1):
        #self.val is a list of numpy arrays of measurement x, y locations
        self.val = []
        #list of widths of each bounding box
        self.widths = []
        #list of widths of each bounding box        
        self.heights = []
        #list of scores for each individual measurement
        self.scores = []
        self.time = time

class TargetSet:
    """
    Contains ground truth states for all targets.  Also contains all generated measurements.
    """

    def __init__(self):
        self.living_targets = []
        self.all_targets = [] #alive and dead targets

        self.living_count = 0 #number of living targets
        self.total_count = 0 #number of living targets plus number of dead targets
        self.measurements = [] #generated measurements for a generative TargetSet 

        self.parent_target_set = None 

        self.living_targets_q = deque([-1 for i in range(SPEC['ONLINE_DELAY'])])

    def create_child(self):
        child_target_set = TargetSet()
        child_target_set.parent_target_set = self
        child_target_set.total_count = self.total_count
        child_target_set.living_count = self.living_count
        child_target_set.all_targets = copy.deepcopy(self.living_targets)
        for target in child_target_set.all_targets:
            child_target_set.living_targets.append(target)
        child_target_set.living_targets_q = copy.deepcopy(self.living_targets_q)
        return child_target_set

    def create_new_target(self, measurement, width, height, cur_time):
        if SPEC['RUN_ONLINE']:
            global NEXT_TARGET_ID
            new_target = Target(cur_time, NEXT_TARGET_ID, np.squeeze(measurement), width, height)
            NEXT_TARGET_ID += 1
        else:
            new_target = Target(cur_time, self.total_count, np.squeeze(measurement), width, height)
        self.living_targets.append(new_target)
        self.all_targets.append(new_target)
        self.living_count += 1
        self.total_count += 1
        if not USE_CREATE_CHILD:
            assert(len(self.living_targets) == self.living_count and len(self.all_targets) == self.total_count)


    def kill_target(self, living_target_index):
        """
        Kill target self.living_targets[living_target_index], note that living_target_index
        may not be the target's id_ (or index in all_targets)
        """

        #kf predict was run for this time instance, but the target actually died, so remove the predicted state
        del self.living_targets[living_target_index].all_states[-1]
        del self.living_targets[living_target_index].all_time_stamps[-1]

        del self.living_targets[living_target_index]

        self.living_count -= 1
        if not USE_CREATE_CHILD:
            assert(len(self.living_targets) == self.living_count and len(self.all_targets) == self.total_count)

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

    def collect_ancestral_targets(self, descendant_target_ids=[]):
        """
        Inputs:
        - descendant_target_ids: a list of target ids that exist in the calling child's all_targets list
            (or the all_targets list of a descendant of the calling child)

        Outputs:
        - every_target: every target in this TargetSet's all_targets list and
        #every target in any of this TargetSet's ancestors' all_targets lists that does not
        #appear in the all_targets list of a descendant
        """
        every_target = []
        found_target_ids = descendant_target_ids
        for target in self.all_targets:
            if(not target.id_ in found_target_ids):
                every_target.append(target)
                found_target_ids.append(target.id_)
        if self.parent_target_set == None:
            return every_target
        else:
            ancestral_targets = self.parent_target_set.collect_ancestral_targets(found_target_ids)

        every_target = every_target + ancestral_targets # + operator used to concatenate lists!
        return every_target


    def write_online_results(self, online_results_filename, frame_idx, total_frame_count, extra_info):
        """
        Inputs:
        - extra_info: dictionary containing the particle's importance weight (key 'importance_weight') 
            and boolean whether this is the first time the particle is the max importance weight 
            particle (key 'first_time_as_max_imprt_part')

        """
        if frame_idx == SPEC['ONLINE_DELAY']:
            f = open(online_results_filename, "w") #write over old results if first frame
        else:
            f = open(online_results_filename, "a") #write at end of file

        if SPEC['ONLINE_DELAY'] == 0:
            for target in self.living_targets:
                assert(target.all_time_stamps[-1] == round(frame_idx*default_time_step, 2))
                x_pos = target.all_states[-1][0][0][0]
                y_pos = target.all_states[-1][0][2][0]
                width = target.all_states[-1][1]
                height = target.all_states[-1][2]

                left = x_pos - width/2.0
                top = y_pos - height/2.0
                right = x_pos + width/2.0
                bottom = y_pos + height/2.0      
                if SAVE_EXTRA_INFO:
                    f.write( "%d %d Car -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                        (frame_idx, target.id_, left, top, right, bottom, extra_info['importance_weight'], \
                        extra_info['first_time_as_max_imprt_part'], self.living_count))
                else:
                    f.write( "%d %d Car -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                        (frame_idx, target.id_, left, top, right, bottom))

        else:
            print self.living_targets_q
            (delayed_frame_idx, delayed_liv_targets) = self.living_targets_q[0]
            print delayed_frame_idx
            print delayed_liv_targets
            assert(delayed_frame_idx == frame_idx - SPEC['ONLINE_DELAY']), (delayed_frame_idx, frame_idx, SPEC['ONLINE_DELAY'])
            for target in delayed_liv_targets:
                assert(target.all_time_stamps[-1] == round((frame_idx - SPEC['ONLINE_DELAY'])*default_time_step, 2)), (target.all_time_stamps[-1], frame_idx, SPEC['ONLINE_DELAY'], round((frame_idx - SPEC['ONLINE_DELAY'])*default_time_step, 2))
                x_pos = target.all_states[-1][0][0][0]
                y_pos = target.all_states[-1][0][2][0]
                width = target.all_states[-1][1]
                height = target.all_states[-1][2]

                left = x_pos - width/2.0
                top = y_pos - height/2.0
                right = x_pos + width/2.0
                bottom = y_pos + height/2.0      
                if SAVE_EXTRA_INFO:
                    f.write( "%d %d Car -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                        (frame_idx - SPEC['ONLINE_DELAY'], target.id_, left, top, right, bottom, extra_info['importance_weight'], \
                        extra_info['first_time_as_max_imprt_part'], self.living_count))
                else:
                    f.write( "%d %d Car -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                        (frame_idx - SPEC['ONLINE_DELAY'], target.id_, left, top, right, bottom))

            if frame_idx == total_frame_count - 1:
                q_idx = 1
                for cur_frame_idx in range(frame_idx - SPEC['ONLINE_DELAY'] + 1, total_frame_count - 1):
                    print '-'*20
                    print cur_frame_idx
                    print frame_idx - SPEC['ONLINE_DELAY'] + 1
                    print total_frame_count
                    print q_idx
                    print len(self.living_targets_q)
                    (delayed_frame_idx, delayed_liv_targets) = self.living_targets_q[q_idx]
                    q_idx+=1
                    assert(delayed_frame_idx == cur_frame_idx), (delayed_frame_idx, cur_frame_idx, SPEC['ONLINE_DELAY'])
                    for target in delayed_liv_targets:
                        assert(target.all_time_stamps[-1] == round((cur_frame_idx)*default_time_step, 2))
                        x_pos = target.all_states[-1][0][0][0]
                        y_pos = target.all_states[-1][0][2][0]
                        width = target.all_states[-1][1]
                        height = target.all_states[-1][2]

                        left = x_pos - width/2.0
                        top = y_pos - height/2.0
                        right = x_pos + width/2.0
                        bottom = y_pos + height/2.0      
                        if SAVE_EXTRA_INFO:
                            f.write( "%d %d Car -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                                (cur_frame_idx, target.id_, left, top, right, bottom, extra_info['importance_weight'], \
                                extra_info['first_time_as_max_imprt_part'], self.living_count))
                            print "%d %d Car -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                                (cur_frame_idx, target.id_, left, top, right, bottom, extra_info['importance_weight'], \
                                extra_info['first_time_as_max_imprt_part'], self.living_count)
                        else:
                            f.write( "%d %d Car -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                                (cur_frame_idx, target.id_, left, top, right, bottom))

                print "&&&&&&&&&"
                for target in self.living_targets:
                    assert(target.all_time_stamps[-1] == round(frame_idx*default_time_step, 2))
                    x_pos = target.all_states[-1][0][0][0]
                    y_pos = target.all_states[-1][0][2][0]
                    width = target.all_states[-1][1]
                    height = target.all_states[-1][2]

                    left = x_pos - width/2.0
                    top = y_pos - height/2.0
                    right = x_pos + width/2.0
                    bottom = y_pos + height/2.0      
                    if SAVE_EXTRA_INFO:
                        f.write( "%d %d Car -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                            (frame_idx, target.id_, left, top, right, bottom, extra_info['importance_weight'], \
                            extra_info['first_time_as_max_imprt_part'], self.living_count))
                        print  "%d %d Car -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                            (frame_idx, target.id_, left, top, right, bottom, extra_info['importance_weight'], \
                            extra_info['first_time_as_max_imprt_part'], self.living_count)
                    else:
                        f.write( "%d %d Car -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                            (frame_idx, target.id_, left, top, right, bottom))



    def write_targets_to_KITTI_format(self, num_frames, results_filename, plot_filename):
        x_locations_all_targets = defaultdict(list)
        y_locations_all_targets = defaultdict(list)
        if USE_GENERATED_DATA:
            num_frames = NUM_GEN_FRAMES
        if USE_CREATE_CHILD:
            every_target = self.collect_ancestral_targets()
            f = open(results_filename, "w")
            for frame_idx in range(num_frames):
                timestamp = round(frame_idx*default_time_step, 2)

                for target in every_target:
                    if timestamp in target.all_time_stamps:
                        x_pos = target.all_states[target.all_time_stamps.index(timestamp)][0][0][0]
                        y_pos = target.all_states[target.all_time_stamps.index(timestamp)][0][2][0]
                        width = target.all_states[target.all_time_stamps.index(timestamp)][1]
                        height = target.all_states[target.all_time_stamps.index(timestamp)][2]

                        left = x_pos - width/2.0
                        top = y_pos - height/2.0
                        right = x_pos + width/2.0
                        bottom = y_pos + height/2.0      
                        f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                            (frame_idx, target.id_, left, top, right, bottom))

                        x_locations_all_targets[target.id_].append(x_pos)
                        y_locations_all_targets[target.id_].append(y_pos)

            f.close()

        else:
            f = open(results_filename, "w")
            for frame_idx in range(num_frames):
                timestamp = round(frame_idx*default_time_step, 2)
                for target in self.all_targets:
                    if timestamp in target.all_time_stamps:
                        x_pos = target.all_states[target.all_time_stamps.index(timestamp)][0][0][0]
                        y_pos = target.all_states[target.all_time_stamps.index(timestamp)][0][2][0]
                        width = target.all_states[target.all_time_stamps.index(timestamp)][1]
                        height = target.all_states[target.all_time_stamps.index(timestamp)][2]

                        left = x_pos - width/2.0
                        top = y_pos - height/2.0
                        right = x_pos + width/2.0
                        bottom = y_pos + height/2.0      
                        f.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                            (frame_idx, target.id_, left, top, right, bottom))

                        x_locations_all_targets[target.id_].append(x_pos)
                        y_locations_all_targets[target.id_].append(y_pos)

            f.close()

        #plot target locations
        if(PLOT_TARGET_LOCATIONS):
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


class Particle:
    def __init__(self, id_):
        #Targets tracked by this particle
        self.targets = TargetSet()

        self.importance_weight = 1.0/N_PARTICLES
        self.likelihood_DOUBLE_CHECK_ME = -1
        #cache for memoizing association likelihood computation
        self.assoc_likelihood_cache = {}

        self.id_ = id_ #will be the same as the parent's id when copying in create_child

        self.parent_id = -1

        #for debugging
        self.c_debug = -1
        self.imprt_re_weight_debug = -1
        self.pi_birth_debug = -1
        self.pi_clutter_debug = -1
        self.pi_targets_debug = []

        #bool for debugging, indicating maximum importance weight from previous time instance
        self.max_importance_weight = False 

    def create_child(self):
        global NEXT_PARTICLE_ID
        child_particle = Particle(NEXT_PARTICLE_ID)
        NEXT_PARTICLE_ID += 1
        child_particle.importance_weight = self.importance_weight
        child_particle.targets = self.targets.create_child()
        return child_particle

    def create_new_target(self, measurement, width, height, cur_time):
        self.targets.create_new_target(measurement, width, height, cur_time)

    def update_target_death_probabilities(self, cur_time, prev_time):
        for target in self.targets.living_targets:
            target.death_prob = target.target_death_prob(cur_time, prev_time)

    def debug_target_creation(self):
        print
        print "Particle ", self.id_, "importance distribution:"
        print "pi_birth = ", self.pi_birth_debug, "pi_clutter = ", self.pi_clutter_debug, \
            "pi_targets = ", self.pi_targets_debug
        print "sampled association c = ", self.c_debug, "importance reweighting factor = ", self.imprt_re_weight_debug
        self.plot_all_target_locations()

    def process_meas_grp_assoc(self, birth_value, measurement_association, meas_grp_mean, meas_grp_cov, cur_time):
        """
        - meas_source_index: the index of the measurement source being processed (i.e. in SCORE_INTERVALS)

        """
        #create new target
        if(measurement_association == birth_value):
            self.create_new_target(meas_grp_mean[0:2], meas_grp_mean[2], meas_grp_mean[3], cur_time)
            new_target = True 
        #update the target corresponding to the association we have sampled
        elif((measurement_association >= 0) and (measurement_association < birth_value)):
            self.targets.living_targets[measurement_association].update(meas_grp_mean[0:2], meas_grp_mean[2], \
                            meas_grp_mean[3], cur_time, meas_grp_cov[0:2, 0:2])
        else:
            #otherwise the measurement was associated with clutter
            assert(measurement_association == -1), ("measurement_association = ", measurement_association)


    def process_meas_assoc(self, birth_value, meas_source_index, measurement_associations, measurements, \
        widths, heights, measurement_scores, cur_time):
        """
        - meas_source_index: the index of the measurement source being processed (i.e. in SCORE_INTERVALS)

        """
        for meas_index, meas_assoc in enumerate(measurement_associations):
            #create new target
            if(meas_assoc == birth_value):
                self.create_new_target(measurements[meas_index], widths[meas_index], heights[meas_index], cur_time)
                new_target = True 
            #update the target corresponding to the association we have sampled
            elif((meas_assoc >= 0) and (meas_assoc < birth_value)):
                assert(meas_source_index >= 0 and meas_source_index < len(SCORE_INTERVALS)), (meas_source_index, len(SCORE_INTERVALS), SCORE_INTERVALS)
                assert(meas_index >= 0 and meas_index < len(measurement_scores)), (meas_index, len(measurement_scores), measurement_scores)
                #store measurement association for update after all measurements have been associated
                if SPEC['UPDATE_MULT_MEAS_SIMUL']:
                    score_index = get_score_index(SCORE_INTERVALS[meas_source_index], measurement_scores[meas_index])
                    cur_meas = {'meas_loc': measurements[meas_index], 'width': widths[meas_index], \
                                'height': heights[meas_index], 'cur_time': cur_time,
                                'meas_noise_cov': MEAS_NOISE_COVS[meas_source_index][score_index]}
                    self.targets.living_targets[meas_assoc].associated_measurements.append(cur_meas)
                #update the target corresponding to the association we have sampled right now, unless already updated
                #and we only allow a max of 1 update
                elif not (SPEC['MAX_1_MEAS_UPDATE'] and self.targets.living_targets[meas_assoc].updated_this_time_instance):
                    score_index = get_score_index(SCORE_INTERVALS[meas_source_index], measurement_scores[meas_index])
                    self.targets.living_targets[meas_assoc].update(measurements[meas_index], widths[meas_index], \
                                    heights[meas_index], cur_time, MEAS_NOISE_COVS[meas_source_index][score_index])
            else:
                #otherwise the measurement was associated with clutter
                assert(meas_assoc == -1), ("meas_assoc = ", meas_assoc)

    def update_mult_meas_simultaneously(self):
        """
        If SPEC['UPDATE_MULT_MEAS_SIMUL'] = True, run this after associating all measurements to update all targets
        """
        for target in self.targets.living_targets:
            if len(target.associated_measurements) == 1:
                target.update(target.associated_measurements[0]['meas_loc'], target.associated_measurements[0]['width'], \
                            target.associated_measurements[0]['height'], target.associated_measurements[0]['cur_time'], \
                            target.associated_measurements[0]['meas_noise_cov'])
            elif len(target.associated_measurements) == 2:
                target.update_2meas_simul()
            else:
                #not associated with any measurements
                assert(len(target.associated_measurements) == 0)


    #@profile
    def update_particle_with_measurement(self, cur_time, measurement_lists, widths, heights, measurement_scores, params):
        """
        Input:
        - measurement_lists: a list where measurement_lists[i] is a list of all measurements from the current
            time instance from the ith measurement source (i.e. different object detection algorithms
            or different sensors)
        - measurement_scores: a list where measurement_scores[i] is a list containing scores for every measurement in
            measurement_list[i]
        
        -widths: a list where widths[i] is a list of bounding box widths for the corresponding measurements
        -heights: a list where heights[i] is a list of bounding box heights for the corresponding measurements

        Debugging output:
        - new_target: True if a new target was created
        """
        new_target = False #debugging

        birth_value = self.targets.living_count


        if SPEC['use_general_num_dets'] == True:
            (meas_grp_associations, meas_grp_means, meas_grp_covs, dead_target_indices, imprt_re_weight) = \
            sample_and_reweight(self, measurement_lists,  widths, heights, SPEC['det_names'], \
                cur_time, measurement_scores, params)
            self.importance_weight *= imprt_re_weight #update particle's importance weight            
            assert(len(meas_grp_associations) == len(meas_grp_means) and len(meas_grp_means) == len(meas_grp_covs))
            for meas_grp_idx, meas_grp_assoc in enumerate(meas_grp_associations):
                self.process_meas_grp_assoc(birth_value, meas_grp_assoc, meas_grp_means[meas_grp_idx], meas_grp_covs[meas_grp_idx], cur_time)
        else:
            (measurement_associations, dead_target_indices, imprt_re_weight) = \
                sample_and_reweight(self, measurement_lists, \
                    cur_time, measurement_scores, params)


            assert(len(measurement_associations) == len(measurement_lists))
            assert(imprt_re_weight != 0.0), imprt_re_weight
            self.importance_weight *= imprt_re_weight #update particle's importance weight
            #process measurement associations
            for meas_source_index in range(len(measurement_associations)):
                assert(len(measurement_associations[meas_source_index]) == len(measurement_lists[meas_source_index]) and
                       len(measurement_associations[meas_source_index]) == len(widths[meas_source_index]) and
                       len(measurement_associations[meas_source_index]) == len(heights[meas_source_index]))
                self.process_meas_assoc(birth_value, meas_source_index, measurement_associations[meas_source_index], \
                    measurement_lists[meas_source_index], widths[meas_source_index], heights[meas_source_index], \
                    measurement_scores[meas_source_index], cur_time)
            if SPEC['UPDATE_MULT_MEAS_SIMUL']:
                self.update_mult_meas_simultaneously()

        #process target deaths
        #double check dead_target_indices is sorted
        assert(all([dead_target_indices[i] <= dead_target_indices[i+1] for i in xrange(len(dead_target_indices)-1)]))
        #important to delete larger indices first to preserve values of the remaining indices
        for index in reversed(dead_target_indices):
            self.targets.kill_target(index)

        if SPEC['use_general_num_dets'] == True:
            #checking if something funny is happening
            original_num_targets = birth_value
            num_targets_born = 0
            num_targets_born = meas_grp_associations.count(birth_value)
            num_targets_killed = len(dead_target_indices)
            assert(self.targets.living_count == original_num_targets + num_targets_born - num_targets_killed)
            #done checking if something funny is happening

        else:

            #checking if something funny is happening
            original_num_targets = birth_value
            num_targets_born = 0
            for meas_source_index in range(len(measurement_associations)):
                num_targets_born += measurement_associations[meas_source_index].count(birth_value)
            num_targets_killed = len(dead_target_indices)
            assert(self.targets.living_count == original_num_targets + num_targets_born - num_targets_killed)
            #done checking if something funny is happening

        return new_target

    def plot_all_target_locations(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(self.targets.total_count):
            life = len(self.targets.all_targets[i].all_states) #length of current targets life 
            locations_1D =  [self.targets.all_targets[i].all_states[j][0] for j in range(life)]
            ax.plot(self.targets.all_targets[i].all_time_stamps, locations_1D,
                    '-o', label='Target %d' % i)

        legend = ax.legend(loc='lower left', shadow=True)
        plt.title('Particle %d, Importance Weight = %f, unique targets = %d, #targets alive = %d' % \
            (self.id_, self.importance_weight, self.targets.total_count, self.targets.living_count)) # subplot 211 title
#       plt.show()




def normalize_importance_weights(particle_set):
    normalization_constant = 0.0
    for particle in particle_set:
        normalization_constant += particle.importance_weight
    assert(normalization_constant != 0.0), normalization_constant
    for particle in particle_set:
        particle.importance_weight /= normalization_constant


def perform_resampling(particle_set):
    print "memory used before resampling: %d" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert(len(particle_set) == N_PARTICLES)
    weights = []
    for particle in particle_set:
        weights.append(particle.importance_weight)
    assert(abs(sum(weights) - 1.0) < .0000001)

    new_particles = stratified_resample(weights)
    new_particle_set = []
    for index in new_particles:
        if USE_CREATE_CHILD:
            new_particle_set.append(particle_set[index].create_child())
        else:
            new_particle_set.append(copy.deepcopy(particle_set[index]))
    del particle_set[:]
    for particle in new_particle_set:
        particle.importance_weight = 1.0/N_PARTICLES
        particle_set.append(particle)
    assert(len(particle_set) == N_PARTICLES)
    #testing
    weights = []
    for particle in particle_set:
        weights.append(particle.importance_weight)
        assert(particle.importance_weight == 1.0/N_PARTICLES)
    assert(abs(sum(weights) - 1.0) < .01), sum(weights)
    print "memory used after resampling: %d" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #done testing

def display_target_counts(particle_set, cur_time):
    target_counts = []
    for particle in particle_set:
        target_counts.append(particle.targets.living_count)
    print target_counts

    target_counts = []
    importance_weights = []
    for particle in particle_set:
        cur_target_count = 0
        for target in particle.targets.living_targets:
            if (cur_time - target.birth_time) > min_target_age:
                cur_target_count += 1
        target_counts.append(cur_target_count)
        importance_weights.append(particle.importance_weight)
    print "targets older than ", min_target_age, "seconds: ", target_counts
    print "importance weights ", min_target_age, "filler :", importance_weights


def get_eff_num_particles(particle_set):
    n_eff = 0
    weight_sum = 0
    for particle in particle_set:
        n_eff += particle.importance_weight**2
        weight_sum += particle.importance_weight

    assert(abs(weight_sum - 1.0) < .000001), (weight_sum, n_eff)
    return 1.0/n_eff



def run_rbpf_on_targetset(target_sets, online_results_filename, params):
    """
    Measurement class designed to only have 1 measurement/time instance
    Input:
    - target_sets: a list where target_sets[i] is a TargetSet containing measurements from
        the ith measurement source
    Output:
    - max_weight_target_set: TargetSet from a (could be multiple with equal weight) maximum
        importance weight particle after processing all measurements
    - number_resamplings: the number of times resampling was performed
    """
    particle_set = []
    global NEXT_PARTICLE_ID
    #Create the particle set
    for i in range(0, N_PARTICLES):
        particle_set.append(Particle(NEXT_PARTICLE_ID))
        NEXT_PARTICLE_ID += 1
    prev_time_stamp = -1


    #for displaying results
    time_stamps = []
    positions = []

    iter = 0 # for plotting only occasionally
    number_resamplings = 0

    number_time_instances = len(target_sets[0].measurements)
    for target_set in target_sets:
        assert(len(target_set.measurements) == number_time_instances)


    #the particle with the maximum importance weight on the previous time instance 
    prv_max_weight_particle = None

    for time_instance_index in range(number_time_instances):
        time_stamp = target_sets[0].measurements[time_instance_index].time
        for target_set in target_sets:
            assert(target_set.measurements[time_instance_index].time == time_stamp)

        measurement_lists = []
        widths = []
        heights = []
        measurement_scores = []
        for target_set in target_sets:
            measurement_lists.append(target_set.measurements[time_instance_index].val)
            widths.append(target_set.measurements[time_instance_index].widths)
            heights.append(target_set.measurements[time_instance_index].heights)
            measurement_scores.append(target_set.measurements[time_instance_index].scores)

        print "time_stamp = ", time_stamp, "living target count in first particle = ",\
        particle_set[0].targets.living_count
        print "number of measurements from source 0:", len(measurement_lists[0])

        for particle in particle_set:
            #update particle death probabilities
            if(prev_time_stamp != -1):
                particle.assoc_likelihood_cache = {} #clear likelihood cache
                #Run Kalman filter prediction for all living targets
                for target in particle.targets.living_targets:
                    dt = time_stamp - prev_time_stamp
                    assert(abs(dt - default_time_step) < .00000001), (dt, default_time_step, time_stamp, prev_time_stamp)
                    target.predict(dt, time_stamp)
                #update particle death probabilities AFTER predict so that targets that moved
                #off screen this time instance will be killed
                particle.update_target_death_probabilities(time_stamp, prev_time_stamp)

        new_target_list = [] #for debugging, list of booleans whether each particle created a new target
        pIdxDebugInfo = 0
        for particle in particle_set:
            #this is where 
            new_target = particle.update_particle_with_measurement(time_stamp, measurement_lists, widths, heights, measurement_scores, params)
            new_target_list.append(new_target)
            pIdxDebugInfo += 1

        print "about to normalize importance weights"
        normalize_importance_weights(particle_set)
        #debugging
        if DEBUG:
            assert(len(new_target_list) == N_PARTICLES)
            for (particle_number, new_target) in enumerate(new_target_list):
                if new_target:
                    print "\n\n -------Particle %d created a new target-------" % particle_number
                    for particle in particle_set:
                        particle.debug_target_creation()
                    plt.show()
                    break
        #done debugging



        if SPEC['RUN_ONLINE']:
            if time_instance_index >= SPEC['ONLINE_DELAY']:
                #find the particle that currently has the largest importance weight

                if FIND_MAX_IMPRT_TIMES_LIKELIHOOD:
                    max_weight = -1
                    for particle in particle_set:
                        if(particle.importance_weight*particle.likelihood_DOUBLE_CHECK_ME > max_weight):
                            max_weight = particle.importance_weight*particle.likelihood_DOUBLE_CHECK_ME
                    cur_max_weight_target_set = None
                    cur_max_weight_particle = None
                    for particle in particle_set:
                        if(particle.importance_weight*particle.likelihood_DOUBLE_CHECK_ME == max_weight):
                            cur_max_weight_target_set = particle.targets        
                            cur_max_weight_particle = particle
                    print "max weight particle id = ", cur_max_weight_particle.id_

                else:
                    max_imprt_weight = -1
                    for particle in particle_set:
                        if(particle.importance_weight > max_imprt_weight):
                            max_imprt_weight = particle.importance_weight
                    cur_max_weight_target_set = None
                    cur_max_weight_particle = None
                    for particle in particle_set:
                        if(particle.importance_weight == max_imprt_weight):
                            cur_max_weight_target_set = particle.targets        
                            cur_max_weight_particle = particle
                    print "max weight particle id = ", cur_max_weight_particle.id_


            if prv_max_weight_particle != None and prv_max_weight_particle != cur_max_weight_particle:
                if SPEC['ONLINE_DELAY'] == 0:
                    (target_associations, duplicate_ids) = match_target_ids(cur_max_weight_target_set.living_targets,\
                                                           prv_max_weight_particle.targets.living_targets)
                    #replace associated target IDs with the IDs from the previous maximum importance weight
                    #particle for ID conistency in the online results we output
                    for cur_target in cur_max_weight_target_set.living_targets:
                        if cur_target.id_ in target_associations:
                            cur_target.id_ = target_associations[cur_target.id_]
                elif time_instance_index >= SPEC['ONLINE_DELAY']:
                    (target_associations, duplicate_ids) = match_target_ids(cur_max_weight_target_set.living_targets_q[0][1],\
                                                           prv_max_weight_particle.targets.living_targets_q[0][1])
                    #replace associated target IDs with the IDs from the previous maximum importance weight
                    #particle for ID conistency in the online results we output
                    for q_idx in range(SPEC['ONLINE_DELAY']):
                        for cur_target in cur_max_weight_target_set.living_targets_q[q_idx][1]:
                            if cur_target.id_ in duplicate_ids:
                                cur_target.id_ = duplicate_ids[cur_target.id_]
                            if cur_target.id_ in target_associations:
                                cur_target.id_ = target_associations[cur_target.id_]
                    for cur_target in cur_max_weight_target_set.living_targets:
                        if cur_target.id_ in duplicate_ids:
                            cur_target.id_ = duplicate_ids[cur_target.id_]                      
                        if cur_target.id_ in target_associations:
                            cur_target.id_ = target_associations[cur_target.id_]



            #write current time step's results to results file
            if time_instance_index >= SPEC['ONLINE_DELAY']:
                extra_info = {'importance_weight': max_imprt_weight,
                          'first_time_as_max_imprt_part': prv_max_weight_particle != cur_max_weight_particle}
                cur_max_weight_target_set.write_online_results(online_results_filename, time_instance_index, number_time_instances,
                                            extra_info)

            if time_instance_index >= SPEC['ONLINE_DELAY']:
                prv_max_weight_particle = cur_max_weight_particle


            if SPEC['ONLINE_DELAY'] != 0:
                print "popped on time_instance_index", time_instance_index
                for particle in particle_set:
                    particle.targets.living_targets_q.popleft()

                for particle in particle_set:
                    particle.targets.living_targets_q.append((time_instance_index, copy.deepcopy(particle.targets.living_targets)))
        
        if (get_eff_num_particles(particle_set) < N_PARTICLES/RESAMPLE_RATIO):

            perform_resampling(particle_set)
            print "resampled on iter: ", iter
            number_resamplings += 1
        prev_time_stamp = time_stamp

        iter+=1
        print "finished the time step"
#DEBUGGING        
        cur_max_imprt_weight = -1
        for particle in particle_set:
            particle.max_importance_weight = False
            if(particle.importance_weight > cur_max_imprt_weight):
                cur_max_imprt_weight = particle.importance_weight
        particle_count_with_max_imprt_weight = 0
        for particle in particle_set:
            if(particle.importance_weight == cur_max_imprt_weight):
                particle.max_importance_weight = True
                particle_count_with_max_imprt_weight += 1
        print particle_count_with_max_imprt_weight, "particles have max importance weight of", cur_max_imprt_weight
        max_imprt_weight_count_dict[particle_count_with_max_imprt_weight] += 1
#END DEBUGGING

    max_imprt_weight = -1
    for particle in particle_set:
        if(particle.importance_weight > max_imprt_weight):
            max_imprt_weight = particle.importance_weight
    max_weight_target_set = None
    for particle in particle_set:
        if(particle.importance_weight == max_imprt_weight):
            max_weight_target_set = particle.targets

    run_info = [number_resamplings]
    return (max_weight_target_set, run_info, number_resamplings)


def test_read_write_data_KITTI(target_set):
    """
    Measurement class designed to only have 1 measurement/time instance
    Input:
    - target_set: generated TargetSet containing generated measurements and ground truth
    Output:
    - max_weight_target_set: TargetSet from a (could be multiple with equal weight) maximum
        importance weight particle after processing all measurements
    """
    output_target_set = TargetSet()

    for measurement_set in target_set.measurements:
        time_stamp = measurement_set.time
        measurements = measurement_set.val
        widths = measurement_set.widths
        heights = measurement_set.heights

        for i in range(len(measurements)):
            output_target_set.create_new_target(measurements[i], widths[i], heights[i], time_stamp)

    return output_target_set



def convert_to_clearmetrics_dictionary(target_set, all_time_stamps):
    """
    Convert the locations of a TargetSet to clearmetrics dictionary format

    Input:
    - target_set: TargetSet to be converted

    Output:
    - target_dict: Converted locations in clearmetrics dictionary format
    """
    target_dict = {}
    for target in target_set.all_targets:
        for t in all_time_stamps:
            if target == target_set.all_targets[0]: #this is the first target
                if t in target.all_time_stamps: #target exists at this time
                    target_dict[t] = [target.all_states[target.all_time_stamps.index(t)]]
                else: #target doesn't exit at this time
                    target_dict[t] = [None]
            else: #this isn't the first target
                if t in target.all_time_stamps: #target exists at this time
                    target_dict[t].append(target.all_states[target.all_time_stamps.index(t)])
                else: #target doesn't exit at this time
                    target_dict[t].append(None)
    return target_dict

def calc_tracking_performance(ground_truth_ts, estimated_ts):
    """
    !!I think clearmetrics calculates #mismatches incorrectly, look into more!!
    (has to do with whether a measurement can be mismatched to a target that doesn't exist at the current time)

    Calculate MOTA and MOTP ("Evaluating Multiple Object Tracking Performance:
    The CLEAR MOT Metrics", K. Bernardin and R. Stiefelhagen)

    Inputs:
    - ground_truth_ts: TargetSet containing ground truth target locations
    - estimated_ts: TargetSet containing esimated target locations
    """

    #convert TargetSets to dictionary format for calling clearmetrics

    all_time_stamps = [ground_truth_ts.measurements[i].time for i in range(len(ground_truth_ts.measurements))]
    ground_truth = convert_to_clearmetrics_dictionary(ground_truth_ts, all_time_stamps)
    estimated_tracks = convert_to_clearmetrics_dictionary(estimated_ts, all_time_stamps)

    clear = clearmetrics.ClearMetrics(ground_truth, estimated_tracks, MAX_ASSOCIATION_DIST)
    clear.match_sequence()
    evaluation = [clear.get_mota(),
                  clear.get_motp(),
                  clear.get_fn_count(),
                  clear.get_fp_count(),
                  clear.get_mismatches_count(),
                  clear.get_object_count(),
                  clear.get_matches_count()]
    print 'MOTA, MOTP, FN, FP, mismatches, objects, matches'
    print evaluation     
    ground_truth_ts.plot_all_target_locations("Ground Truth")         
    ground_truth_ts.plot_generated_measurements()    
    estimated_ts.plot_all_target_locations("Estimated Tracks")      
    plt.show()

class KittiTarget:
    """
    Used for computing target associations when outputing online results and the particle with
    the largest importance weight changes

    Values:
    - x1: smaller x coordinate of bounding box
    - x2: larger x coordinate of bounding box
    - y1: smaller y coordinate of bounding box
    - y2: larger y coordinate of bounding box
    """
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

def boxoverlap(a,b):
    """
        Copied from  KITTI devkit_tracking/python/evaluate_tracking.py

        boxoverlap computes intersection over union for bbox a and b in KITTI format.
        If the criterion is 'union', overlap = (a inter b) / a union b).
        If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
    """
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    
    w = x2-x1
    h = y2-y1

    if w<=0. or h<=0.:
        return 0.
    inter = w*h
    aarea = (a.x2-a.x1) * (a.y2-a.y1)
    barea = (b.x2-b.x1) * (b.y2-b.y1)
    # intersection over union overlap
    o = inter / float(aarea+barea-inter)
    return o

def convert_targets(input_targets):
    kitti_format_targets = []
    for cur_target in input_targets:
        x_pos = cur_target.x[0][0]
        y_pos = cur_target.x[2][0]
        width = cur_target.width
        height = cur_target.height

        left = x_pos - width/2.0
        top = y_pos - height/2.0
        right = x_pos + width/2.0
        bottom = y_pos + height/2.0     

        kitti_format_targets.append(KittiTarget(left, right, top, bottom))
    return kitti_format_targets

def match_target_ids(particle1_targets, particle2_targets):
    """
    Use the same association as in  KITTI devkit_tracking/python/evaluate_tracking.py

    Inputs:
    - particle1_targets: a list of targets from particle1
    - particle2_targets: a list of targets from particle2

    Output:
    - associations: a dictionary of associations between targets in particle1 and particle2.  
        associations[particle1_targetID] = particle2_targetID where particle1_targetID and
        particle2_targetID are IDs of associated targets
    """
    kitti_targets1 = convert_targets(particle1_targets)
    kitti_targets2 = convert_targets(particle2_targets)

    #if any targets in particle1 have the same ID as a target in particle2,
    #assign the particle1 target a new ID
    duplicate_ids = {}
    global NEXT_TARGET_ID
    p2_target_ids = []
    for cur_t2 in particle2_targets:
        p2_target_ids.append(cur_t2.id_)
    for cur_t1 in particle1_targets:
        if cur_t1.id_ in p2_target_ids:
            duplicate_ids[cur_t1.id_] = NEXT_TARGET_ID
            cur_t1.id_ = NEXT_TARGET_ID
            NEXT_TARGET_ID += 1

    hm = Munkres()
    max_cost = 1e9
    cost_matrix = []
    for cur_t1 in kitti_targets1:
        cost_row = []
        for cur_t2 in kitti_targets2:
            # overlap == 1 is cost ==0
            c = 1-boxoverlap(cur_t1,cur_t2)
            # gating for boxoverlap
            if c<=.5:
                cost_row.append(c)
            else:
                cost_row.append(max_cost)
        cost_matrix.append(cost_row)

    if len(kitti_targets1) is 0:
        cost_matrix=[[]]

    # associate
    association_matrix = hm.compute(cost_matrix)
    associations = {}
    for row,col in association_matrix:
        c = cost_matrix[row][col]
        if c < max_cost:
            associations[particle1_targets[row].id_] = particle2_targets[col].id_

    return (associations, duplicate_ids)




def combine_arbitrary_number_measurements_4d(blocked_cov_inv, meas_noise_mean, gt_obj):
    """
    
    Inputs:
    - blocked_cov_inv: dictionary containing the inverse of the measurement noise covariance matrix, between
    all measurement source

    [sigma_11    sigma_1j     sigma_1n]
    [.       .                        ]
    [.          .                     ]
    [.             .                  ]
    [sigma_i1    sigma_ij     sigma_in]
    [.                 .              ]
    [.                    .           ]
    [.                       .        ]
    [sigma_n1    sigma_nj     sigma_nn]
    
    Where there are n measurement sources and sigma_ij represents the block of the INVERSE of the noise covariance
    corresponding to the ith blocked row and the jth blocked column.  To access sigma_ij, call 
    blocked_cov_inv[('meas_namei','meas_namej')] where 'meas_namei' is the string representation of the name of
    measurement source i.

    -meas_noise_mean: a dictionary where meas_noise_mean['meas_namei'] = the mean measurement noise for measurement
    source with name 'meas_namei'

    -gt_obj: the ground truth object whose associated measurements will be combined, can have an arbitrary number
    of associations

    """
    meas_count = len(gt_obj.assoc_dets) #number of associated measurements

#    #dictionary containing all measurements in appropriately formatted numpy arrays
#    reformatted_zs = {}
#    for det_name, det in gt_obj.assoc_dets.iteritems():
#        cur_z = np.array([det.x - meas_noise_mean[det_name][0], 
#                          det.y - meas_noise_mean[det_name][1],
#                          det.width - meas_noise_mean[det_name][2],
#                          det.height - meas_noise_mean[det_name][3]])
#        reformatted_zs[det_name] = cur_z
#
#    A = 0
#    b = 0
#    for det_name1, det in reformatted_zs.iteritems():
#        for det_name2, ignore_me_det in gt_obj.assoc_dets.iteritems():
#            A += blocked_cov_inv[(det_name1, det_name2)]
#            b += np.dot(det, blocked_cov_inv[(det_name1, det_name2)])
#
#    combined_meas_mean = np.dot(inv(A), b)
#    combined_covariance = inv(A)
#
#    assert(combined_meas_mean.shape == (4,)), (meas_count, gt_obj.assoc_dets)
#    return (combined_meas_mean, combined_covariance)

    #dictionary containing all measurements in appropriately formatted numpy arrays
    reformatted_zs = {}
    for det_name, det in gt_obj.assoc_dets.iteritems():
        cur_z = np.array([det.x - meas_noise_mean[det_name][0], 
                          det.y - meas_noise_mean[det_name][1],
                          det.width - meas_noise_mean[det_name][2],
                          det.height - meas_noise_mean[det_name][3]])
        reformatted_zs[det_name] = cur_z
    A = 0
    b = 0
    for det_name1, det in reformatted_zs.iteritems():
        for det_name2, ignore_me_det in gt_obj.assoc_dets.iteritems():
            A += blocked_cov_inv[(det_name1, det_name2)]
            b += np.dot(det, blocked_cov_inv[(det_name1, det_name2)])
    combined_meas_mean = np.dot(inv(A), b.transpose())
    combined_covariance = inv(A)
    assert(combined_meas_mean.shape == (4,)), (meas_count, gt_obj.assoc_dets)
    return (combined_meas_mean.flatten(), combined_covariance)



@explicit_serialize
class RunRBPF(FireTaskBase):   
 #   _fw_name = "Run RBPF Task"
    def run_task(self, fw_spec):
        #debugging
        global max_imprt_weight_count_dict
        max_imprt_weight_count_dict = defaultdict(int)
        #end debugging

        global NEXT_PARTICLE_ID
        global NEXT_TARGET_ID
        global N_PARTICLES

        #allow the firework spec to be accessed globally
        global SPEC
        SPEC = fw_spec

        SPEC['P'] = np.array(SPEC['P'])
        SPEC['R'] = np.array(SPEC['R'])
        SPEC['Q'] = np.array(SPEC['Q'])

        #Better practice to make these NOT global
        global SCORE_INTERVALS
        global MEAS_NOISE_COVS
        global BORDER_DEATH_PROBABILITIES
        global NOT_BORDER_DEATH_PROBABILITIES
        global JOINT_MEAS_NOISE_COV

        NEXT_PARTICLE_ID = 0
        if SPEC['RUN_ONLINE']:
            NEXT_TARGET_ID = 0 #all targets have unique IDs, even if they are in different particles
    
        #Get run parameters from the firework spec
        N_PARTICLES = fw_spec['num_particles']
        run_idx = fw_spec['run_idx'] #the index of this run
        seq_idx = fw_spec['seq_idx'] #the index of the sequence to process
        #Should ignored ground truth objects be included when calculating probabilities? (double check specifics)
        include_ignored_gt = fw_spec['include_ignored_gt']
        include_dontcare_in_gt = fw_spec['include_dontcare_in_gt']
        det_names = fw_spec['det_names']
        det1_name = det_names[0]
        if len(det_names) > 1:
            det2_name = det_names[1]
        else:
            det2_name = None
        sort_dets_on_intervals = fw_spec['sort_dets_on_intervals']
        results_folder = fw_spec['results_folder']

        derandomize_with_seed = fw_spec['derandomize_with_seed']
        if derandomize_with_seed:
            random.seed(5)
            np.random.seed(seed=5)

        use_general_num_dets = fw_spec['use_general_num_dets']
        global sample_and_reweight
        if use_general_num_dets:
            from rbpf_sampling_manyMeasSrcs import sample_and_reweight
            from rbpf_sampling_manyMeasSrcs import Parameters

        else:
            from rbpf_sampling import sample_and_reweight
            from rbpf_sampling import Parameters
####def run_rbpf(num_particles,run_idx,seq_idx,include_ignored_gt,include_dontcare_in_gt,
####            use_regionlets,det1_name,det2_name,sort_dets_on_intervals):
####    global NEXT_PARTICLE_ID
####    global NEXT_TARGET_ID
####    global N_PARTICLES
####
#### 
####    #Better practice to make these NOT global
####    global SCORE_INTERVALS
####    global MEAS_NOISE_COVS
####    global BORDER_DEATH_PROBABILITIES
####    global NOT_BORDER_DEATH_PROBABILITIES
####    global JOINT_MEAS_NOISE_COV
####        NEXT_PARTICLE_ID = 0
####        if SPEC['RUN_ONLINE']:
####            NEXT_TARGET_ID = 0 #all targets have unique IDs, even if they are in different particles
####        #Get run parameters from the firework spec
####        N_PARTICLES = num_particles
####        run_idx = run_idx #the index of this run
####        seq_idx = seq_idx #the index of the sequence to process
####        #Should ignored ground truth objects be included when calculating probabilities? (double check specifics)
####        include_ignored_gt = include_ignored_gt
####        include_dontcare_in_gt = include_dontcare_in_gt
####        use_regionlets = use_regionlets
####        det1_name = det1_name
####        det2_name = det2_name
####        if(det2_name == 'None'):
####            det2_name = None
####        sort_dets_on_intervals = sort_dets_on_intervals


        #########################
        if LSTM_MOTION:
            # LSTM initialization
            model = Sequential()
            model.add(LSTM(32, input_shape=(3,2)))
            model.add(Dense(2))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.load_weights('/atlas/u/daniter/models/cv-mu-weights%d.h5' % seq_idx)

            varmodel = Sequential()
            varmodel.add(LSTM(32, input_shape=(4,2)))
            varmodel.add(Dense(2))
            varmodel.compile(loss='mean_squared_error', optimizer='adam')
            varmodel.load_weights('/atlas/u/daniter/models/cv-cov-weights%d.h5' % seq_idx)
            scaler = None
            varscaler = None
            with open('/atlas/u/daniter/models/varscaler.pickle', 'r') as handle:
              varscaler = pickle.load(handle)
            with open('/atlas/u/daniter/models/scaler.pickle', 'r') as handle:
              scaler = pickle.load(handle)
        #########################

        filename_mapping = DATA_PATH + "/evaluate_tracking.seqmap"
        n_frames         = []
        sequence_name    = []
        with open(filename_mapping, "r") as fh:
            for i,l in enumerate(fh):
                fields = l.split(" ")
                sequence_name.append("%04d" % int(fields[0]))
                n_frames.append(int(fields[3]) - int(fields[2]))
        fh.close() 
        print n_frames
        print sequence_name     
        assert(len(n_frames) == len(sequence_name))

        print 'begin run'
    #debug
        indicate_run_started_filename = '%s/results_by_run/run_%d/seq_%d_started.txt' % (results_folder, run_idx, seq_idx)
        run_started_f = open(indicate_run_started_filename, 'w')
        run_started_f.write("This run was started\n")
        run_started_f.close()
    #end debug


        indicate_run_complete_filename = '%s/results_by_run/run_%d/seq_%d_done.txt' % (results_folder, run_idx, seq_idx)
        #if we haven't already run, run now:
        if not os.path.isfile(indicate_run_complete_filename):


            #False doesn't really make sense because when actually running without ground truth information we don't know
            #whether or not a detection is ignored, but debugging. (An ignored detection is a detection not associated with
            #a ground truth object that would be associated with a don't care ground truth object if they were included.  It 
            #can also be a neighobring object type, e.g. "van" instead of "car", but this never seems to occur in the data.
            #If this occured, it would make sense to try excluding these detections.)
            include_ignored_detections = True 

            if sort_dets_on_intervals:
                score_interval_dict_all_det = {\
                    'mscnn' : [float(i)*.1 for i in range(3,10)],              
                    'regionlets' : [i for i in range(2, 20)],
                    '3dop' : [float(i)*.1 for i in range(2,10)],            
                    'mono3d' : [float(i)*.1 for i in range(2,10)],            
                    'mv3d' : [float(i)*.1 for i in range(2,10)]}        
    #            'regionlets' = [i for i in range(2, 16)]
            else:
                score_interval_dict_all_det = {\
    #            'mscnn' = [.5],                                
                'mscnn' : [.3],                                
                'regionlets' : [2],
                '3dop' : [.2],
                'mono3d' : [.2],
                'mv3d' : [.2]}

            #train on all training sequences, except the current sequence we are testing on
            training_sequences = [i for i in [i for i in range(21)] if i != seq_idx]

            SCORE_INTERVALS = []
            for det_name in det_names:
                SCORE_INTERVALS.append(score_interval_dict_all_det[det_name])

            if use_general_num_dets:
                #dictionary of score intervals for only detection sources we are using
                SCORE_INTERVALS_DET_USED = {}
                for det_name in det_names:
                    SCORE_INTERVALS_DET_USED[det_name] = score_interval_dict_all_det[det_name]

                (measurementTargetSetsBySequence, target_groupEmission_priors, clutter_grpCountByFrame_priors, clutter_group_priors, 
                birth_count_priors, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES, 
                posAndSize_inv_covariance_blocks, meas_noise_mean, posOnly_covariance_blocks) = get_meas_target_sets_general(
                            training_sequences, SCORE_INTERVALS_DET_USED, det_names, \
                            obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True,\
                            include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
                            include_ignored_detections = include_ignored_detections)

                params = Parameters(det_names, target_groupEmission_priors, clutter_grpCountByFrame_priors,\
                         clutter_group_priors, birth_count_priors, posOnly_covariance_blocks, \
                         meas_noise_mean, posAndSize_inv_covariance_blocks, SPEC['R'], H,\
                         USE_PYTHON_GAUSSIAN, SPEC['USE_CONSTANT_R'], SCORE_INTERVALS,\
                         p_birth_likelihood, p_clutter_likelihood, SPEC['CHECK_K_NEAREST_TARGETS'],
                         SPEC['K_NEAREST_TARGETS'], SPEC['scale_prior_by_meas_orderings'])

#                print "BORDER_DEATH_PROBABILITIES:", BORDER_DEATH_PROBABILITIES
#                print "NOT_BORDER_DEATH_PROBABILITIES:", NOT_BORDER_DEATH_PROBABILITIES
#                sleep(2.5)

            else:
                det1_score_intervals = score_interval_dict_all_det[det1_name]
                if det2_name:
                    det2_score_intervals = score_interval_dict_all_det[det2_name]
                    (measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
                        MEAS_NOISE_COVS, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES, JOINT_MEAS_NOISE_COV) = \
                            get_meas_target_sets_2sources_general(training_sequences, det1_score_intervals, \
                            det2_score_intervals, det1_name, det2_name, obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True,\
                            include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
                            include_ignored_detections = include_ignored_detections)

                else:
                    (measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
                        MEAS_NOISE_COVS, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES) = \
                            get_meas_target_sets_1sources_general(training_sequences, det1_score_intervals, \
                            det1_name, obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True,\
                            include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
                            include_ignored_detections = include_ignored_detections)            

                params = Parameters(TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES,\
                         BIRTH_PROBABILITIES, MEAS_NOISE_COVS, SPEC['R'], H,\
                         USE_PYTHON_GAUSSIAN, SPEC['USE_CONSTANT_R'], SCORE_INTERVALS,\
                         p_birth_likelihood, p_clutter_likelihood, SPEC['CHECK_K_NEAREST_TARGETS'],
                         SPEC['K_NEAREST_TARGETS'], SPEC['scale_prior_by_meas_orderings'], SPEC)

            assert(len(n_frames) == len(measurementTargetSetsBySequence))

            t0 = time.time()
            info_by_run = [] #list of info from each run
            cur_run_info = None
            results_filename = '%s/results_by_run/run_%d/%s.txt' % (results_folder, run_idx, sequence_name[seq_idx])
            plot_filename = '%s/results_by_run/run_%d/%s_plot.png' % (results_folder, run_idx, sequence_name[seq_idx])
            measurements_filename = '%s/results_by_run/run_%d/%s_measurements_plot.png' % (results_folder, run_idx, sequence_name[seq_idx])


            print "Processing sequence: ", seq_idx
            tA = time.time()
            if USE_GENERATED_DATA:
                meas_target_set = gen_data(measurements_filename)
                if PROFILE: 
                    cProfile.run('run_rbpf_on_targetset([meas_target_set], results_filename, params)')
                else:
                    (estimated_ts, cur_seq_info, number_resamplings) = run_rbpf_on_targetset([meas_target_set], results_filename, params)
            else:       
                if PROFILE:
                    cProfile.run('run_rbpf_on_targetset(measurementTargetSetsBySequence[seq_idx], results_filename, params)')
                else:
                    (estimated_ts, cur_seq_info, number_resamplings) = run_rbpf_on_targetset(measurementTargetSetsBySequence[seq_idx], results_filename, params)
            print "done processing sequence: ", seq_idx
            
            tB = time.time()
            this_seq_run_time = tB - tA
            cur_seq_info.append(this_seq_run_time)
            if cur_run_info == None:
                cur_run_info = cur_seq_info
            else:
                assert(len(cur_run_info) == len(cur_seq_info))
                for info_idx in len(cur_run_info):
                    #assuming for now info can be summed over each sequence in a run!
                    #works for runtime and number of times resampling is performed
                    cur_run_info[info_idx] += cur_seq_info[info_idx]

            print "about to write results"

            if not SPEC['RUN_ONLINE']:
                estimated_ts.write_targets_to_KITTI_format(num_frames = n_frames[seq_idx], results_filename = results_filename,\
                                                           plot_filename = plot_filename)
            print "done write results"
            print "running the rbpf took %f seconds" % (tB-tA)
            
            info_by_run.append(cur_run_info)
            t1 = time.time()

            stdout = sys.stdout
            sys.stdout = open(indicate_run_complete_filename, 'w')

            print "This run is finished (and this file indicates the fact)\n"
            print "Resampling was performed %d times\n" % number_resamplings
            print "This run took %f seconds\n" % (t1-t0)

#            print "TARGET_EMISSION_PROBS=", TARGET_EMISSION_PROBS
#            print "CLUTTER_PROBABILITIES=", CLUTTER_PROBABILITIES
#            print "BIRTH_PROBABILITIES=", BIRTH_PROBABILITIES
#            print "MEAS_NOISE_COVS=", MEAS_NOISE_COVS
#            print "BORDER_DEATH_PROBABILITIES=", BORDER_DEATH_PROBABILITIES
#            print "NOT_BORDER_DEATH_PROBABILITIES=", NOT_BORDER_DEATH_PROBABILITIES


            sys.stdout.close()
            sys.stdout = stdout
            print "max_imprt_weight_count_dict: "
            print max_imprt_weight_count_dict

        print 'end run'


