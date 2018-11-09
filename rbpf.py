from __future__ import division

import numpy as np
import tensorflow as tf
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FWAction, FireTaskBase
#from fireworks.core.firework import FiretaskBase
#from filterpy.kalman import KalmanFilter
#from filterpy.common import Q_discrete_white_noise
from filterpy.monte_carlo import stratified_resample
import filterpy
from pymatgen.optimization import linear_assignment

#import matplotlib
#matplotlib.use('Agg')
#uncomment for plotting
#import matplotlib.pyplot as plt


#import matplotlib.cm as cmx
#import matplotlib.colors as colors
#from scipy.stats import multivariate_normal
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
from sets import ImmutableSet
import traceback

sys.path.insert(0, '/atlas/u/jkuck/gumbel_sample_permanent')
from tracking_specific_nestingUB_gumbel_sample_permanent import associationMatrix, multi_matrix_sample_associations_without_replacement

from cluster_config import RBPF_HOME_DIRECTORY
sys.path.insert(0, "%sKITTI_helpers" % RBPF_HOME_DIRECTORY)
import mailpy
from learn_params1 import get_meas_target_set
from learn_params1 import get_meas_target_sets_lsvm_and_regionlets
from learn_params1 import get_meas_target_sets_regionlets_general_format
from learn_params1 import get_meas_target_sets_mscnn_general_format
from learn_params1 import get_meas_target_sets_mscnn_and_regionlets
from learn_params1 import get_meas_target_sets_2sources_general
from learn_params1 import get_meas_target_sets_1sources_general
from learn_params1 import get_meas_target_sets_general
#from learn_params1_local import get_meas_target_sets_general
from learn_params1 import evaluate

sys.path.insert(0, "%ssampling" % RBPF_HOME_DIRECTORY)
from sample_wo_replacement import calc_prop_prob

sys.path.insert(0, "%scondition_priors" % RBPF_HOME_DIRECTORY)
from test_tf_sess import load_emmision_prior_model
from deep_target_emission_priors import get_deepsort_feature_arrays

from get_test_targetSets import get_meas_target_sets_test
from generate_data import KITTI_detection_file_to_TargetSet
import cProfile
import time
import os
sys.path.insert(0, "%sgeneral_tracking" % RBPF_HOME_DIRECTORY)
from global_params import DEFAULT_TIME_STEP
#Entries in the cost matrix that cannot be chosen as associations are set to this value or greater
from global_params import INFEASIBLE_COST

from rbpf_sampling_manyMeasSrcs import group_detections
from rbpf_sampling_manyMeasSrcs import solve_perturbed_max_gumbel
from rbpf_sampling_manyMeasSrcs import solve_perturbed_max_gumbel_exact
from rbpf_sampling_manyMeasSrcs import combine_arbitrary_number_measurements_4d as combine_4d_detections
from rbpf_sampling_manyMeasSrcs import sample_target_deaths
from rbpf_sampling_manyMeasSrcs import get_likelihood
from rbpf_sampling_manyMeasSrcs import get_assoc_prior
from rbpf_sampling_manyMeasSrcs import calc_death_prior
from rbpf_sampling_manyMeasSrcs import unnormalized_marginal_meas_target_assoc
from rbpf_sampling_manyMeasSrcs import get_immutable_set_meas_names
from rbpf_sampling_manyMeasSrcs import conditional_birth_clutter_distribution
from rbpf_sampling_manyMeasSrcs import nCr
from rbpf_sampling_manyMeasSrcs import construct_log_probs_matrix3
from rbpf_sampling_manyMeasSrcs import construct_log_probs_matrix4
from rbpf_sampling_manyMeasSrcs import convert_assignment_matrix3
from rbpf_sampling_manyMeasSrcs import convert_assignment_pairs_to_matrix3

sys.path.insert(0, "%smht_helpers" % RBPF_HOME_DIRECTORY)
from k_best_assign_birth_clutter_death_matrix import k_best_assign_mult_cost_matrices


#from k_best_assignment import k_best_assign_mult_cost_matrices
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


PROFILE = False
TRACE_PRINTS = False #useful for getting rid of unwanted print statements
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
#print stuff for debugging, etc
PRINT_INFO = False

#if True, save the MAP particle weight, whether this is the particles first time as
#the max importance weight particle (this might be meaningless, check details), and the number of living targets along with every
#line of the results file
SAVE_EXTRA_INFO = True

#(if False bug, using R_default instead of S, check SPEC['USE_CONSTANT_R']
#I'm pretty sure this is actually FIXED, but check out some time)

USE_PYTHON_GAUSSIAN = False 

#For testing why score interval for R are slow
CACHED_LIKELIHOODS = 0
NOT_CACHED_LIKELIHOODS = 0

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

class TracePrints(object):
  def __init__(self):    
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)

if TRACE_PRINTS:
    sys.stdout = TracePrints()

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
    def __init__(self, fw_spec, cur_time, id_, measurement = None, width=-1, height=-1, img_features=None):
#       if measurement is None: #for data generation
#           position = np.random.uniform(min_pos,max_pos)
#           velocity = np.random.uniform(min_vel,max_vel)
#           self.x = np.array([[position], [velocity]])
#           self.P = P_default
#       else:
        assert(measurement.all() != None)
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

        self.image_width = fw_spec['image_widths'][fw_spec['seq_idx']]
        self.image_height = fw_spec['image_heights'][fw_spec['seq_idx']]

        #features extracted from the image bounding box around this target
        self.img_features = img_features
        
        #used if SPEC['proposal_distr'] == 'ground_truth_assoc'
        #track_id of the gt_object this target was last associated with
        self.last_gt_assoc = None

    def near_border(self):
        near_border = False
        x1 = self.x[0][0] - self.width/2.0
        x2 = self.x[0][0] + self.width/2.0
        y1 = self.x[2][0] - self.height/2.0
        y2 = self.x[2][0] + self.height/2.0
        if(x1 < 10 or x2 > (self.image_width - 15) or y1 < 10 or y2 > (self.image_height - 15)):
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
               updated_P[3][3] > 0), (self.P, SPEC['R'], SPEC['USE_CONSTANT_R'], meas_noise_cov, K, updated_P)
#        print "kf_update called :)"
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


    def update(self, measurement, width, height, cur_time, meas_noise_cov, img_features=None,\
        meas_assoc_gt_obj_id=None):
        """ Perform update step and replace predicted position for the current time step
        with the updated position in self.all_states
        Input:
        - measurement: the measurement (numpy array)
        - cur_time: time when the measurement was taken (float)
        - meas_assoc_gt_obj_id: 
            - when SPEC['proposal_distr'] == 'ground_truth_assoc':
                the track_id of the ground truth object this measurement is 
                associated with, or -1 if the measurement is clutter
            - when SPEC['proposal_distr'] != 'ground_truth_assoc': None

!!!!!!!!!PREDICTION HAS BEEN RUN AT THE BEGINNING OF TIME STEP FOR EVERY TARGET!!!!!!!!!
        """        
        self.img_features = img_features
        self.last_gt_assoc = meas_assoc_gt_obj_id
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
#        print "kf_predict called :)"

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


        x1 = self.x[0][0] - self.width/2.0
        x2 = self.x[0][0] + self.width/2.0
        y1 = self.x[2][0] - self.height/2.0
        y2 = self.x[2][0] + self.height/2.0

        if(x2<0 or x1>=self.image_width or \
           y2<0 or y1>=self.image_height):
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
                frames_since_last_assoc = int(round((cur_time - self.last_measurement_association)/DEFAULT_TIME_STEP))
                assert(abs(float(frames_since_last_assoc) - (cur_time - self.last_measurement_association)/DEFAULT_TIME_STEP) < .00000001)
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
        #each numpy array has shape (2,)
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

    def __init__(self, fw_spec):
        #list of type Target containing targets currently alive
        self.living_targets = []
        self.all_targets = [] #alive and dead targets

        self.living_count = 0 #number of living targets
        self.total_count = 0 #number of living targets plus number of dead targets
        self.measurements = [] #generated measurements for a generative TargetSet 

        self.parent_target_set = None 

        self.living_targets_q = deque([-1 for i in range(SPEC['ONLINE_DELAY']+1)])

        self.fw_spec = fw_spec

    def create_child(self):
        child_target_set = TargetSet(self.fw_spec)
        child_target_set.parent_target_set = self
        child_target_set.total_count = self.total_count
        child_target_set.living_count = self.living_count
        child_target_set.all_targets = copy.deepcopy(self.living_targets)
        for target in child_target_set.all_targets:
            child_target_set.living_targets.append(target)
        child_target_set.living_targets_q = copy.deepcopy(self.living_targets_q)
        return child_target_set

    def create_new_target(self, measurement, width, height, cur_time, img_features, meas_assoc_gt_obj_id):
        '''

        Inputs:
        - meas_assoc_gt_obj_id: 
            - when SPEC['proposal_distr'] == 'ground_truth_assoc':
                the track_id of the ground truth object this measurement is 
                associated with, or -1 if the measurement is clutter
            - when SPEC['proposal_distr'] != 'ground_truth_assoc': None
        '''
        if SPEC['RUN_ONLINE']:
            global NEXT_TARGET_ID
            new_target = Target(self.fw_spec, cur_time, NEXT_TARGET_ID, np.squeeze(measurement), width, height, img_features=img_features)
            NEXT_TARGET_ID += 1
        else:
            new_target = Target(self.fw_spec, cur_time, self.total_count, np.squeeze(measurement), width, height, img_features=img_features)
        new_target.last_gt_assoc = meas_assoc_gt_obj_id
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

    def kill_offscreen_targets(self):
        '''
        Kill offscreen targets, run after predict, before association
        '''
        off_screen_target_indices = []
        for idx, target in enumerate(self.living_targets):
            if target.offscreen:
                off_screen_target_indices.append(idx)
        for offscreen_idx in reversed(off_screen_target_indices):
            self.kill_target(offscreen_idx)

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
        "collect_ancestral_targets called!"
        every_target = []
        found_target_ids = descendant_target_ids
        for target in self.all_targets:
            if(not target.id_ in found_target_ids):
                every_target.append(target)
                found_target_ids.append(target.id_)
            print "len(every_target)", len(every_target)
        if self.parent_target_set == None:
            "self.parent_target_set == None !"
            return every_target
        else:
            ancestral_targets = self.parent_target_set.collect_ancestral_targets(found_target_ids)

        every_target = every_target + ancestral_targets # + operator used to concatenate lists!
        return every_target


    def write_online_results(self, online_results_filename, frame_idx, total_frame_count, extra_info, fw_spec,\
        min_target_life=0):
        """
        Inputs:
        - extra_info: dictionary containing the particle's importance weight (key 'importance_weight') 
            and boolean whether this is the first time the particle is the max importance weight 
            particle (key 'first_time_as_max_imprt_part')
        - min_target_life: (int) only write targets that have been alive for at least a total of this many time steps

        """
        if frame_idx == SPEC['ONLINE_DELAY']:
            f = open(online_results_filename, "w") #write over old results if first frame
        else:
            f = open(online_results_filename, "a") #write at end of file

        if SPEC['ONLINE_DELAY'] == 0:
            print "fw_spec['obj_class']:", fw_spec['obj_class']

            for target in self.living_targets:
                assert(target.all_time_stamps[-1] == round(frame_idx*DEFAULT_TIME_STEP, 2)), (target.all_time_stamps[-1], round(frame_idx*DEFAULT_TIME_STEP, 2))
                if len(target.all_time_stamps) < min_target_life: #target is not old enough
                    continue
                x_pos = target.all_states[-1][0][0][0]
                y_pos = target.all_states[-1][0][2][0]
                width = target.all_states[-1][1]
                height = target.all_states[-1][2]

                left = x_pos - width/2.0
                top = y_pos - height/2.0
                right = x_pos + width/2.0
                bottom = y_pos + height/2.0      
                if SAVE_EXTRA_INFO:
                    f.write( "%d %d %s -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                        (frame_idx, target.id_, fw_spec['obj_class'], left, top, right, bottom, extra_info['MAP_particle_prob'], \
                        extra_info['first_time_as_max_imprt_part'], extra_info['sampled_meas_targ_assoc_idx']))
                else:
                    f.write( "%d %d %s -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                        (frame_idx, target.id_, fw_spec['obj_class'], left, top, right, bottom))

        else:
            print self.living_targets_q
            (delayed_frame_idx, delayed_liv_targets) = self.living_targets_q[1]
            print delayed_frame_idx
            print delayed_liv_targets
            assert(delayed_frame_idx == frame_idx - SPEC['ONLINE_DELAY']), (delayed_frame_idx, frame_idx, SPEC['ONLINE_DELAY'])
            for target in delayed_liv_targets:
                assert(target.all_time_stamps[-1] == round((frame_idx - SPEC['ONLINE_DELAY'])*DEFAULT_TIME_STEP, 2)), (target.all_time_stamps[-1], frame_idx, SPEC['ONLINE_DELAY'], round((frame_idx - SPEC['ONLINE_DELAY'])*DEFAULT_TIME_STEP, 2))
                x_pos = target.all_states[-1][0][0][0]
                y_pos = target.all_states[-1][0][2][0]
                width = target.all_states[-1][1]
                height = target.all_states[-1][2]

                left = x_pos - width/2.0
                top = y_pos - height/2.0
                right = x_pos + width/2.0
                bottom = y_pos + height/2.0      
                if SAVE_EXTRA_INFO:
                    f.write( "%d %d %s -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                        (frame_idx - SPEC['ONLINE_DELAY'], target.id_, fw_spec['obj_class'], left, top, right, bottom, extra_info['MAP_particle_prob'], \
                        extra_info['first_time_as_max_imprt_part'], extra_info['sampled_meas_targ_assoc_idx']))
                else:
                    f.write( "%d %d %s -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                        (frame_idx - SPEC['ONLINE_DELAY'], target.id_, fw_spec['obj_class'], left, top, right, bottom))

            if frame_idx == total_frame_count - 1:
                q_idx = 2
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
                        assert(target.all_time_stamps[-1] == round((cur_frame_idx)*DEFAULT_TIME_STEP, 2))
                        x_pos = target.all_states[-1][0][0][0]
                        y_pos = target.all_states[-1][0][2][0]
                        width = target.all_states[-1][1]
                        height = target.all_states[-1][2]

                        left = x_pos - width/2.0
                        top = y_pos - height/2.0
                        right = x_pos + width/2.0
                        bottom = y_pos + height/2.0      
                        if SAVE_EXTRA_INFO:
                            f.write( "%d %d %s -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                                (cur_frame_idx, target.id_, fw_spec['obj_class'], left, top, right, bottom, extra_info['MAP_particle_prob'], \
                                extra_info['first_time_as_max_imprt_part'], extra_info['sampled_meas_targ_assoc_idx']))
                            print "%d %d %s -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                                (cur_frame_idx, target.id_, fw_spec['obj_class'], left, top, right, bottom, extra_info['MAP_particle_prob'], \
                                extra_info['first_time_as_max_imprt_part'], extra_info['sampled_meas_targ_assoc_idx'])
                        else:
                            f.write( "%d %d %s -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                                (cur_frame_idx, target.id_, fw_spec['obj_class'], left, top, right, bottom))

                print "&&&&&&&&&"
                for target in self.living_targets:
                    assert(target.all_time_stamps[-1] == round(frame_idx*DEFAULT_TIME_STEP, 2))
                    x_pos = target.all_states[-1][0][0][0]
                    y_pos = target.all_states[-1][0][2][0]
                    width = target.all_states[-1][1]
                    height = target.all_states[-1][2]

                    left = x_pos - width/2.0
                    top = y_pos - height/2.0
                    right = x_pos + width/2.0
                    bottom = y_pos + height/2.0      
                    if SAVE_EXTRA_INFO:
                        f.write( "%d %d %s -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                            (frame_idx, target.id_, fw_spec['obj_class'], left, top, right, bottom, extra_info['MAP_particle_prob'], \
                            extra_info['first_time_as_max_imprt_part'], extra_info['sampled_meas_targ_assoc_idx']))
                        print  "%d %d %s -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1 %f %s %d\n" % \
                            (frame_idx, target.id_, fw_spec['obj_class'], left, top, right, bottom, extra_info['MAP_particle_prob'], \
                            extra_info['first_time_as_max_imprt_part'], extra_info['sampled_meas_targ_assoc_idx'])
                    else:
                        f.write( "%d %d %s -1 -1 2.57 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                            (frame_idx, target.id_, fw_spec['obj_class'], left, top, right, bottom))



    def write_targets_to_KITTI_format(self, num_frames, results_filename, plot_filename, fw_spec):
        print "write_targets_to_KITTI_format called!"
        x_locations_all_targets = defaultdict(list)
        y_locations_all_targets = defaultdict(list)
        if USE_GENERATED_DATA:
            num_frames = NUM_GEN_FRAMES
        if USE_CREATE_CHILD:
            print "about to call self.collect_ancestral_targets()"
            every_target = self.collect_ancestral_targets()
            f = open(results_filename, "w")
            for frame_idx in range(num_frames):
                timestamp = round(frame_idx*DEFAULT_TIME_STEP, 2)

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
                        f.write( "%d %d %s -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                            (frame_idx, target.id_, fw_spec['obj_class'], left, top, right, bottom))

                        x_locations_all_targets[target.id_].append(x_pos)
                        y_locations_all_targets[target.id_].append(y_pos)

            f.close()

        else:
            print "did not call self.collect_ancestral_targets()"
            print "USE_CREATE_CHILD =", USE_CREATE_CHILD
            f = open(results_filename, "w")
            for frame_idx in range(num_frames):
                timestamp = round(frame_idx*DEFAULT_TIME_STEP, 2)
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
                        f.write( "%d %d %s -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1\n" % \
                            (frame_idx, target.id_, fw_spec['obj_class'], left, top, right, bottom))

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
    def __init__(self, id_, fw_spec):
        #Targets tracked by this particle
        self.targets = TargetSet(fw_spec)
        self.fw_spec = fw_spec
        #all_measurement_associations[i] will be a list of measurement associations on time step i.
        #all_dead_targets[i] will be a list of all targets killed on time step i.
        #these two lists uniquely identifies the state of this particle and are used for determining whether
        #two separate particles represent the same state
        #NOTE, only implemented when SPEC['use_general_num_dets'] == True
        self.all_measurement_associations = []
        self.all_dead_targets = []

        self.importance_weight = 1.0/N_PARTICLES

        #for debuging
        self.exact_log_probability = 0
        self.cur_conditional_log_prob = -1
        self.proposal_probability = -1
        #end for debugging

        self.likelihood_DOUBLE_CHECK_ME = -1
        #cache for memoizing association likelihood computation
        self.assoc_likelihood_cache = {}

        self.id_ = id_ 

        self.parent_particle = None

        #for debugging
        self.c_debug = -1
        self.imprt_re_weight_debug = -1
        self.pi_birth_debug = -1
        self.pi_clutter_debug = -1
        self.pi_targets_debug = []

        #bool for debugging, indicating maximum importance weight from previous time instance
        self.max_importance_weight = False 

        self.sampled_meas_targ_assoc_idx = -99 #garbage value


    def create_child(self):
        global NEXT_PARTICLE_ID
        child_particle = Particle(NEXT_PARTICLE_ID, self.fw_spec)
        NEXT_PARTICLE_ID += 1
        child_particle.parent_particle = self #this might hurt memory with real data
        child_particle.importance_weight = self.importance_weight
        child_particle.targets = self.targets.create_child()
        child_particle.all_measurement_associations = copy.deepcopy(self.all_measurement_associations)
        child_particle.all_dead_targets = copy.deepcopy(self.all_dead_targets)
        return child_particle

    def create_new_target(self, measurement, width, height, cur_time, img_features, meas_assoc_gt_obj_id):
        self.targets.create_new_target(measurement, width, height, cur_time,img_features=img_features,\
            meas_assoc_gt_obj_id=meas_assoc_gt_obj_id)

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

    def process_meas_grp_assoc(self, birth_value, measurement_association, meas_grp_mean, meas_grp_cov, cur_time, \
                               img_features=None, meas_assoc_gt_obj_id=None):
        """
        - meas_source_index: the index of the measurement source being processed (i.e. in SCORE_INTERVALS)
        - meas_assoc_gt_obj_id: 
            - when SPEC['proposal_distr'] == 'ground_truth_assoc':
                the track_id of the ground truth object this measurement is 
                associated with, or -1 if the measurement is clutter
            - when SPEC['proposal_distr'] != 'ground_truth_assoc': None

        """
        #create new target
        if(measurement_association == birth_value):
            self.create_new_target(meas_grp_mean[0:2], meas_grp_mean[2], meas_grp_mean[3], cur_time, \
                img_features=img_features, meas_assoc_gt_obj_id=meas_assoc_gt_obj_id)
            new_target = True 
        #update the target corresponding to the association we have sampled
        elif((measurement_association >= 0) and (measurement_association < birth_value)):
            self.targets.living_targets[measurement_association].update(meas_grp_mean[0:2], meas_grp_mean[2], \
                            meas_grp_mean[3], cur_time, meas_grp_cov[0:2, 0:2], img_features=img_features,\
                            meas_assoc_gt_obj_id=meas_assoc_gt_obj_id)
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
            (meas_grp_associations, meas_grp_means, meas_grp_covs, meas_grp_img_feats, dead_target_indices, imprt_re_weight, exact_log_probability, proposal_probability, meas_assoc_gt_obj_ids, sampled_meas_targ_assoc_idx) = \
            sample_and_reweight(self, measurement_lists,  widths, heights, SPEC['det_names'], \
                cur_time, measurement_scores, params)
            self.sampled_meas_targ_assoc_idx = sampled_meas_targ_assoc_idx
            #debug
            self.exact_log_probability += exact_log_probability
            self.cur_conditional_log_prob = exact_log_probability
#            self.proposal_probability = proposal_probability
            #end debug

            self.all_measurement_associations.append(meas_grp_associations)
            self.all_dead_targets.append(dead_target_indices)
            if SPEC['normalize_log_importance_weights'] == True:
                if self.importance_weight > 0.0:
                    self.importance_weight = imprt_re_weight + math.log(self.importance_weight)
                else:#use very small log probability
                    self.importance_weight =imprt_re_weight - 500
            else:
                self.importance_weight *= imprt_re_weight #update particle's importance weight            
            assert(len(meas_grp_associations) == len(meas_grp_means) and len(meas_grp_means) == len(meas_grp_covs))
            if meas_assoc_gt_obj_ids != None:            
                assert(len(meas_grp_associations) == len(meas_assoc_gt_obj_ids))
            for meas_grp_idx, meas_grp_assoc in enumerate(meas_grp_associations):
                cur_meas_assoc_gt_obj_id = None
                if meas_assoc_gt_obj_ids != None:
                    cur_meas_assoc_gt_obj_id = meas_assoc_gt_obj_ids[meas_grp_idx]

                if params.SPEC['condition_emission_prior_img_feat']:
                    self.process_meas_grp_assoc(birth_value, meas_grp_assoc, meas_grp_means[meas_grp_idx], meas_grp_covs[meas_grp_idx], \
                                            cur_time, img_features=meas_grp_img_feats[meas_grp_idx], meas_assoc_gt_obj_id=cur_meas_assoc_gt_obj_id)
                else:
                    self.process_meas_grp_assoc(birth_value, meas_grp_assoc, meas_grp_means[meas_grp_idx], meas_grp_covs[meas_grp_idx], \
                                            cur_time, img_features=None, meas_assoc_gt_obj_id=cur_meas_assoc_gt_obj_id)

        else:
            (measurement_associations, dead_target_indices, imprt_re_weight) = \
                sample_and_reweight(self, measurement_lists, \
                    cur_time, measurement_scores, params)


            assert(len(measurement_associations) == len(measurement_lists))
            assert(imprt_re_weight != 0.0), imprt_re_weight
            if SPEC['normalize_log_importance_weights'] == True:
                #update particle's importance weight to be the log of updated importance weight                           
                self.importance_weight = imprt_re_weight + math.log(self.importance_weight)
            else:            
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




def normalize_importance_weights(particle_set, check_normalization_before_call=False):

    #####debugging#########
    print 'particle weights before normalization:',
    imp_weight_sum = 0.0
    for particle in particle_set:
        imp_weight_sum += particle.importance_weight
        print particle.importance_weight,
    print  
    if check_normalization_before_call:      
        assert(np.abs(imp_weight_sum - 1.0) > .0001), (SPEC['normalize_log_importance_weights'], 'normalize_importance_weights called when weights appear normalized')
    #####end debugging#########


    if SPEC['normalize_log_importance_weights'] == True:
        max_imprt_weight = 'not_set'
        for particle in particle_set:
            if max_imprt_weight == 'not_set':
                max_imprt_weight = particle.importance_weight
            elif particle.importance_weight > max_imprt_weight:
                max_imprt_weight = particle.importance_weight

        for particle in particle_set:
            #divide all importance weights by the largest importance weight (in log, so subtract)
            #in case importance weights are all very small
            #and convert log importance weights back to actual importance weights
            particle.importance_weight = math.exp(particle.importance_weight - max_imprt_weight)


    #now normalize importance weights
    normalization_constant = 0.0
    for particle in particle_set:
        normalization_constant += particle.importance_weight
    assert(normalization_constant != 0.0), normalization_constant
    for particle in particle_set:
        particle.importance_weight /= normalization_constant

    print 'particle weights after normalization:',
    for particle in particle_set:
        print particle.importance_weight,
    print


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

def cur_particle_states_match(particleA, particleB, min_delay, max_delay):
    '''
    Inputs:
    - particleA: type Particle
    - particleB: type Particle
    - min_delay: positive integer, check particles had the same state [min_delay,max_delay] time instances in the past,
        i.e. if min_delay=0 and max_delay=2 we check that the states match on this time instance, the previous time
        instance, and two time instances in the past
    - max_delay: positive integer


    Outputs:
    - match: type Boolean, whether particleA and particleB have the same current state
        (the same number of living targets in the same locations).  We assume matching
        targets should be in the same locations in each particle's target list.  Also
        check importance weights are the same (they should be).
    '''
    assert(min_delay <= SPEC['ONLINE_DELAY'] and max_delay <= SPEC['ONLINE_DELAY'] and min_delay<=max_delay)

    match = True

    if min_delay == 0:    
        #check the number of targets is the same
        if len(particleA.targets.living_targets) != len(particleB.targets.living_targets):
            match = False
            return (match, 'different number of targets')
    
        #check all targets have the same state, position mean and covariance, width, and height
        for idx in range(len(particleA.targets.living_targets)):
            targetA = particleA.targets.living_targets[idx]
            targetB = particleB.targets.living_targets[idx]
            if not (targetA.x == targetB.x).all():
                match = False
                return (match, 'different states')
            if not (targetA.P == targetB.P).all():
                match = False
                return (match, 'different covariances')
            if targetA.width != targetB.width:
                match = False
                return (match, 'different widths')
            if targetA.height != targetB.height:
                match = False
                return (match, 'different heights')

    for cur_delay in range(max(1, min_delay), max_delay+1):
        pastTargetsA = particleA.targets.living_targets_q[-cur_delay][1]
        pastTargetsB = particleB.targets.living_targets_q[-cur_delay][1]

        #check the number of targets is the same
        if len(pastTargetsA) != len(pastTargetsB):
            match = False
            return (match, 'different number of targets')
    
        #check all targets have the same state, position mean and covariance, width, and height
        for idx in range(len(pastTargetsA)):
            targetA = pastTargetsA[idx]
            targetB = pastTargetsB[idx]
            if not (targetA.x == targetB.x).all():
                match = False
                return (match, 'different states')
            if not (targetA.P == targetB.P).all():
                match = False
                return (match, 'different covariances')
            if targetA.width != targetB.width:
                match = False
                return (match, 'different widths')
            if targetA.height != targetB.height:
                match = False
                return (match, 'different heights')

    #Actually, particles with the same state may not have the same importance weight if
    #the proposal distribution we used was different for the two particles, in the case
    #of min_cost proposal distribution.  CHECK DETAILS ON WHETHER THIS IS ALLOWED IN SIS FRAMEWORK!!!
    #Also, we could have the same state at time k, but have arrived at the state through different
    #associations (e.g. particleA gives birth to a target and then kills it that particleB never
    #gave birth to.)

    #check both particles have the same importance weight
#    if particleA.importance_weight != particleB.importance_weight:
#        match = False
#        return (match, 'different importance weights')

    return (match, 'match!')

def group_particles(particle_set, min_delay, max_delay): 
    '''
    ###CONSIDER ROUNDING PARTICLE POSITIONS TO SOME DEGREE, I don't THINK not rounding causes a bug###
    ###We assume the same targets are always in the same list position, I THINK this is ok###
    ### We don't check how long targets have been unassociated for, but I think this should be OK,
    because it should be basically impossible for two targets to have the same position and covariance
    with different association histories ###
    Input:
    - particle_set: list of type Particle
    - min_delay: positive integer, group particles that had the same state [min_delay,max_delay] time instances in the past,
        i.e. if min_delay=0 and max_delay=2 we require that the state match on this time instance, the previous time
        instance, and two time instances in the past
    - max_delay: positive integer
    Outputs:
    - particle_group_probs: a dictionary where each entry represents a particle group.  Keys represent
        the state of all particles in the group and values the sum of importance weights of particles 
        in the group.
    - particle_groups: a dictionary with the same keys as particle_group_probs.  Values are one of the
        actual particles belonging to the group (all have the same CURRENT state, so it doesn't matter which one)
    '''
    # assert(SPEC['RUN_ONLINE'])
    if SPEC['RUN_ONLINE']:
        assert(min_delay <= SPEC['ONLINE_DELAY'] and max_delay <= SPEC['ONLINE_DELAY'] and min_delay<=max_delay)
    num_time_steps = len(particle_set[0].all_measurement_associations)
    particle_group_probs = {}
    #particle_groups is a dictionary with the same keys as particle_group_probs.  Values are one of the
    #actual particles belonging to the group (all are the same, so it doesn't matter which one)
    particle_groups = {}
    for particle in particle_set:
        #check for funny business
        assert(len(particle.all_measurement_associations) == num_time_steps), (particle.all_measurement_associations, len(particle.all_measurement_associations))
        if min_delay == 0:
            particle_state_key = \
                tuple([tuple([tuple(target.x.flatten()),tuple(target.P.flatten()),target.width,target.height]) \
                       for target in particle.targets.living_targets])
        else:
            particle_state_key = ()

        for cur_delay in range(max(1, min_delay), max_delay+1):
            particle_state_key = \
                tuple([particle_state_key,\
                tuple([tuple([tuple(target.x.flatten()),tuple(target.P.flatten()),target.width,target.height]) \
                       for target in particle.targets.living_targets_q[-cur_delay][1]])])

        if particle_state_key in particle_group_probs:
            particle_group_probs[particle_state_key] += particle.importance_weight
            (match_bool, match_str) = cur_particle_states_match(particle_groups[particle_state_key], particle, min_delay, max_delay)
            assert(match_bool), (match_str, particle_state_key, time_instance_index, len(particle.all_measurement_associations), 
                particle.importance_weight, particle_groups[particle_state_key].importance_weight, 
                particle.all_measurement_associations, particle_groups[particle_state_key].all_measurement_associations,
                particle.all_dead_targets, particle_groups[particle_state_key].all_dead_targets,
                measurement_lists)
        else:
            particle_group_probs[particle_state_key] = particle.importance_weight
            particle_groups[particle_state_key] = particle

        ############testing############
        for cur_key, cur_particle in particle_groups.iteritems():
            if(cur_key != particle_state_key):
                (match_bool, match_str) = cur_particle_states_match(cur_particle, particle, min_delay, max_delay)
                assert(not match_bool), (match_str, particle_state_key, cur_key, len(particle.all_measurement_associations), 
                    particle.importance_weight, particle_groups[particle_state_key].importance_weight, 
                    particle.all_measurement_associations, particle_groups[particle_state_key].all_measurement_associations,
                    cur_particle.all_measurement_associations,
                    particle.all_dead_targets, particle_groups[particle_state_key].all_dead_targets)
        ############done testing############

    ############testing############
    total_prob = 0.0
    for key, prob in particle_group_probs.iteritems():
        total_prob += prob
    # assert(np.isclose(total_prob, 1, rtol=1e-04, atol=1e-04)), total_prob
    ############done testing############
    return(particle_group_probs, particle_groups)

def modified_SIS_gumbel_step(particle_set, measurement_lists, widths, heights, cur_time, params):
    (particle_group_probs, particle_groups) = group_particles(particle_set, 0, SPEC['ONLINE_DELAY'])
    #housekeeping, should make this nicer somehow
    meas_groups = []
    for det_idx, det_name in enumerate(SPEC['det_names']):
        group_detections(meas_groups, det_name, measurement_lists[det_idx], widths[det_idx], heights[det_idx], params)


    #now that we have estimates of p(x_1:k-1|y_1:k-1), perform modified SIS step
    new_particle_set = []
    particle_group_log_probs = {}
    for idx in range(N_PARTICLES):
        #1. solve perturbed max(log(p(x_k, y_k | x_1:k-1, y_1:k-1))) problem for each particle group
        for p_key, particle in particle_groups.iteritems():
            #should clean this up
            p_target_deaths = []
            for target in particle.targets.living_targets:
                p_target_deaths.append(target.death_prob)
                assert(p_target_deaths[len(p_target_deaths) - 1] >= 0 and p_target_deaths[len(p_target_deaths) - 1] <= 1)

            assert(len(particle.targets.living_targets) == particle.targets.living_count)
            (meas_associations, dead_target_indices, max_log_prob) = \
                solve_perturbed_max_gumbel(particle, meas_groups, len(particle.targets.living_targets), 
                p_target_deaths, params)
#                solve_perturbed_max_gumbel_exact(particle, meas_groups, len(particle.targets.living_targets), 
#                p_target_deaths, params)

            #add log(p_hat(x_1:k-1|y_1:k-1)) to max(log(p(x_k, y_k | x_1:k-1, y_1:k-1))) 
            #for each particle group and store x_k
            particle_group_log_probs[p_key] = \
                {'max_log_prob': max_log_prob + np.log(particle_group_probs[p_key]),
                 'meas_associations': meas_associations,
                 'dead_target_indices': dead_target_indices}

        #2. find the particle group with the maximum log probability
        maximum_log_prob = -999999999999999999999999
        maximum_log_prob_p_key = None
        for p_key, assoc_dict in particle_group_log_probs.iteritems():
            if assoc_dict['max_log_prob'] > maximum_log_prob:
                maximum_log_prob = assoc_dict['max_log_prob']
                maximum_log_prob_p_key = p_key
                print 'found a new max log prob particle group!!!!',  assoc_dict['max_log_prob']
            else:
                print 'max log prob,', assoc_dict['max_log_prob'], 'too small'
        #important to have != None, assert(()) on the empty tuple produces an error    
        assert(maximum_log_prob_p_key != None), particle_group_log_probs 


        #3. create a new particle that is a copy of the max group, and associate measurements / kill
        #targets according to the max x_k from 2.
        assert(SPEC['use_general_num_dets'])
        new_particle = particle_groups[maximum_log_prob_p_key].create_child()
        birth_value = new_particle.targets.living_count


    ############ MESSY ############
        #list of detection group centers, meas_grp_means[i] is a 2-d numpy array
        #of the position of meas_groups[i]
        meas_grp_covs = []   
        meas_grp_means2D = []
        meas_grp_means = []
        for (index, detection_group) in enumerate(meas_groups):
            (combined_meas_mean, combined_covariance) = combine_4d_detections(params.posAndSize_inv_covariance_blocks, 
                                params.meas_noise_mean, detection_group)
            combined_meas_pos = combined_meas_mean[0:2]
            meas_grp_means2D.append(combined_meas_pos)
            meas_grp_means.append(combined_meas_mean)
            meas_grp_covs.append(combined_covariance)
    ############ END MESSY ############

        meas_grp_associations = particle_group_log_probs[maximum_log_prob_p_key]['meas_associations']
        dead_target_indices = particle_group_log_probs[maximum_log_prob_p_key]['dead_target_indices']


        new_particle.all_measurement_associations.append(meas_grp_associations)
        new_particle.all_dead_targets.append(dead_target_indices)  
        assert(len(meas_grp_associations) == len(meas_grp_means) and len(meas_grp_means) == len(meas_grp_covs))
        for meas_grp_idx, meas_grp_assoc in enumerate(meas_grp_associations):
            new_particle.process_meas_grp_assoc(birth_value, meas_grp_assoc, meas_grp_means[meas_grp_idx], meas_grp_covs[meas_grp_idx], cur_time)

        #process target deaths
        #double check dead_target_indices is sorted
        assert(all([dead_target_indices[i] <= dead_target_indices[i+1] for i in xrange(len(dead_target_indices)-1)]))
        #important to delete larger indices first to preserve values of the remaining indices
        for index in reversed(dead_target_indices):
            new_particle.targets.kill_target(index)

        #checking if something funny is happening
        original_num_targets = birth_value
        num_targets_born = 0
        num_targets_born = meas_grp_associations.count(birth_value)
        num_targets_killed = len(dead_target_indices)
        assert(new_particle.targets.living_count == original_num_targets + num_targets_born - num_targets_killed)
        #done checking if something funny is happening
        new_particle_set.append(new_particle)
        assert(new_particle.parent_particle != None)

    return new_particle_set




def modified_SIS_MHT_gumbel_step(particle_set, measurement_lists, widths, heights, cur_time, params):
    '''
    Very similar to modified_SIS_gumbel_step, but we sample new particles w/o replacement.  Also
    params.SPEC['gumbel_scale'] should exist.  When params.SPEC['gumbel_scale'] = 0, we get MHT back.
    When params.SPEC['gumbel_scale'] = 1, we are taking the mean (instead of max) of gumbel perturbations
    to assignment cost matrix and multiplying it by params.SPEC['gumbel_scale'].

    proposal options implemented here:
    params.SPEC['proposal_distr'] == 'modified_SIS_gumbel':
        - when gumbel scale is 0, this is MHT

    params.SPEC['proposal_distr'] == 'modified_SIS_wo_replacement_approx':
        -sample 'num_particles' hypotheses from 'num_top_hypotheses_to_sample_from' without replacement,
        and ignore the fact that when sampling w/o replacement samples are dependent and we need to
        calculate the proposal probability and reweight particles by dividing by this prob

    params.SPEC['proposal_distr'] == 'modified_SIS_w_replacement':
        -sample 'num_particles' hypotheses from 'num_top_hypotheses_to_sample_from' with replacement,

    params.SPEC['proposal_distr'] == 'modified_SIS_w_replacement_unique':
        -sample 'num_particles' unique hypotheses from 'num_top_hypotheses_to_sample_from' with replacement,
        that is, keep sampling until we have 'num_particles' different hypotheses

    '''
    if SPEC['RUN_ONLINE'] == True:
        (particle_group_probs, particle_groups) = group_particles(particle_set, 0, SPEC['ONLINE_DELAY'])
    else: #make two dictionaries of all particles, to match output of group_particles without performing any grouping
        # (particle_group_probs, particle_groups) = group_particles(particle_set, 0, int(cur_time*10)) 
        particle_group_probs = {}
        particle_groups = {}
        for idx, particle in enumerate(particle_set):
            particle_group_probs[idx] = particle.importance_weight
            particle_groups[idx] = particle
            if cur_time == 0: #only one particle group on first time instance
                break

    #housekeeping, should make this nicer somehow
    meas_groups = []
    for det_idx, det_name in enumerate(SPEC['det_names']):
        group_detections(meas_groups, det_name, measurement_lists[det_idx], widths[det_idx], heights[det_idx], params)

    M = len(meas_groups) #number of measurement groups
    #now that we have estimates of p(x_1:k-1|y_1:k-1), perform modified SIS step
    new_particle_set = []
    particle_group_log_probs = {}
#    for idx in range(N_PARTICLES): 

    perturbed_cost_matrices = [] #list of negative perturbed log prob matrices for each particle group
    log_prob_matrices = [] #list of log prob matrices for each particle group
    ordered_particle_groups = [] #list of particle groups in the same order as perturbed_cost_matrices
    particle_neg_log_probs = [] # negative log probabilities 
    particle_costs = [] # negative log probabilities + 2*(M+T) of the min_cost for each particle group to pass along when solving minimum cost assignments
    #we need all entries of all cost matrices to be positive, so we keep track of the smallest value
    min_cost = 0.0

    invalid_low_prob_sample_count = 0 

    zero_assignments = [] #particle group(s) exist with zero targets and we have zero measurements
    for p_key, particle in particle_groups.iteritems():
        T = len(particle.targets.living_targets)
        print "T =", T
        assert(T == particle.targets.living_count)
        #should clean this up
        p_target_deaths = []
        for target in particle.targets.living_targets:
            p_target_deaths.append(target.death_prob)
            assert(p_target_deaths[len(p_target_deaths) - 1] >= 0 and p_target_deaths[len(p_target_deaths) - 1] <= 1)


        #1. construct log probs matrix for  particle GROUP
        cur_log_probs = construct_log_probs_matrix3(particle, meas_groups, T, p_target_deaths, params)
        log_prob_matrices.append(cur_log_probs) #store to calculate probabilities later
        assert((cur_log_probs <= .000001).all()), (cur_log_probs)

        if params.SPEC['proposal_distr'] == 'modified_SIS_gumbel':
            #3. add gumbel matrix to log probs matrix, scaled by params.SPEC['gumbel_scale']/(number of assignments)
            G = np.random.gumbel(loc=0.0, scale=1.0, size=(cur_log_probs.shape[0], cur_log_probs.shape[1]))
            number_of_assignments = 2*(M + T)
            G = G*params.SPEC['gumbel_scale']/number_of_assignments
            cur_log_probs += G

        cur_cost_matrix = -1*cur_log_probs #k_best_assign_mult_cost_matrices is set up to find minimum cost, not max log prob
        
        if cur_cost_matrix.size > 0:
            cur_min_cost = np.min(cur_cost_matrix)
        else:
            cur_min_cost = 0.0

        if cur_min_cost < min_cost:
            min_cost = cur_min_cost

        perturbed_cost_matrices.append(cur_cost_matrix)
        ordered_particle_groups.append(particle)
        #add min_cost*2*(M+T) term because T varies between particle groups and we subtract min_cost from all entries in the perturbed cost matrix
        particle_costs.append(-1*np.log(particle_group_probs[p_key]) + min_cost*2*(M+T))
        particle_neg_log_probs.append(-1*np.log(particle_group_probs[p_key]))


    #make all entries of all cost matrices non-negative
    for idx in range(len(perturbed_cost_matrices)):
        perturbed_cost_matrices[idx] = perturbed_cost_matrices[idx] - min_cost
        assert((perturbed_cost_matrices[idx] >= 0.0).all()), (perturbed_cost_matrices[idx])

    #4. find N_PARTICLES most likely assignments among all assignments in log probs matrices of ALL particle GROUPS

    #best_assignments: (list of triplets) best_assignments[i][0] is the cost of the ith best
    #assignment.  best_assignments[i][1] is the ith best assignment, which is a list of pairs
    #where each pair represents an association in the assignment (1's in assignment matrix),
    #best_assignments[i][2] is the index in the input cost_matrices of the cost matrix used
    #for the ith best assignment

    print 'M =', M

    if params.SPEC['proposal_distr'] == 'modified_SIS_gumbel':
        best_assignments = k_best_assign_mult_cost_matrices(N_PARTICLES, perturbed_cost_matrices, particle_costs, M)
#    best_assignments = k_best_assign_mult_cost_matrices(N_PARTICLES, perturbed_cost_matrices)

    else: 
        #now we sample without replacemenent from the most likely assignments
        best_assignments = k_best_assign_mult_cost_matrices(params.SPEC['num_top_hypotheses_to_sample_from'], perturbed_cost_matrices, particle_costs, M)        
        
        assignment_proposal_distr = []
        for (cur_cost, cur_assignment, cur_particle_idx) in best_assignments:
            T = len(ordered_particle_groups[cur_particle_idx].targets.living_targets)            
            cur_assignment_matrix = convert_assignment_pairs_to_matrix3(cur_assignment, M, T)

            assignment_log_prob = np.trace(np.dot(log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T))
            assignment_prob = np.exp(assignment_log_prob - particle_neg_log_probs[cur_particle_idx])
            assignment_proposal_distr.append(assignment_prob)

        assignment_proposal_distr = np.asarray(assignment_proposal_distr)
        assignment_proposal_distr /= float(np.sum(assignment_proposal_distr))
        assert(np.abs(np.sum(assignment_proposal_distr) - 1.0) < .000001)

        if params.SPEC['proposal_distr'] == 'modified_SIS_wo_replacement_approx':
            #-sample 'num_particles' hypotheses from 'num_top_hypotheses_to_sample_from' without replacement,
            #and ignore the fact that when sampling w/o replacement samples are dependent and we need to
            #calculate the proposal probability and reweight particles by dividing by this prob
            sampled_assignment_indices = np.random.choice(len(assignment_proposal_distr), size=min((len(particle_set)), len(assignment_proposal_distr)), replace=False, p=assignment_proposal_distr)
            sampled_assignments = []
            for sampled_idx in sampled_assignment_indices:
                sampled_assignments.append(best_assignments[sampled_idx])
            best_assignments = sampled_assignments



        elif params.SPEC['proposal_distr'] == 'modified_SIS_w_replacement':
            #-sample 'num_particles' hypotheses from 'num_top_hypotheses_to_sample_from' with replacement,
            sampled_assignment_indices = np.random.choice(len(assignment_proposal_distr), size=(len(particle_set)), replace=True, p=assignment_proposal_distr)
            sampled_assignments = []
            for sampled_idx in sampled_assignment_indices:
                sampled_assignments.append(best_assignments[sampled_idx])
            best_assignments = sampled_assignments


        else:
            assert(params.SPEC['proposal_distr'] == 'modified_SIS_w_replacement_unique')
            #-sample 'num_particles' unique hypotheses from 'num_top_hypotheses_to_sample_from' with replacement,
            #that is, keep sampling until we have 'num_particles' different hypotheses
            total_sample_count = 0
            unique_sample_counts = defaultdict(int) #key: unique sample index, value: number of times this index has been sampled

            while(len(unique_sample_counts) < min(len(assignment_proposal_distr), len(particle_set))):
                sampled_idx = np.random.choice(len(assignment_proposal_distr), size=1, replace=True, p=assignment_proposal_distr)
                unique_sample_counts[sampled_idx[0]] += 1
                total_sample_count += 1

            sampled_assignments = []
            assignment_importance_weights = []
            for (sampled_idx, idx_sample_count) in unique_sample_counts.iteritems():
                sampled_assignments.append(best_assignments[sampled_idx])
                assignment_importance_weights.append(float(idx_sample_count)/float(total_sample_count))

            print assignment_importance_weights
            best_assignments = sampled_assignments




############    else: #params.SPEC['proposal_distr'] == 'modified_SIS_exact'
############        #now we sample without replacemenent from the most likely assignments
############        best_assignments = k_best_assign_mult_cost_matrices(params.SPEC['num_top_hypotheses_to_sample_from'], perturbed_cost_matrices, particle_costs, M)        
############        
############        assignment_proposal_distr = []
############        for (cur_cost, cur_assignment, cur_particle_idx) in best_assignments:
############            T = len(ordered_particle_groups[cur_particle_idx].targets.living_targets)            
############            cur_assignment_matrix = convert_assignment_pairs_to_matrix3(cur_assignment, M, T)
############
############            assignment_log_prob = np.trace(np.dot(log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T))
############            assignment_prob = np.exp(assignment_log_prob - particle_neg_log_probs[cur_particle_idx])
############            assignment_proposal_distr.append(assignment_prob)
############
############        assert(sum(assignment_proposal_distr) <= 1.0)
############        assignment_proposal_distr.append(1.0 - sum(assignment_proposal_distr))
############        while True:
############            sampled_assignment_indices = np.random.choice(len(assignment_proposal_distr), size=(len(particle_set)), replace=False, p=assignment_proposal_distr)
############            print assignment_proposal_distr
############            print sampled_assignment_indices
############            if not len(best_assignments) in sampled_assignment_indices:
############                break
############            else:
############                invalid_low_prob_sample_count += 1
############
############        sampled_assignments = []
############        for sampled_idx in sampled_assignment_indices:
############            sampled_assignments.append(best_assignments[sampled_idx])
############        best_assignments = sampled_assignments
    #5. For each of the most likely assignments, create a new particle that is a copy of its particle GROUP, 
    # and associate measurements / kill targets according to assignment.

    for (idx, (cur_cost, cur_assignment, cur_particle_idx)) in enumerate(best_assignments):
#        if cur_cost > 1000000000:
#            break #invalid assignment, we've exhausted all valid assignments
        assert(cur_cost < INFEASIBLE_COST)
        #3. create a new particle that is a copy of the max group, and associate measurements / kill
        #targets according to the max x_k from 2.
        assert(SPEC['use_general_num_dets'])
        new_particle = ordered_particle_groups[cur_particle_idx].create_child()
        birth_value = new_particle.targets.living_count

        T = len(new_particle.targets.living_targets)
        assert(T == new_particle.targets.living_count)

        cur_assignment_matrix = convert_assignment_pairs_to_matrix3(cur_assignment, M, T)
        assert(SPEC['normalize_log_importance_weights'] == True)
        #set to log of importance weight
        assignment_log_prob = np.trace(np.dot(log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T))
        if params.SPEC['proposal_distr'] == 'modified_SIS_gumbel':
            new_particle.importance_weight = assignment_log_prob - particle_neg_log_probs[cur_particle_idx] #log prob
            if math.isnan(new_particle.importance_weight):
                print np.isnan(log_prob_matrices[cur_particle_idx]).any()
                random_number = np.random.random()
                matrix_file_name = './inspect_matrices%f' % random_number
                print "saving matrices in %s" % matrix_file_name
                f = open(matrix_file_name, 'w')
                pickle.dump((log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T), f)
                f.close()                  

            if np.isnan(log_prob_matrices[cur_particle_idx]).any():
                for ii in range(log_prob_matrices[cur_particle_idx].shape[0]):
                    for jj in range(log_prob_matrices[cur_particle_idx].shape[1]):
                        if np.isnan(log_prob_matrices[cur_particle_idx][ii][jj]):
                            print "isnan at location:", ii, jj
            assert(not math.isnan(new_particle.importance_weight)), (assignment_log_prob, particle_neg_log_probs[cur_particle_idx], log_prob_matrices[cur_particle_idx], cur_assignment_matrix.T)


        elif params.SPEC['proposal_distr'] == 'modified_SIS_wo_replacement_approx':
            #-sample 'num_particles' hypotheses from 'num_top_hypotheses_to_sample_from' without replacement,
            #and ignore the fact that when sampling w/o replacement samples are dependent and we need to
            #calculate the proposal probability and reweight particles by dividing by this prob
            new_particle.importance_weight = assignment_log_prob - particle_neg_log_probs[cur_particle_idx] #log prob

        elif params.SPEC['proposal_distr'] == 'modified_SIS_w_replacement':
            #-sample 'num_particles' hypotheses from 'num_top_hypotheses_to_sample_from' with replacement,
            #all particle weights should be same, parents should already be correct, just double check
            assert(np.abs(new_particle.importance_weight - 1.0/len(particle_set)) < .000001)

        else:
            assert(params.SPEC['proposal_distr'] == 'modified_SIS_w_replacement_unique')
            new_particle.importance_weight = assignment_importance_weights[idx]



############        else: #params.SPEC['proposal_distr'] == 'modified_SIS_exact'
############            exact_log_prob = assignment_log_prob - particle_neg_log_probs[cur_particle_idx] #log prob
############            proposal_log_prob = np.log(calc_prop_prob(assignment_proposal_distr, sampled_assignment_indices[idx], len(particle_set)))
############            new_particle.importance_weight = exact_prob - proposal_log_prob #log prob
       
        (meas_grp_associations, dead_target_indices) = convert_assignment_matrix3(cur_assignment_matrix, M, T)


        if PRINT_INFO:
            print "just set importance weight for new particle to:", new_particle.importance_weight
            print 'M =', M
            print 'T =', T
            print 'cur_cost:', cur_cost
            print 'cur_assignment:', cur_assignment
            print 'perturbed_cost_matrices[cur_particle_idx]:', perturbed_cost_matrices[cur_particle_idx]
            print 'unperturbed log_prob_matrix:', log_prob_matrices
            print 'parent_particle idx:', cur_particle_idx
            print 'meas_grp_associations: ', meas_grp_associations
            print 'dead_target_indices: ', dead_target_indices
    ############ MESSY ############
        #list of detection group centers, meas_grp_means[i] is a 2-d numpy array
        #of the position of meas_groups[i]
        meas_grp_covs = []   
        meas_grp_means2D = []
        meas_grp_means = []
        for (index, detection_group) in enumerate(meas_groups):
            (combined_meas_mean, combined_covariance) = combine_4d_detections(params.posAndSize_inv_covariance_blocks, 
                                params.meas_noise_mean, detection_group)
            combined_meas_pos = combined_meas_mean[0:2]
            meas_grp_means2D.append(combined_meas_pos)
            meas_grp_means.append(combined_meas_mean)
            meas_grp_covs.append(combined_covariance)
    ############ END MESSY ############


        new_particle.all_measurement_associations.append(meas_grp_associations)
        new_particle.all_dead_targets.append(dead_target_indices)  
        assert(len(meas_grp_associations) == len(meas_grp_means) and len(meas_grp_means) == len(meas_grp_covs))
        for meas_grp_idx, meas_grp_assoc in enumerate(meas_grp_associations):
            new_particle.process_meas_grp_assoc(birth_value, meas_grp_assoc, meas_grp_means[meas_grp_idx], meas_grp_covs[meas_grp_idx], cur_time)

        #process target deaths
        #double check dead_target_indices is sorted
        assert(all([dead_target_indices[i] <= dead_target_indices[i+1] for i in xrange(len(dead_target_indices)-1)]))
        #important to delete larger indices first to preserve values of the remaining indices
        for index in reversed(dead_target_indices):
            new_particle.targets.kill_target(index)

        #checking if something funny is happening
        original_num_targets = birth_value
        num_targets_born = 0
        num_targets_born = meas_grp_associations.count(birth_value)
        num_targets_killed = len(dead_target_indices)
        assert(new_particle.targets.living_count == original_num_targets + num_targets_born - num_targets_killed)
        #done checking if something funny is happening

        new_particle_set.append(new_particle)
        assert(new_particle.parent_particle != None)

    return (new_particle_set, invalid_low_prob_sample_count)

def exact_sampling_step(particle_set, measurement_lists, widths, heights, cur_time, params):
    '''
    perform exact sampling without replacement using upper bounds on the permanent
    '''
    #housekeeping, should make this nicer somehow
    print '-'*80
    print 'beginning exact_sampling_step for cur_time =', cur_time
    
    meas_groups = []
    for det_idx, det_name in enumerate(SPEC['det_names']):
        group_detections(meas_groups, det_name, measurement_lists[det_idx], widths[det_idx], heights[det_idx], params)

    M = len(meas_groups) #number of measurement groups
    print 'M =', M    
    print 
    
    #now that we have estimates of p(x_1:k-1|y_1:k-1), perform modified SIS step
    particle_group_log_probs = {}
#    for idx in range(N_PARTICLES): 

    invalid_low_prob_sample_count = 0 

    zero_assignments = [] #particle group(s) exist with zero targets and we have zero measurements

    all_association_matrices = []
    for particle in particle_set:
        T = len(particle.targets.living_targets)
        print "T =", T
        assert(T == particle.targets.living_count)
        #should clean this up
        p_target_deaths = []
        for target in particle.targets.living_targets:
            p_target_deaths.append(target.death_prob)
            assert(p_target_deaths[len(p_target_deaths) - 1] >= 0 and p_target_deaths[len(p_target_deaths) - 1] <= 1)


        (cur_log_probs, conditional_birth_probs, conditional_death_probs) = construct_log_probs_matrix4(particle, meas_groups, T, p_target_deaths, params)
        assert((cur_log_probs <= .000001).all()), (cur_log_probs)

        cur_a_matrix = associationMatrix(matrix=np.exp(cur_log_probs), M=M, T=T,\
            conditional_birth_probs=conditional_birth_probs, conditional_death_probs=conditional_death_probs,\
            prior_prob=particle.importance_weight)
        all_association_matrices.append(cur_a_matrix)

    sampled_associations = multi_matrix_sample_associations_without_replacement(num_samples=len(particle_set), all_association_matrices=all_association_matrices)


    new_particle_set = []
    for sampled_association in sampled_associations:
        meas_grp_associations = sampled_association.meas_grp_associations
        dead_target_indices = sampled_association.dead_target_indices

        dead_target_indices.sort()

    ############ MESSY ############
        #list of detection group centers, meas_grp_means[i] is a 2-d numpy array
        #of the position of meas_groups[i]
        meas_grp_covs = []   
        meas_grp_means2D = []
        meas_grp_means = []
        for (index, detection_group) in enumerate(meas_groups):
            (combined_meas_mean, combined_covariance) = combine_4d_detections(params.posAndSize_inv_covariance_blocks, 
                                params.meas_noise_mean, detection_group)
            combined_meas_pos = combined_meas_mean[0:2]
            meas_grp_means2D.append(combined_meas_pos)
            meas_grp_means.append(combined_meas_mean)
            meas_grp_covs.append(combined_covariance)
    ############ END MESSY ############

        new_particle = particle_set[sampled_association.matrix_index].create_child()
        T = len(new_particle.targets.living_targets)

        new_particle.importance_weight = sampled_association.complete_assoc_probability
        new_particle.all_measurement_associations.append(meas_grp_associations)
        new_particle.all_dead_targets.append(dead_target_indices)  
        assert(len(meas_grp_associations) == len(meas_grp_means) and len(meas_grp_means) == len(meas_grp_covs))
        for meas_grp_idx, meas_grp_assoc in enumerate(meas_grp_associations):
            new_particle.process_meas_grp_assoc(T, meas_grp_assoc, meas_grp_means[meas_grp_idx], meas_grp_covs[meas_grp_idx], cur_time)

        #process target deaths
        #double check dead_target_indices is sorted
        assert(all([dead_target_indices[i] <= dead_target_indices[i+1] for i in xrange(len(dead_target_indices)-1)])), dead_target_indices
        #important to delete larger indices first to preserve values of the remaining indices
        for index in reversed(dead_target_indices):
            new_particle.targets.kill_target(index)

        #checking if something funny is happening
        original_num_targets = T
        num_targets_born = 0
        num_targets_born = meas_grp_associations.count(T)
        num_targets_killed = len(dead_target_indices)
        assert(new_particle.targets.living_count == original_num_targets + num_targets_born - num_targets_killed)
        #done checking if something funny is happening

        new_particle_set.append(new_particle)

    return (new_particle_set, invalid_low_prob_sample_count)



def modified_SIS_min_cost_proposal_step(particle_set, measurement_lists, widths, heights, cur_time, params):
    (particle_group_probs, particle_groups) = group_particles(particle_set, 0, SPEC['ONLINE_DELAY'])
    #housekeeping, should make this nicer somehow
    meas_groups = []
    for det_idx, det_name in enumerate(SPEC['det_names']):
        group_detections(meas_groups, det_name, measurement_lists[det_idx], widths[det_idx], heights[det_idx], params)


    #construct distribution over particles and measurement target associations
    proposal_distr = [] 
    marginal_proposal_info = []
    for p_key, particle in particle_groups.iteritems():
        (meas_grp_means4D, meas_grp_covs, marginal_meas_target_proposal_distr, proposal_measurement_target_associations) = \
        unnormalized_marginal_meas_target_assoc(particle, meas_groups, len(particle.targets.living_targets), params)
        for m_t_prop_idx, associations in enumerate(proposal_measurement_target_associations):
            proposal_distr.append(particle_group_probs[p_key]*marginal_meas_target_proposal_distr[m_t_prop_idx])
            marginal_proposal_info.append({'particle_key': p_key,
                                  'measurement_target_associations': associations})

    proposal_distr = np.asarray(proposal_distr)
    assert(np.sum(proposal_distr) != 0.0)
    proposal_distr /= float(np.sum(proposal_distr))    

    new_particle_set = []
    particle_group_log_probs = {}
    for idx in range(N_PARTICLES):

        #1. sample particle and measurement-target associations
        sampled_part_assoc_idx = np.random.choice(len(proposal_distr), p=proposal_distr)
        proposal_probability = proposal_distr[sampled_part_assoc_idx] 
        sampled_particle_key = marginal_proposal_info[sampled_part_assoc_idx]['particle_key']
        sampled_assoc = marginal_proposal_info[sampled_part_assoc_idx]['measurement_target_associations'][:] #make a deep copy

        #get a list of all possible measurement groups, length (2^#measurement sources)-1, each detection source can be in the set
        #or out of the set, but we can't have the empty set
        detection_groups = params.get_all_possible_measurement_groups()
        remaining_meas_count_by_groups = defaultdict(int)
        unassociated_meas_indices_by_groups = defaultdict(list)
        #count remaining measurements by measurement sources present in the group
        for (index, meas_group) in enumerate(meas_groups):
            if sampled_assoc[index] == -1:
                remaining_meas_count_by_groups[get_immutable_set_meas_names(meas_group)] += 1
                unassociated_meas_indices_by_groups[get_immutable_set_meas_names(meas_group)].append(index)
        total_birth_count = 0
        total_clutter_count = 0
        total_target_count = len(particle_groups[sampled_particle_key].targets.living_targets)
        conditional_proposals = conditional_birth_clutter_distribution(remaining_meas_count_by_groups, params)
        for meas_group, conditional_proposal_info in conditional_proposals.iteritems():
            # 2. sample # of births and clutter conditioned on 1. or each measurement group type
            sampled_birth_count_idx = np.random.choice(len(conditional_proposal_info['proposal_distribution']),
                                                        p=conditional_proposal_info['proposal_distribution'])
            sampled_birth_count = conditional_proposal_info['birth_counts'][sampled_birth_count_idx]
            sampled_clutter_count = remaining_meas_count_by_groups[meas_group] - sampled_birth_count
            total_birth_count += sampled_birth_count
            total_clutter_count += sampled_clutter_count
            birth_count_proposal_prob = conditional_proposal_info['proposal_distribution'][sampled_birth_count_idx]
            proposal_probability *= birth_count_proposal_prob

            # 3. uniformly sample which unassociated measurements are birth/clutter according to the counts from 2.
            unassociated_measurements = unassociated_meas_indices_by_groups[meas_group]
            proposal_probability *= nCr(len(unassociated_measurements), sampled_birth_count)
            for b_c_idx in range(sampled_birth_count):
                sampled_birth_idx = np.random.choice(len(unassociated_measurements))
                sampled_assoc[unassociated_measurements[sampled_birth_idx]] = total_target_count #set to birth val
                del unassociated_measurements[sampled_birth_idx]
     
        assert(sampled_assoc.count(total_target_count) == total_birth_count), (sampled_assoc.count(total_target_count), total_birth_count, sampled_assoc, idx, cur_time, total_target_count)
        assert(sampled_assoc.count(-1) == total_clutter_count)

        # 4. sample target deaths from unassociated targets
        unassociated_targets = []
        unassociated_target_death_probs = []
        p_target_deaths = []
        for target in particle_groups[sampled_particle_key].targets.living_targets:
            p_target_deaths.append(target.death_prob)
            assert(p_target_deaths[len(p_target_deaths) - 1] >= 0 and p_target_deaths[len(p_target_deaths) - 1] <= 1)

        for i in range(total_target_count):
            if i in sampled_assoc:
                target_unassociated = False
            else:
                target_unassociated = True            
            if target_unassociated:
                unassociated_targets.append(i)
                unassociated_target_death_probs.append(p_target_deaths[i])
            else:
                unassociated_target_death_probs.append(0.0)

        (targets_to_kill, death_probability) =  \
            sample_target_deaths(particle_groups[sampled_particle_key], unassociated_targets, cur_time)


        #probability of sampling all associations
        proposal_probability *= death_probability
        assert(proposal_probability != 0.0)


        # 5. create a new particle that is a copy of the max group, and associate measurements / kill
        #targets according to the max x_k from 2.
        assert(SPEC['use_general_num_dets'])
        new_particle = particle_groups[sampled_particle_key].create_child()
        #calc importance weight
        living_target_indices = []
        unassociated_target_indices = []
        for i in range(new_particle.targets.living_count):
            if(not i in targets_to_kill):
                living_target_indices.append(i)
            ####DONE DEBUGGING#######
            else:
                assert(p_target_deaths[i] > 0.0), p_target_deaths
            ####DONE DEBUGGING#######
            if(not i in sampled_assoc):
                unassociated_target_indices.append(i)

        assert(unassociated_target_indices == unassociated_targets), (unassociated_target_indices, unassociated_targets, new_particle.targets.living_count, new_particle.targets.living_targets)

        # a list containing the number of measurements detected by each source
        # used in prior calculation to count the number of ordered vectors given
        # an unordered association set
        meas_counts_by_source = [] 
        for meas_list in measurement_lists:
            meas_counts_by_source.append(len(meas_list))

        likelihood = get_likelihood(particle_groups[sampled_particle_key], meas_groups, particle_groups[sampled_particle_key].targets.living_count,
                                       sampled_assoc, params, log=False)
        assoc_prior = get_assoc_prior(particle_groups[sampled_particle_key].targets.living_count, meas_groups, sampled_assoc, params, meas_counts_by_source, log=False)
        death_prior = calc_death_prior(living_target_indices, p_target_deaths, unassociated_target_indices, log=False)
        exact_probability = likelihood * assoc_prior * death_prior        
        new_particle.importance_weight = particle_group_probs[sampled_particle_key]*exact_probability/proposal_probability
        birth_value = new_particle.targets.living_count


    ############ MESSY ############
        #list of detection group centers, meas_grp_means[i] is a 2-d numpy array
        #of the position of meas_groups[i]
        meas_grp_covs = []   
        meas_grp_means2D = []
        meas_grp_means = []
        for (index, detection_group) in enumerate(meas_groups):
            (combined_meas_mean, combined_covariance) = combine_4d_detections(params.posAndSize_inv_covariance_blocks, 
                                params.meas_noise_mean, detection_group)
            combined_meas_pos = combined_meas_mean[0:2]
            meas_grp_means2D.append(combined_meas_pos)
            meas_grp_means.append(combined_meas_mean)
            meas_grp_covs.append(combined_covariance)
    ############ END MESSY ############

        sampled_assoc = sampled_assoc
        dead_target_indices = targets_to_kill


        new_particle.all_measurement_associations.append(sampled_assoc)
        new_particle.all_dead_targets.append(dead_target_indices)  
        assert(len(sampled_assoc) == len(meas_grp_means) and len(meas_grp_means) == len(meas_grp_covs))
        for meas_grp_idx, meas_grp_assoc in enumerate(sampled_assoc):
            new_particle.process_meas_grp_assoc(birth_value, meas_grp_assoc, meas_grp_means[meas_grp_idx], meas_grp_covs[meas_grp_idx], cur_time)

        #process target deaths
        #double check dead_target_indices is sorted
        assert(all([dead_target_indices[i] <= dead_target_indices[i+1] for i in xrange(len(dead_target_indices)-1)]))
        #important to delete larger indices first to preserve values of the remaining indices
        for index in reversed(dead_target_indices):
            new_particle.targets.kill_target(index)

        #checking if something funny is happening
        original_num_targets = birth_value
        num_targets_born = 0
        num_targets_born = sampled_assoc.count(birth_value)
        num_targets_killed = len(dead_target_indices)
        assert(new_particle.targets.living_count == original_num_targets + num_targets_born - num_targets_killed)
        #done checking if something funny is happening
        new_particle_set.append(new_particle)
        assert(new_particle.parent_particle != None)


    return new_particle_set




def run_rbpf_on_targetset(target_sets, online_results_filename, params, fw_spec, filename=None):
    """
    Measurement class designed to only have 1 measurement/time instance
    Input:
    - target_sets: a list where target_sets[i] is a TargetSet containing measurements from
        the ith measurement source
    - filename: (string) filename to write run info to
    Output:
    - max_weight_target_set: TargetSet from a (could be multiple with equal weight) maximum
        importance weight particle after processing all measurements
    - number_resamplings: the number of times resampling was performed
    """
    particle_set = []
    global NEXT_PARTICLE_ID
    #Create the particle set
    for i in range(0, N_PARTICLES):
        particle_set.append(Particle(NEXT_PARTICLE_ID, fw_spec))
        NEXT_PARTICLE_ID += 1
    prev_time_stamp = -1


    #for displaying results
    time_stamps = []
    positions = []

    iter = 0 # for plotting only occasionally
    number_resamplings = 0

    online_conditional_log_probability_sum = 0
    #debugging, checking how serious a bug was
    incorrect_max_weight_particle_count = 0

    if params.SPEC['train_test'] == 'generated_data':
        #we might not have generated a measurement on every time instance
        number_time_instances = params.SPEC['data_generation_spec']['num_time_steps']
    else:
        number_time_instances = len(target_sets[0].measurements)
        for target_set in target_sets:
            assert(len(target_set.measurements) == number_time_instances)

    #the particle with the maximum importance weight on the previous time instance 
    prv_max_weight_particle = None

    invalid_low_prob_sample_count = 0

    min_meas_idx = 0
    for time_instance_index in range(number_time_instances):
        measurement_lists = []
        widths = []
        heights = []
        measurement_scores = []

        if params.SPEC['train_test'] == 'generated_data':
            time_stamp = time_instance_index*DEFAULT_TIME_STEP
            measurement_index = -1
            for m_idx in range(min_meas_idx, len(target_sets[0].measurements)):
                if time_stamp == target_sets[0].measurements[m_idx].time:
                    measurement_index = m_idx
                    min_meas_idx = m_idx + 1
                if time_stamp < target_sets[0].measurements[m_idx].time:
                    break
            if measurement_index != -1: #we generated measurements for this time_stamp
                for target_set in target_sets:            
                    measurement_lists.append(target_set.measurements[measurement_index].val)
                    widths.append(target_set.measurements[measurement_index].widths)
                    heights.append(target_set.measurements[measurement_index].heights)
                    measurement_scores.append(target_set.measurements[measurement_index].scores)
            else: #append empty lists
                for target_set in target_sets:
                    measurement_lists.append([])
                    widths.append([])
                    heights.append([])
                    measurement_scores.append([])

        else:
            time_stamp = target_sets[0].measurements[time_instance_index].time
            for target_set in target_sets:
                assert(target_set.measurements[time_instance_index].time == time_stamp)

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
                    ###############if params.SPEC['train_test'] != 'generated_data': #we might not generate data for a particular time step
                    assert(abs(dt - DEFAULT_TIME_STEP) < .00000001), (dt, DEFAULT_TIME_STEP, time_stamp, prev_time_stamp)
                    target.predict(dt, time_stamp)
                #update particle death probabilities AFTER predict so that targets that moved
                #off screen this time instance will be killed
                particle.update_target_death_probabilities(time_stamp, prev_time_stamp)

                #######kill offscreen targets BEFORE measurement associations##########
                particle.targets.kill_offscreen_targets()
            



        new_target_list = [] #for debugging, list of booleans whether each particle created a new target
        pIdxDebugInfo = 0

        if params.SPEC['proposal_distr'] in ['exact_sampling']:
            (particle_set, cur_invalid_low_prob_sample_count) = exact_sampling_step(particle_set, measurement_lists, widths, heights, time_stamp, params)
        elif params.SPEC['proposal_distr'] in ['modified_SIS_gumbel', 'modified_SIS_wo_replacement_approx', 'modified_SIS_w_replacement', 'modified_SIS_w_replacement_unique']:
            (particle_set, cur_invalid_low_prob_sample_count) = modified_SIS_MHT_gumbel_step(particle_set, measurement_lists, widths, heights, time_stamp, params)
            invalid_low_prob_sample_count += cur_invalid_low_prob_sample_count
#            particle_set = modified_SIS_gumbel_step(particle_set, measurement_lists, widths, heights, time_stamp, params)
        elif params.SPEC['proposal_distr'] == 'modified_SIS_min_cost':
            particle_set = modified_SIS_min_cost_proposal_step(particle_set, measurement_lists, widths, heights, time_stamp, params)
        else:
            for particle in particle_set:
                #this is where 
                assert(len(particle.all_measurement_associations) == time_instance_index), (particle.all_measurement_associations, len(particle.all_measurement_associations), time_instance_index)
                new_target = particle.update_particle_with_measurement(time_stamp, measurement_lists, widths, heights, measurement_scores, params)
                assert(len(particle.all_measurement_associations) == time_instance_index + 1), (particle.all_measurement_associations, len(particle.all_measurement_associations), time_instance_index)            
                new_target_list.append(new_target)
                pIdxDebugInfo += 1

        # if not params.SPEC['proposal_distr'] in ['modified_SIS_exact','modified_SIS_w_replacement', 'modified_SIS_w_replacement_unique', 'exact_sampling']:
        if not params.SPEC['proposal_distr'] in ['modified_SIS_exact','modified_SIS_w_replacement', 'modified_SIS_w_replacement_unique']:
            print "about to normalize importance weights"
            #Using MHT sampling without replacement, we care about importance weights, normalize so they don't get too small            
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
                #find the particle that currently has the largest importance weight, this is WRONG, correct below!!

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
                    assert(cur_max_weight_particle), (max_imprt_weight, len(particle_set))
                    print "max weight particle id = ", cur_max_weight_particle.id_


                (particle_group_probs, particle_groups) = group_particles(particle_set, SPEC['ONLINE_DELAY'], SPEC['ONLINE_DELAY'])
                print '#'*80
                print len(particle_groups), 'particle_groups'

                MAP_particle_key = None
                MAP_particle_prob = -1
                for key, prob in particle_group_probs.iteritems():
                    print len(particle_groups[key].targets.living_targets), 'living targets in particle with probability', prob
                    if prob > MAP_particle_prob:
                        MAP_particle_prob = prob
                        MAP_particle_key = key
                #important to have != None, assert(()) on the empty tuple produces an error                    
                assert(MAP_particle_key != None), particle_group_probs
                MAP_particle = particle_groups[MAP_particle_key]
                (match_bool, match_str) = cur_particle_states_match(MAP_particle, cur_max_weight_particle, SPEC['ONLINE_DELAY'], SPEC['ONLINE_DELAY'])
                if not match_bool:
                    incorrect_max_weight_particle_count += 1

                #Really, this is the MAP particle group, would be better to change names after checking severity of previous problem
                cur_max_weight_particle = MAP_particle
                cur_max_weight_target_set = cur_max_weight_particle.targets
                online_conditional_log_probability_sum += cur_max_weight_particle.cur_conditional_log_prob


            if params.SPEC['proposal_distr'] in ['exact_sampling', 'modified_SIS_gumbel', 'modified_SIS_min_cost', 'modified_SIS_wo_replacement_approx', 'modified_SIS_w_replacement', 'modified_SIS_w_replacement_unique']:
                if time_instance_index>0 and cur_max_weight_particle.parent_particle != prv_max_weight_particle:
                    (target_associations, duplicate_ids) = \
                    match_target_ids(cur_max_weight_particle.parent_particle.targets.living_targets,\
                                     prv_max_weight_particle.targets.living_targets)
                    #replace associated target IDs with the IDs from the previous maximum importance weight
                    #particle for ID conistency in the online results we output
#                    for cur_target in cur_max_weight_target_set.living_targets:
#                        if cur_target.id_ in target_associations:
#                            cur_target.id_ = target_associations[cur_target.id_]                   
                    for q_idx in range(SPEC['ONLINE_DELAY'] + 1):
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

            else:
#                if prv_max_weight_particle != None:
#                    (match_bool, match_str) = cur_particle_states_match(prv_max_weight_particle, cur_max_weight_particle)
#                if prv_max_weight_particle != None and not match_bool:
                if prv_max_weight_particle != None and prv_max_weight_particle != cur_max_weight_particle:
##########                    if SPEC['ONLINE_DELAY'] == 0:
##########                        assert(len(cur_max_weight_target_set.living_targets_q) == 1 and
##########                               len(prv_max_weight_particle.targets.living_targets_q) == 1)
##########                        #associate targets from the PREVIOUS time step
##########                        (target_associations, duplicate_ids) = match_target_ids(cur_max_weight_target_set.living_targets_q[0][1],\
##########                                                               prv_max_weight_particle.targets.living_targets_q[0][1])
###########                        (target_associations, duplicate_ids) = match_target_ids(cur_max_weight_target_set.living_targets,\
###########                                                               prv_max_weight_particle.targets.living_targets)
##########
##########                        #replace associated target IDs with the IDs from the previous maximum importance weight
##########                        #particle for ID conistency in the online results we output
##########                        for cur_target in cur_max_weight_target_set.living_targets:
##########                            if cur_target.id_ in target_associations:
##########                                cur_target.id_ = target_associations[cur_target.id_]
###########                    elif time_instance_index >= SPEC['ONLINE_DELAY']:
                    (target_associations, duplicate_ids) = match_target_ids(cur_max_weight_target_set.living_targets_q[0][1],\
                                                           prv_max_weight_particle.targets.living_targets_q[0][1])
                    #replace associated target IDs with the IDs from the previous maximum importance weight
                    #particle for ID conistency in the online results we output
                    for q_idx in range(SPEC['ONLINE_DELAY'] + 1):
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


            bool_first_time_as_max_imprt_part = prv_max_weight_particle != cur_max_weight_particle and prv_max_weight_particle != cur_max_weight_particle.parent_particle

            #write current time step's results to results file
            if time_instance_index >= SPEC['ONLINE_DELAY']:
                extra_info = {'importance_weight': max_imprt_weight,
                          'first_time_as_max_imprt_part': bool_first_time_as_max_imprt_part,
                          'MAP_particle_prob': MAP_particle_prob,
                          'sampled_meas_targ_assoc_idx': MAP_particle.sampled_meas_targ_assoc_idx}
                frm_idx = int(round(time_stamp/DEFAULT_TIME_STEP))
                print "time_stamp =", time_stamp
                print "frm_idx =", frm_idx
                cur_max_weight_target_set.write_online_results(online_results_filename, frm_idx, number_time_instances,
                                            extra_info, fw_spec)

            if time_instance_index >= SPEC['ONLINE_DELAY']:
                prv_max_weight_particle = cur_max_weight_particle


            print "popped on time_instance_index", time_instance_index
            for particle in particle_set:
                particle.targets.living_targets_q.popleft()

            for particle in particle_set:
                particle.targets.living_targets_q.append((time_instance_index, copy.deepcopy(particle.targets.living_targets)))
        
        else: #we're running in offline mode
            for particle in particle_set:
                particle.targets.living_targets_q.append((time_instance_index, copy.deepcopy(particle.targets.living_targets)))

        #Using modified_SIS_MHT_gumbel, sampling with replacement, importance weights may vary but DON'T resample because sampling
        #was done without replacement
        if (not params.SPEC['proposal_distr'] in ['exact_sampling', 'modified_SIS_gumbel', 'modified_SIS_wo_replacement_approx', 'modified_SIS_w_replacement', 'modified_SIS_w_replacement_unique'] \
            and get_eff_num_particles(particle_set) < N_PARTICLES/RESAMPLE_RATIO):

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
            print "checking importance weights, cur importance_weight:", particle.importance_weight
            if(particle.importance_weight == cur_max_imprt_weight):
                particle.max_importance_weight = True
                particle_count_with_max_imprt_weight += 1
        print particle_count_with_max_imprt_weight, "particles have max importance weight of", cur_max_imprt_weight
        max_imprt_weight_count_dict[particle_count_with_max_imprt_weight] += 1
#END DEBUGGING

    #FIX ME, should return the a target set representing our map estimate
    #not the particle set from the particle with maximum importance weight.
    max_imprt_weight = -1
    for particle in particle_set:
        if(particle.importance_weight > max_imprt_weight):
            max_imprt_weight = particle.importance_weight
    max_weight_target_set = None
    for particle in particle_set:
        if(particle.importance_weight == max_imprt_weight):
            max_weight_target_set = particle.targets
            cur_max_weight_particle = particle

    run_info = [number_resamplings]

    stdout = sys.stdout
    sys.stdout = open(filename, 'a')
    print "Using the max_weight importance weight we would have made mistakes on", incorrect_max_weight_particle_count,\
        "out of", number_time_instances, "time instances"
    print "log probability of final MAP particle=", cur_max_weight_particle.exact_log_probability
    print "sum of online MAP particle log probabilities =", online_conditional_log_probability_sum
    sys.stdout.close()
    sys.stdout = stdout


    return (max_weight_target_set, run_info, number_resamplings, incorrect_max_weight_particle_count, number_time_instances, invalid_low_prob_sample_count)



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
#    lin_assign = linear_assignment.LinearAssignment(cost_matrix)
#    solution = lin_assign.solution
#    association_list = zip([i for i in range(len(solution))], solution)    
    association_list = hm.compute(cost_matrix)

    associations = {}
    for row,col in association_list:
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
        print "results folder: ", fw_spec['results_folder']
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

        print "fw_spec['obj_class']:", fw_spec['obj_class']

        print "SPEC:"
        print SPEC
        
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

        if SPEC['train_test'] == 'train':
            filename_mapping = fw_spec['data_path'] + "/evaluate_tracking.seqmap"
        elif SPEC['train_test'] == 'test':
            filename_mapping = fw_spec['data_path'] + "/evaluate_tracking.seqmap.test"

        if SPEC['train_test'] == 'generated_data':
            n_frames = SPEC['data_generation_spec']['num_time_steps']
            sequence_name = "%04d" % int(seq_idx)
        else:
            n_frames         = []
            sequence_names    = []
            with open(filename_mapping, "r") as fh:
                for i,l in enumerate(fh):
                    fields = l.split(" ")
                    sequence_names.append("%04d" % int(fields[0]))
                    n_frames.append(int(fields[3]) - int(fields[2]))
            fh.close() 
            sequence_name = sequence_names[seq_idx]
            print n_frames
            print sequence_names   
            assert(len(n_frames) == len(sequence_names))

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
                    'rrc' : [float(i)*.1 for i in range(10)],              
                    'mscnn' : [float(i)*.1 for i in range(3,10)],              
                    'subcnn' : [float(i)*.1 for i in range(3,10)],              
                    'regionlets' : [i for i in range(2, 20)],
                    '3dop' : [float(i)*.1 for i in range(2,10)],            
                    'mono3d' : [float(i)*.1 for i in range(2,10)],            
                    'mv3d' : [float(i)*.1 for i in range(2,10)],
                    'single_det_src': [float(i)*.1 for i in range(10)],
                    'DPM': [float(i)*.1 for i in range(10)], 
                    'FRCNN': [float(i)*.1 for i in range(10)], 
                    'SDP': [float(i)*.1 for i in range(10)]}        
    #            'regionlets' = [i for i in range(2, 16)]
            else:
                score_interval_dict_all_det = {\
    #            'mscnn' = [.5],                                
                'rrc' : [0],
                'mscnn' : [.3],
                'subcnn' : [.3],
                'regionlets' : [2],
                '3dop' : [.2],
                'mono3d' : [.2],
                'mv3d' : [.2],
                'single_det_src': [0],
                'DPM': [0], 
                'FRCNN': [0], 
                'SDP': [0]}


            if SPEC['train_test'] == 'train':
                #train on all training sequences, except the current sequence we are testing on
                training_sequences = [i for i in [i for i in range(fw_spec['training_seq_count'])] if i != seq_idx]
            elif SPEC['train_test'] == 'test':
                #train on all training sequences
                training_sequences = [i for i in range(fw_spec['training_seq_count'])]
            else:
                assert(SPEC['train_test'] == 'generated_data')
                training_sequences = [i for i in range(21)]
    
            def get_gt_objects(fw_spec):
                '''
                call evaluate to get gt_objects, where gt_objects[i][j] is a list of gtObjects in frame j of sequence i  
                '''
                mail = mailpy.Mail("") #this is silly and could be cleaned up
            
                (gt_objects, cur_det_objects) = evaluate(fw_spec, fw_spec['data_path'], fw_spec['pickled_data_dir'], min_score=0, \
                            det_method='DPM', mail=mail, obj_class=fw_spec['obj_class'], include_ignored_gt=False,\
                            include_dontcare_in_gt=False, include_ignored_detections=True)
                return gt_objects

            if SPEC['proposal_distr'] == 'ground_truth_assoc':
                SPEC['gt_obects'] = get_gt_objects(SPEC)

            SCORE_INTERVALS = []
            for det_name in det_names:
                print det_name
                SCORE_INTERVALS.append(score_interval_dict_all_det[det_name])
            p_clutter_likelihood = 1.0/float(fw_spec['image_widths'][fw_spec['seq_idx']]*fw_spec['image_heights'][fw_spec['seq_idx']])
            p_birth_likelihood = 1.0/float(fw_spec['image_widths'][fw_spec['seq_idx']]*fw_spec['image_heights'][fw_spec['seq_idx']])

            if use_general_num_dets:
                #dictionary of score intervals for only detection sources we are using
                SCORE_INTERVALS_DET_USED = {}
                for det_name in det_names:
                    SCORE_INTERVALS_DET_USED[det_name] = score_interval_dict_all_det[det_name]

                (measurementTargetSetsBySequence, target_groupEmission_priors, clutter_grpCountByFrame_priors, clutter_group_priors, clutter_lambdas_by_group,
                birth_count_priors, birth_lambdas_by_group, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES, 
                posAndSize_inv_covariance_blocks, meas_noise_mean, posOnly_covariance_blocks,
                clutter_posAndSize_inv_covariance_blocks, clutter_posOnly_covariance_blocks, clutter_meas_noise_mean_posAndSize) =\
                            get_meas_target_sets_general(fw_spec, fw_spec['obj_class'], fw_spec['data_path'], fw_spec['pickled_data_dir'], training_sequences, SCORE_INTERVALS_DET_USED, det_names, \
                            doctor_clutter_probs = True, doctor_birth_probs = True,\
                            include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
                            include_ignored_detections = include_ignored_detections, death_prob_markov_order = SPEC['death_prob_markov_order'])

                print "BORDER_DEATH_PROBABILITIES: ", BORDER_DEATH_PROBABILITIES
                print "NOT_BORDER_DEATH_PROBABILITIES: ", NOT_BORDER_DEATH_PROBABILITIES

                params = Parameters(det_names, target_groupEmission_priors, clutter_grpCountByFrame_priors,\
                         clutter_group_priors, clutter_lambdas_by_group, birth_count_priors, birth_lambdas_by_group, posOnly_covariance_blocks, \
                         meas_noise_mean, posAndSize_inv_covariance_blocks, SPEC['R'], H,\
                         USE_PYTHON_GAUSSIAN, SPEC['USE_CONSTANT_R'], SCORE_INTERVALS,\
                         p_birth_likelihood, p_clutter_likelihood, SPEC['CHECK_K_NEAREST_TARGETS'],
                         SPEC['K_NEAREST_TARGETS'], SPEC['scale_prior_by_meas_orderings'], SPEC,
                         clutter_posAndSize_inv_covariance_blocks, clutter_posOnly_covariance_blocks, clutter_meas_noise_mean_posAndSize)

#                print "BORDER_DEATH_PROBABILITIES:", BORDER_DEATH_PROBABILITIES
#                print "NOT_BORDER_DEATH_PROBABILITIES:", NOT_BORDER_DEATH_PROBABILITIES
#                sleep(2.5)

            else:
                det1_score_intervals = score_interval_dict_all_det[det1_name]
                if det2_name:
                    det2_score_intervals = score_interval_dict_all_det[det2_name]
                    (measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
                        MEAS_NOISE_COVS, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES, JOINT_MEAS_NOISE_COV) = \
                            get_meas_target_sets_2sources_general(fw_spec, fw_spec['obj_class'], fw_spec['data_path'], fw_spec['pickled_data_dir'], training_sequences, det1_score_intervals, \
                            det2_score_intervals, det1_name, det2_name, obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True,\
                            include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
                            include_ignored_detections = include_ignored_detections)

                else:
                    (measurementTargetSetsBySequence, TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES, BIRTH_PROBABILITIES,\
                        MEAS_NOISE_COVS, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES) = \
                            get_meas_target_sets_1sources_general(fw_spec, fw_spec['obj_class'], fw_spec['data_path'], fw_spec['pickled_data_dir'], training_sequences, det1_score_intervals, \
                            det1_name, obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True,\
                            include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
                            include_ignored_detections = include_ignored_detections)            

                params = Parameters(TARGET_EMISSION_PROBS, CLUTTER_PROBABILITIES,\
                         BIRTH_PROBABILITIES, MEAS_NOISE_COVS, SPEC['R'], H,\
                         USE_PYTHON_GAUSSIAN, SPEC['USE_CONSTANT_R'], SCORE_INTERVALS,\
                         p_birth_likelihood, p_clutter_likelihood, SPEC['CHECK_K_NEAREST_TARGETS'],
                         SPEC['K_NEAREST_TARGETS'], SPEC['scale_prior_by_meas_orderings'], SPEC)

            if SPEC['train_test'] == 'test':
                measurementTargetSetsBySequence = get_meas_target_sets_test(fw_spec['data_path'], SCORE_INTERVALS_DET_USED, det_names, \
                                                                            obj_class = SPEC['obj_class'])

            sequenceMeasurementTargetSet = measurementTargetSetsBySequence[seq_idx]

            if SPEC['train_test'] == 'generated_data':
                filename = "%smeasurements/%04d.txt" % (SPEC['data_generation_spec']['data_file_path'], seq_idx)
                time_per_time_step = SPEC['data_generation_spec']['time_per_time_step']
                sequenceMeasurementTargetSet = [KITTI_detection_file_to_TargetSet(filename, time_per_time_step)]
                if SPEC['data_generation_spec']['data_gen_params']:
                    #use the same parameters that the data was generated from:
                    params.clutter_lambda = SPEC['data_generation_spec']['data_gen_params']['lamda_c']
                    params.birth_lambda = SPEC['data_generation_spec']['data_gen_params']['lamda_b']
                    #assume we only have 1 measurement source
                    num_group_types = 0
                    for key, value in params.target_groupEmission_priors.iteritems():
                        num_group_types+=1
                        if key == ImmutableSet([]):
                            params.target_groupEmission_priors[key] = 1.0 - SPEC['data_generation_spec']['data_gen_params']['p_emission']
                        else:
                            params.target_groupEmission_priors[key] = SPEC['data_generation_spec']['data_gen_params']['p_emission']
                    assert(num_group_types == 2)

                    SPEC['Q'] = SPEC['data_generation_spec']['data_gen_params']['process_noise']
                    assert(SPEC['USE_CONSTANT_R'])
                    SPEC['R'] = SPEC['data_generation_spec']['data_gen_params']['meas_noise_target_state']

                    BORDER_DEATH_PROBABILITIES = SPEC['data_generation_spec']['data_gen_params']['BORDER_DEATH_PROBABILITIES']
                    NOT_BORDER_DEATH_PROBABILITIES= SPEC['data_generation_spec']['data_gen_params']['NOT_BORDER_DEATH_PROBABILITIES']

                    SPEC['P'][0][0] = SPEC['data_generation_spec']['data_gen_params']['meas_noise_target_state'][0][0]
                    SPEC['P'][1][1] = SPEC['data_generation_spec']['init_vel_cov'][0][0]
                    SPEC['P'][2][2] = SPEC['data_generation_spec']['data_gen_params']['meas_noise_target_state'][1][1]
                    SPEC['P'][3][3] = SPEC['data_generation_spec']['init_vel_cov'][1][1]

            else:
                assert(len(n_frames) == len(measurementTargetSetsBySequence))

            t0 = time.time()
            info_by_run = [] #list of info from each run
            cur_run_info = None
            results_filename = '%s/results_by_run/run_%d/%s.txt' % (results_folder, run_idx, sequence_name)
            plot_filename = '%s/results_by_run/run_%d/%s_plot.png' % (results_folder, run_idx, sequence_name)
            measurements_filename = '%s/results_by_run/run_%d/%s_measurements_plot.png' % (results_folder, run_idx, sequence_name)


            print "Processing sequence: ", seq_idx
            tA = time.time()
            if USE_GENERATED_DATA:
                meas_target_set = gen_data(measurements_filename)
                if PROFILE: 
                    cProfile.run('run_rbpf_on_targetset([meas_target_set], results_filename, params, fw_spec)')
                else:

                    (estimated_ts, cur_seq_info, number_resamplings, max_weight_mistakes, max_possible_mistakes, invalid_low_prob_sample_count) = \
                    run_rbpf_on_targetset([meas_target_set], results_filename, params, fw_spec, filename=indicate_run_complete_filename)
            else:  
                if SPEC['condition_emission_prior_img_feat'] == True:
                    if SPEC['train_test'] == 'train':
                        seq_det_feature_arrays = get_deepsort_feature_arrays(seq_names_file='/atlas/u/jkuck/%s/kitti_format/training_seq_names.txt'%SPEC['DATA_SET_NAME'],\
                            det_names=det_names, feature_folder='/atlas/u/jkuck/deep_sort/resources/detections/%s_train'%SPEC['DATA_SET_NAME'])
                    elif SPEC['train_test'] == 'test':
                        seq_det_feature_arrays = get_deepsort_feature_arrays(seq_names_file='/atlas/u/jkuck/%s/kitti_format/testing_seq_names.txt'%SPEC['DATA_SET_NAME'],\
                            det_names=det_names, feature_folder='/atlas/u/jkuck/deep_sort/resources/detections/%s_test'%SPEC['DATA_SET_NAME'])
                    params.SPEC['seq_det_feature_arrays'] = seq_det_feature_arrays                       
                    x = tf.placeholder(tf.float32, shape=[None, SPEC['emission_prior_k_NN']])
                    keep_prob = tf.placeholder(tf.float32)
                    with tf.Session() as sess:
                        priors = load_emmision_prior_model(sess, x, keep_prob)
                        tf_emission_priors = {'priors': priors,
                                              'x': x,
                                              'keep_prob': keep_prob}
                        params.SPEC['tf_emission_priors'] = tf_emission_priors
                        if PROFILE:
                            cProfile.runctx('run_rbpf_on_targetset(sequenceMeasurementTargetSet, results_filename, params, fw_spec)',
                                {'sequenceMeasurementTargetSet': sequenceMeasurementTargetSet,
                                'results_filename':results_filename, 'params':params, 'run_rbpf_on_targetset':run_rbpf_on_targetset, 'fw_spec':fw_spec}, {})
                        else:
                            (estimated_ts, cur_seq_info, number_resamplings, max_weight_mistakes, max_possible_mistakes, invalid_low_prob_sample_count) = \
                                    run_rbpf_on_targetset(sequenceMeasurementTargetSet, results_filename, params, fw_spec, filename=indicate_run_complete_filename)
                else:
                    if PROFILE:
                        cProfile.runctx('run_rbpf_on_targetset(sequenceMeasurementTargetSet, results_filename, params, fw_spec)',
                            {'sequenceMeasurementTargetSet': sequenceMeasurementTargetSet,
                            'results_filename':results_filename, 'params':params, 'run_rbpf_on_targetset':run_rbpf_on_targetset, 'fw_spec':fw_spec}, {})
                    else:
                        (estimated_ts, cur_seq_info, number_resamplings, max_weight_mistakes, max_possible_mistakes, invalid_low_prob_sample_count) = \
                                run_rbpf_on_targetset(sequenceMeasurementTargetSet, results_filename, params, fw_spec, filename=indicate_run_complete_filename)
                    print 'hi there'
                    print 'invalid_low_prob_sample_count =', invalid_low_prob_sample_count

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
                                                           plot_filename = plot_filename, fw_spec = fw_spec)
            print "done write results"
            print "running the rbpf took %f seconds" % (tB-tA)
            
            info_by_run.append(cur_run_info)
            t1 = time.time()

            stdout = sys.stdout
            sys.stdout = open(indicate_run_complete_filename, 'a')

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


#        return FWAction(mod_spec=[{'_inc': {"mistakes_by_max_weight_particle": max_weight_mistakes},
#                                   '_inc': {"max_possible_mistakes": max_possible_mistakes}}])
        return FWAction(mod_spec=[{'_inc': {"invalid_low_prob_sample_count": invalid_low_prob_sample_count, \
                                            "total_runtime": this_seq_run_time, \
                                            "mistakes_by_max_weight_particle": max_weight_mistakes, \
                                            "number_resamplings": number_resamplings}}])



