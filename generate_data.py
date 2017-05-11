import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
from numpy.linalg import inv
from sets import ImmutableSet

import copy
import os
import errno
import sys
from fireworks import Firework, Workflow, FWorker, LaunchPad
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FWAction, FireTaskBase

from fireworks.core.rocket_launcher import rapidfire
#from fireworks.queue.queue_launcher import rapidfire
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from fw_tutorials.dynamic_wf.fibadd_task import FibonacciAdderTask
#from fireworks.core.firework import FWAction, Firework, FiretaskBase
#from fireworks.user_objects.firetasks.script_task import PyTask
from cluster_config import RBPF_HOME_DIRECTORY, MONGODB_USERNAME, MONGODB_PASSWORD
#from experiment_config import MONGODB_HOST, MONGODB_PORT, MONGODB_NAME
sys.path.insert(0, "%sKITTI_helpers" % RBPF_HOME_DIRECTORY)
from jdk_helper_evaluate_results import RunEval
from create_launchpad import create_launchpad
from learn_params1 import get_meas_target_sets_general
from create_launchpad import create_launchpad

sys.path.insert(0, "%sgeneral_tracking" % RBPF_HOME_DIRECTORY)
from targets import TargetState, TargetSet
from measurements import BoundingBox, Measurement

#import genral parameters, e.g. measurement area (image size)
#and time step between measurements
from global_params import *


###################################### Experiment Database ######################################
MONGODB_HOST = 'ds153179.mlab.com'
MONGODB_PORT = 53179
MONGODB_NAME = 'local_testing'
#################################################################################################




def gen_data_exact(measurement_plot_filename):
    mat_measurements = sio.loadmat('measurements.mat')
    print mat_measurements['Y'].shape
    measurements = mat_measurements['Y']
    time_stamps = mat_measurements['T']
    associations = mat_measurements['C']

    assert(measurements.shape[1] == time_stamps.shape[1] and measurements.shape[1] == associations.shape[1])
    returnTargetSet = TargetSet()    
    all_x = []
    all_y = []

    i = 0
    while(i < measurements.shape[1]):

        cur_meas = BoundingBox()
        cur_meas.time = time_stamps[0][i]
        j = 0
        while (i + j < measurements.shape[1] and time_stamps[0][i] == time_stamps[0][i+j]):
            if SCALED:
                x = 300*measurements[0][i+j] + 800
                y = 90*measurements[1][i+j] + 200
            else:
                x = measurements[0][i+j]
                y = measurements[1][i+j]
            cur_meas.val.append(np.array([x, y]))        
            all_x.append(x)
            all_y.append(y)
            cur_meas.widths.append(1)
            cur_meas.heights.append(1)
            cur_meas.scores.append(1)
            j += 1
        if(SHUFFLE_MEASUREMENTS):
            np.random.shuffle(cur_meas.val)

        returnTargetSet.measurements.append(cur_meas)

        i = i + j


    #plot measurements
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(all_x, all_y, '+', markersize=2, label='Target %d' % 1)    
    fig.savefig(measurement_plot_filename)  

    return returnTargetSet



def gen_data_1source(params):
    '''
    Generate synthetic data for 1 measurement source according to the specified parameters.
    The parameters contain the implicit parameters N (the dimension of the measurement
    space) and M (the dimension of the state space).
    Inputs:
    - params: A dictionary specifying the data generation parameters with entries
        * key: 'num_time_steps'
        * value: int, generate this many time steps of data

        * key: 'time_per_time_step'
        * value: float, time between succesive time steps (e.g. in seconds)
        
        * key: 'lamda_c'
        * value: float, Poisson parameter (average #) for clutter counts

        * key: ''lamda_b''
        * value: float, Poisson parameter (average #) for birth counts

        * key: 'p_emission'
        * value: float, the probability a target emits a measurement

        * key: 'process_noise'
        * value: numpy array (MxM), process noise for motion of targets

        * key: 'meas_noise_target_state'
        * value: numpy array (NxN), measurement noise for valid target measurements

        * key: 'avg_bb_birth'
        * value: numpy array (2x1)CHECK vs. TRANSPOSE, mean bounding box dimensions
            for ground truth objects (GET BIRTH INSTEAD?)

        * key: 'var_bb_birth'
        *value: numpy array (2x2), covariance matrix for valid object bounding box 
            dimensions            

        * key: 'avg_bb_clutter'
        * value: numpy array (2x1)CHECK vs. TRANSPOSE, mean bounding box dimensions
            for clutter objects 

        * key: 'var_bb_clutter'
        *value: numpy array (2x2), covariance matrix for clutter object bounding box 
            dimensions    

        * key: 'BORDER_DEATH_PROBABILITIES': BORDER_DEATH_PROBABILITIES
        * value: list of floats, ith entry is probability of target death
            after being unassociatied with a measurement for i time steps when the
            target is near the image border

        * key: 'NOT_BORDER_DEATH_PROBABILITIES': NOT_BORDER_DEATH_PROBABILITIES
        * value: list of floats, ith entry is probability of target death
            after being unassociatied with a measurement for i time steps when the
            target is not near the image border

        * key: 'init_vel_cov'
        * value: numpy array, covariance for sampling initial target velocities
            from Gaussian with mean 0
        Outputs:
        - measurementSet: type TargetSet, contains generated measurements in
            measurementSet.measurements
        - groundTruthSet: type TargetSet, contains ground truth bounding boxes in
            groundTruthSet.measurements      
    '''
    params['init_vel_cov'] = np.asarray(params['init_vel_cov'])

    measurementSet = TargetSet() #contains generated measurements
    groundTruthSet = TargetSet() #contains ground truth target locations
    next_t_id = 0
    for time_step in range(params['num_time_steps']):
        cur_time = time_step * params['time_per_time_step']
        target_offscreen = False #whether any targets move offscreen
        measurementSet.measurements.append(Measurement(time = cur_time))
        groundTruthSet.measurements.append(Measurement(time = cur_time))

        for target in measurementSet.living_targets:
            #move targets
            target.move(params['time_per_time_step'], params['process_noise']) #IMPLEMENT ME!!! and kill if target goes offscreen
            if target.offscreen:
                target_offscreen = True
        if target_offscreen: #kill offscreen targets (remove from living_targets list)
            measurementSet.living_targets = [t for t in measurementSet.living_targets if not t.offscreen]
        target_died = False
        for target in measurementSet.living_targets:      
            #add all target locations to ground truth
            target_position = np.dot(H, target.x).reshape(-1)
            groundTruthSet.measurements[-1].val.append(target_position)
            groundTruthSet.measurements[-1].widths.append(target.width)
            groundTruthSet.measurements[-1].heights.append(target.height)
            groundTruthSet.measurements[-1].ids.append(target.id_)            

            #sample wether each target produces a measurement
            if np.random.random() < params['p_emission']:
                #sample measurement with noise if this target emits a measurement
                target_measurement = target.sample_measurement(params['meas_noise_target_state'], cur_time)
                measurementSet.measurements[-1].val.append(target_measurement)
                measurementSet.measurements[-1].widths.append(target.width)
                measurementSet.measurements[-1].heights.append(target.height)
            else: #sample target death for targets that do not produce measurements
                death_prob = target.target_death_prob(cur_time, cur_time - params['time_per_time_step'], params)
                if np.random.random() < death_prob: #kill the target
                    target.alive = False
                    target_died = True
        if target_died: #remove dead targets from living_targets list
            measurementSet.living_targets = [t for t in measurementSet.living_targets if t.alive]

        #sample the number of births
        if time_step == 0:
            birth_count = params['init_target_count']
        else:
            birth_count = np.random.poisson(params['lamda_b'])
        #sample the number of clutter objects
        clutter_count = np.random.poisson(params['lamda_c'])
        #create birth and clutter measurements
        for b in range(birth_count):
            bb_size = np.random.multivariate_normal(params['avg_bb_birth'], params['var_bb_birth'])
            x_pos = np.random.uniform(X_MIN, X_MAX)
            y_pos = np.random.uniform(Y_MIN, Y_MAX)
            new_target = TargetState(cur_time, next_t_id, BoundingBox(x_pos, y_pos, abs(bb_size[0]), abs(bb_size[1]), cur_time))
            next_t_id += 1
            target_velocity = np.random.multivariate_normal([0,0], params['init_vel_cov'])
            if params['init_vel_to_center']: #point velocity towards image center
                target_velocity[0] = abs(target_velocity[0])*np.sign((X_MAX + X_MIN)/2.0 - x_pos)
                target_velocity[1] = abs(target_velocity[1])*np.sign((Y_MAX + Y_MIN)/2.0 - y_pos)
            new_target.x[(1,0)] = target_velocity[0]
            new_target.x[(3,0)] = target_velocity[1]
            measurementSet.living_targets.append(new_target)
            #add to ground truth
            target_position = np.dot(H, new_target.x).reshape(-1)
            groundTruthSet.measurements[-1].val.append(target_position)
            groundTruthSet.measurements[-1].widths.append(new_target.width)
            groundTruthSet.measurements[-1].heights.append(new_target.height)            
            groundTruthSet.measurements[-1].ids.append(new_target.id_)            
            #sample measurement with noise 
            target_measurement = new_target.sample_measurement(params['meas_noise_target_state'], cur_time)
            measurementSet.measurements[-1].val.append(target_measurement)
            measurementSet.measurements[-1].widths.append(new_target.width)
            measurementSet.measurements[-1].heights.append(new_target.height)            

        for c in range(clutter_count):
            bb_size = np.random.multivariate_normal(params['avg_bb_clutter'], params['var_bb_clutter'])
            x_pos = np.random.uniform(X_MIN, X_MAX)
            y_pos = np.random.uniform(Y_MIN, Y_MAX)
            measurementSet.measurements[-1].val.append(np.array([x_pos, y_pos]))
            measurementSet.measurements[-1].widths.append(bb_size[0])
            measurementSet.measurements[-1].heights.append(bb_size[1])            
 
        #randomize the order of measurements
        if len(measurementSet.measurements[-1].val)>1: #if we have multiple measurements on this time step
            combined_list = list(zip(measurementSet.measurements[-1].val, measurementSet.measurements[-1].widths,
                measurementSet.measurements[-1].heights))
            random.shuffle(combined_list)
            (measurementSet.measurements[-1].val, measurementSet.measurements[-1].widths,
                measurementSet.measurements[-1].heights) = zip(*combined_list)
    return (measurementSet, groundTruthSet)

def KITTI_detection_file_to_TargetSet(filename, time_per_time_step):
    '''
    Inputs:
    - filename: (string) the location of the object detections file in KITTI format
    - time_per_time_step: (float) how much time elapses between time steps (or frames)

    Outputs:
    - measurementSet: (TargetSet) containing the measurements from filename
    '''
    prev_frame_idx = -99
    measurementSet = TargetSet()
    with open(filename, "r") as fh:
        for line in fh:
            # KITTI tracking benchmark data format:
            # (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
            line = line.strip()
            fields       = line.split(" ")
            frame_idx    = int(round(float(fields[0])))    # frame
            x1           = float(fields[6])          # left   [px]
            y1           = float(fields[7])          # top    [px]
            x2           = float(fields[8])          # right  [px]
            y2           = float(fields[9])          # bottom [px]
            bb_center = np.array([(x2+x1)/2.0, (y2+y1)/2.0])
            width = x2-x1
            height = y2-y1
            time_stamp = frame_idx*time_per_time_step
            if frame_idx != prev_frame_idx:
                measurementSet.measurements.append(Measurement(time = time_stamp))
                prev_frame_idx = frame_idx

            measurementSet.measurements[-1].val.append(bb_center)
            measurementSet.measurements[-1].widths.append(width)
            measurementSet.measurements[-1].heights.append(height)                  
    fh.close()
    return measurementSet

@explicit_serialize
class GenData(FireTaskBase):   
 #   _fw_name = "Run RBPF Task"
    def run_task(self, fw_spec):
        #calculate data generation parameters on KITTI training data with measurements of type
        #fw_spec['det_names']
        if fw_spec['use_KITTI_data_gen_params'] == True:
            include_ignored_gt=False
            include_dontcare_in_gt=False
            include_ignored_detections = True 

            (measurementTargetSetsBySequence, target_groupEmission_priors, clutter_grpCountByFrame_priors, clutter_group_priors, 
                birth_count_priors, BORDER_DEATH_PROBABILITIES, NOT_BORDER_DEATH_PROBABILITIES, 
                posAndSize_inv_covariance_blocks, meas_noise_mean, posOnly_covariance_blocks,
                clutter_posAndSize_inv_covariance_blocks, clutter_posOnly_covariance_blocks, clutter_meas_noise_mean_posAndSize,
                gt_bounding_box_mean_size, clutter_bounding_box_mean_size, gt_bb_size_var, clutter_bb_size_var) =\
                get_meas_target_sets_general(fw_spec['training_sequences'], fw_spec['SCORE_INTERVALS'], fw_spec['det_names'], \
                obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True,\
                include_ignored_gt = include_ignored_gt, include_dontcare_in_gt = include_dontcare_in_gt, \
                include_ignored_detections = include_ignored_detections, return_bb_size_info = True)

            print 'emission priors:', target_groupEmission_priors
            print 1 - target_groupEmission_priors[ImmutableSet([])]
            print 'clutter_grpCountByFrame_priors:', clutter_grpCountByFrame_priors
            print 'clutter_group_priors:', clutter_group_priors
            print 'birth_count_priors:', birth_count_priors
            print 'BORDER_DEATH_PROBABILITIES:', BORDER_DEATH_PROBABILITIES
            print 'NOT_BORDER_DEATH_PROBABILITIES:', NOT_BORDER_DEATH_PROBABILITIES

            for key, val in posAndSize_inv_covariance_blocks.iteritems():
                print 'posAndSize_inv_covariance_blocks,', key
                print val
                print inv(val)
            print 'meas_noise_mean:', meas_noise_mean
            print "!"*80
            print 'posOnly_covariance_blocks:', posOnly_covariance_blocks
            print posOnly_covariance_blocks[(fw_spec['det_names'][0], fw_spec['det_names'][0])]
            print "!"*80
            print 'clutter_posAndSize_inv_covariance_blocks:', clutter_posAndSize_inv_covariance_blocks

            for key, val in clutter_posAndSize_inv_covariance_blocks.iteritems():
                print 'clutter_posAndSize_inv_covariance_blocks,', key
                print val
                print inv(val)
            print 'clutter_posOnly_covariance_blocks:', clutter_posOnly_covariance_blocks 
            print 'clutter_meas_noise_mean_posAndSize:', clutter_meas_noise_mean_posAndSize

            print 'gt_bounding_box_mean_size:', gt_bounding_box_mean_size
            print 'clutter_bounding_box_mean_size:', clutter_bounding_box_mean_size
            print type(clutter_bounding_box_mean_size)
            print 'gt_bb_size_var:', gt_bb_size_var
            print 'clutter_bb_size_var:', clutter_bb_size_var

            print "#"*40, "Investigate Poisson Distribution", "#"*40
            #Calculate maximum likelihood estimates of Poisson parameter for clutter and birth counts
            lamda_C = 0 #The expected number of clutter objects in a frame, also MLE of lambda for Poisson distribution
            for clutter_count, probability in clutter_grpCountByFrame_priors.iteritems():
                lamda_C += clutter_count*probability
            lamda_B = 0 #The expected number of birth objects in a frame, also MLE of lambda for Poisson distribution
            for birth_count, probability in birth_count_priors.iteritems():
                lamda_B += birth_count*probability

            print "MLE of clutter lambda (expected clutter count) = ", lamda_C
            for clutter_count, probability in clutter_grpCountByFrame_priors.iteritems():
                poison_prob = lamda_C**clutter_count*math.exp(-lamda_C)/math.factorial(clutter_count)
                print 'clutter count =', clutter_count, ', probability in data =', probability, ', Poisson probability =', \
                poison_prob, ', percent error =', (poison_prob - probability)/probability

            print "MLE of birth lambda (expected birth count) = ", lamda_B
            for birth_count, probability in birth_count_priors.iteritems():
                poison_prob = lamda_B**birth_count*math.exp(-lamda_B)/math.factorial(birth_count)
                print 'birth count =', birth_count, ', probability in data =', probability, ', Poisson probability =', \
                poison_prob, ', percent error =', (poison_prob - probability)/probability

            data_gen_params = {'num_time_steps': fw_spec['num_time_steps'],
                'time_per_time_step': fw_spec['time_per_time_step'],        
                'lamda_c': lamda_C,
                'lamda_b': lamda_B,
                'p_emission': 1 - target_groupEmission_priors[ImmutableSet([])],
                'process_noise': Q_DEFAULT,
                'meas_noise_target_state': posOnly_covariance_blocks[(fw_spec['det_names'][0], fw_spec['det_names'][0])],
                'avg_bb_birth': gt_bounding_box_mean_size,
                'var_bb_birth': gt_bb_size_var,
                'avg_bb_clutter': clutter_bounding_box_mean_size,
                'var_bb_clutter': clutter_bb_size_var,
                'BORDER_DEATH_PROBABILITIES': BORDER_DEATH_PROBABILITIES,
                'NOT_BORDER_DEATH_PROBABILITIES': NOT_BORDER_DEATH_PROBABILITIES,
                'init_vel_cov': fw_spec['init_vel_cov'],
                'init_target_count' : fw_spec['init_target_count'],
                'init_vel_to_center' : fw_spec['init_vel_to_center']}

        else: #use the provided data generation parameters
            data_gen_params = fw_spec['data_gen_params']
            data_gen_params['num_time_steps'] = fw_spec['num_time_steps']
            data_gen_params['time_per_time_step'] = fw_spec['time_per_time_step']
            data_gen_params['init_vel_cov'] = fw_spec['init_vel_cov']
            data_gen_params['init_target_count'] = fw_spec['init_target_count']
            data_gen_params['init_vel_to_center'] = fw_spec['init_vel_to_center']


        for gen_seq_idx in range(fw_spec['num_seq_to_generate']):
            (measurementSet, groundTruthSet) = gen_data_1source(data_gen_params)
            measurementSet.write_measurements_to_KITTI_format("%smeasurements/%04d.txt" % (fw_spec['data_file_path'], gen_seq_idx), fw_spec)
            groundTruthSet.write_measurements_to_KITTI_format("%sground_truth/%04d.txt" % (fw_spec['data_file_path'], gen_seq_idx), fw_spec, gt = True)

if __name__ == "__main__":
    spec = {#if True calculate data generation parameters on KITTI training data with measurements of type
            #fw_spec['det_names'], otherwise use data generation parameters we provide in 'data_gen_params'
            'use_KITTI_data_gen_params' : True,
            'det_names' : ['regionlets'], #used if 'use_KITTI_data_gen_params' is True
            'SCORE_INTERVALS' : {'regionlets': [2]}, #used if 'use_KITTI_data_gen_params' is True
            'training_sequences' : [i for i in range(21)], #used if 'use_KITTI_data_gen_params' is True
            #supply data generation parameters if 'use_KITTI_data_gen_params' is False
            'data_gen_params' : None,
            #always supply these parameters, whether 'use_KITTI_data_gen_params' is True or False
            'data_file_path': ("%sgenerated_data_KITTI_regionlets_params/" % RBPF_HOME_DIRECTORY),
            'num_seq_to_generate': 20, #how many sequences with these params do we generate?
            'num_time_steps': 100, #time steps per sequence
            'time_per_time_step': DEFAULT_TIME_STEP,
            'init_vel_cov': np.array([[10, 0],
                                      [0, 10]])}

    # write new launchpad file, not positive if this is necessary
    create_launchpad()

    # set up the LaunchPad and reset it
    launchpad = LaunchPad(host=MONGODB_HOST, port=MONGODB_PORT, name=MONGODB_NAME, username=MONGODB_USERNAME, password=MONGODB_PASSWORD,
                     logdir=None, strm_lvl='INFO', user_indices=None, wf_user_indices=None, ssl_ca_file=None)
    launchpad.reset('', require_password=False)


    # store workflow and launch it locally
    firework = Firework(GenData(), spec=spec)
    workflow = Workflow([firework])
    launchpad.add_wf(workflow)
    rapidfire(launchpad, FWorker())




