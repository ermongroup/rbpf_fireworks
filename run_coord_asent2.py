#!/usr/bin/env python
# -*- coding: utf-8 -*- 

#Note, on Atlas before this script:
# $ PACKAGE_DIR=/atlas/u/jkuck/software
# $ export PATH=$PACKAGE_DIR/anaconda2/bin:$PATH
# $ export LD_LIBRARY_PATH=$PACKAGE_DIR/anaconda2/local:$LD_LIBRARY_PATH
# $ source activate anaconda_venv
# $ cd /atlas/u/jkuck/rbpf_fireworks/
#
#May need to run $ kinit -r 30d
#
# Add the following line to the file ~/.bashrc.user on Atlas:
# export PYTHONPATH="/atlas/u/jkuck/rbpf_fireworks:$PYTHONPATH"
# Weird, but to run commands like "lpad -l my_launchpad.yaml get_fws",
# add the following line to the file ~/.bashrc.user on Atlas:
# export PYTHONPATH="${PYTHONPATH}:/atlas/u/jkuck/rbpf_fireworks/KITTI_helpers/"
#
##########################################################################################
#
#Note, on Sherlock before this script:
# $ ml load python/2.7.5
# $ easy_install-2.7 --user pip
# $ export PATH=~/.local/bin:$PATH
# $ pip2.7 install --user fireworks #and others
# $ cd /scratch/users/kuck/rbpf_fireworks/
#
# Add the following line to the file ~/.bashrc on Sherlock:
# export PYTHONPATH="/scratch/users/kuck/rbpf_fireworks:$PYTHONPATH"
# Weird, but to run commands like "lpad -l my_launchpad.yaml get_fws",
# add the following line to the file ~/.bashrc.user on Atlas:
# export PYTHONPATH="${PYTHONPATH}:/scratch/users/kuck/rbpf_fireworks/KITTI_helpers/"
from __future__ import division
import os
import errno
import sys
import numpy as np
import copy
import math

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FWAction, FireTaskBase
from fireworks import Firework, Workflow, FWorker, LaunchPad
#from fireworks.core.rocket_launcher import rapidfire
from fireworks.queue.queue_launcher import rapidfire
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from fw_tutorials.dynamic_wf.fibadd_task import FibonacciAdderTask
#from fireworks.core.firework import FWAction, Firework, FiretaskBase
#from fireworks.user_objects.firetasks.script_task import PyTask

from cluster_config import RBPF_HOME_DIRECTORY, MONGODB_USERNAME, MONGODB_PASSWORD
from experiment_config import MONGODB_HOST, MONGODB_PORT, MONGODB_NAME
from create_launchpad import create_launchpad

from rbpf import RunRBPF
sys.path.insert(0, "%sKITTI_helpers" % RBPF_HOME_DIRECTORY)
from jdk_helper_evaluate_results import RunEval

#from intermediate import RunRBPF
###################################### Experiment Parameters ######################################
NUM_RUNS=1
#TRAINING_SEQUENCES = [i for i in range(21)]
TRAINING_SEQUENCES = [i for i in range(9)]
VALIDATION_SEQUENCES = [i for i in range(9,21)]
#TRAINING_SEQUENCES = [0]
#TRAINING_SEQUENCES = [11]
#TRAINING_SEQUENCES = [12,13,17]
#TRAINING_SEQUENCES = [13]
#VALIDATION_SEQUENCES = [13]
NUM_PARTICLES = 100


###################################### Experiment Organization ######################################
#DIRECTORY_OF_ALL_RESULTS = './ICML_prep_correctedOnline/propose_k=1_nearest_targets'
DIRECTORY_OF_ALL_RESULTS = '%scoord_ascent2' % RBPF_HOME_DIRECTORY
#CUR_EXPERIMENT_BATCH_NAME = 'genData_origRBPF_multMeas_probDet95_clutLambdaPoint1_noise05_noShuffle_beta1'
#CUR_EXPERIMENT_BATCH_NAME = 'genData_origRBPF_multMeas_fixedRounding_resampleRatio4_scaled_ShuffleMeas_timeScaled_PQdiv100'
#CUR_EXPERIMENT_BATCH_NAME = 'Rto0_4xQ_multMeas1update_online3frameDelay2'
CUR_EXPERIMENT_BATCH_NAME = 'CHECK_1_NEAREST_TARGETS/Rto0_4xQ_max1MeasUpdate_online3frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = 'CHECK_K_NEAREST_TARGETS=False/Reference/Rto0_4xQ_max1MeasUpdate_online3frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = '/Reference/Rto0_4xQ_max1MeasUpdate_online0frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = 'measuredR_1xQ_max1MeasUpdate_online3frameDelay'

###################################### Coordinate Descent Parameters ######################################
#Choose whether to use the new or old eval script
EVAL_SCRIPT = 'NEW'
#Objective to maximize (need to adjust code to minimize 'Mostly Lost')
OBJECTIVE_METRIC = 'MOTA'
#The percent by which to initially increase and decrease every parameter during coordinate descent
alpha_init = 50.0
#If changing a parameter improves the objective, increase the parameter's alpha value by a factor of alpha_inc
alpha_inc = 1.3
#If changing a parameter does not improve the objective, decrease the parameter's alpha value by multiplying it
#by alpha_dec
alpha_dec = .5
Q_ALPHA_INIT = np.array([[ alpha_init,  alpha_init,  alpha_init,   alpha_init],
                           [ alpha_init,  alpha_init,  alpha_init,   alpha_init],
                           [ alpha_init,  alpha_init,  alpha_init,   alpha_init],
                           [ alpha_init,  alpha_init,  alpha_init,   alpha_init]])
R_ALPHA_INIT = np.array([[ alpha_init,   alpha_init],
                         [ alpha_init,   alpha_init]])
ALPHA_INIT = [alpha_init for i in range(8)]
#standard deviations are required to be larger than EPSILON and correlations in the
#range [-1+EPSILON, 1-EPSILON]
EPSILON = .05

###################################### RBPF Parameters ######################################
#Specify how the proposal distribution should be pruned
CHECK_K_NEAREST_TARGETS = True
K_NEAREST_TARGETS = 1

#If False, ONLINE_DELAY is maximized (we wait until the video ends before picking a particle)
RUN_ONLINE = True #save near online results 
#near online mode, wait this many frames before picking max weight particle 
ONLINE_DELAY = 3

MAX_1_MEAS_UPDATE = True
#if true, view measurements as jointly gaussian and update
#target once per time stamp with combination of associated measurements
UPDATE_MULT_MEAS_SIMUL = True
if(MAX_1_MEAS_UPDATE):
    UPDATE_MULT_MEAS_SIMUL = False
#for debugging, zero out covariance between measurement sources when
#UPDATE_MULT_MEAS_SIMUL=True, should be the same result as sequential updates
TREAT_MEAS_INDEP = False
#for debugging, actually do 2 sequential updates, but after all associations
TREAT_MEAS_INDEP_2 = False

USE_CONSTANT_R = True

###################################### Kalman Filter Parameters ######################################
USE_LEARNED_KF_PARAMS = True

TIME_SCALED = False #For testing with generated data
SCALED = False #For testing with generated data
#from rbpf_ORIGINAL_sampling import SCALED #For testing with generated data

#Dimensions of the Q matrix
Q_DIM = 4
#Dimensions of the R matrix
R_DIM = 2

if USE_LEARNED_KF_PARAMS:
    P_default = np.array([[40.64558317, 0,           0, 0],
                          [0,          10,           0, 0],
                          [0,           0, 5.56278505, 0],
                          [0,           0,           0, 3]])
#    R_default = np.array([[ 0.0,   0.0],
#                          [ 0.0,   0.0]])
    R_default = np.array([[ 10.0,   1.0],
                          [ 1.0,   10.0]])    
    
    
    #learned from all GT
    Q_default = np.array([[  60.33442497,  102.95992102,   -5.50458177,   -0.22813535],
                          [ 102.95992102,  179.84877761,  -13.37640528,   -9.70601621],
                          [  -5.50458177,  -13.37640528,    4.56034398,    9.48945108],
                          [  -0.22813535,   -9.70601621,    9.48945108,   22.32984314]])
    
    Q_default = 4*Q_default
#####################replicate ORIG
else:
    if SCALED:
        P_default = np.array([[(NOISE_SD*300)**2,      0,           0,  0],
                              [0,          10*300**2,           0,  0],
                              [0,           0,      (NOISE_SD*90)**2,  0],
                              [0,           0,           0, 10*90**2]])

        R_default = np.array([[ (NOISE_SD*300)**2,             0.0],
                              [          0.0,   (NOISE_SD*90)**2]])
        Q_default = np.array([[     (300**2)*0.00003333,    (300**2)*0.0050,         0,         0],
                              [         (300**2)*0.0050,       (300**2)*1.0,         0,         0],
                              [              0,         0,(90**2)*0.00003333,    (90**2)*0.0050],
                              [              0,         0,    (90**2)*0.0050,    (90**2)*1.0000]])
        Q_default = Q_default*10**(-3)
        if TIME_SCALED:
            Q_default = np.array([[     (300**2)*0.00003333,    (300**2)*0.0005,         0,         0],
                                  [         (300**2)*0.0005,       (300**2)*.01,         0,         0],
                                  [              0,         0,(90**2)*0.00003333,    (90**2)*0.0005],
                                  [              0,         0,    (90**2)*0.0005,    (90**2)*.01]])    


    else:
        P_default = np.array([[(NOISE_SD)**2,    0,           0,  0],
                              [0,          10,           0,  0],
                              [0,           0,   (NOISE_SD)**2,  0],
                              [0,           0,           0, 10]])

        R_default = np.array([[ (NOISE_SD)**2,             0.0],
                              [      0.0,   (NOISE_SD)**2]])
        Q_default = np.array([[     0.00003333,    0.0050,         0,         0],
                              [         0.0050,       1.0,         0,         0],
                              [              0,         0,0.00003333,    0.0050],
                              [              0,         0,    0.0050,    1.0000]])
        Q_default = Q_default*10**(-3)

        if TIME_SCALED:
            Q_default = np.array([[     0.00003333,    0.0005,         0,         0],
                                  [         0.0005,       .01,         0,         0],
                                  [              0,         0,0.00003333,    0.0005],
                                  [              0,         0,    0.0005,    .01]])  
#####################end replicate ORIG



def get_description_of_run(include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals,
                           det1_name, det2_name):

    if det2_name == 'None' or det2_name == None:
        if (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and sort_dets_on_intervals:
            description_of_run = "%s_with_score_intervals" % (det1_name)
        elif (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and (not sort_dets_on_intervals):
            description_of_run = "%s_no_score_intervals" % (det1_name)
        else:
            print "Unexpected combination of boolean arguments"
            print include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals, use_mscnn
            sys.exit(1);

    else:

        if (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and sort_dets_on_intervals:
            description_of_run = "%s_%s_with_score_intervals" % (det1_name, det2_name)
        elif (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and (not sort_dets_on_intervals):
            description_of_run = "%s_%s_no_score_intervals" % (det1_name, det2_name)
        else:
            print "Unexpected combination of boolean arguments"
            print include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals, use_mscnn
            sys.exit(1);

    return description_of_run


def setup_results_folder(results_folder):
    for cur_run_idx in range(1, NUM_RUNS + 1):
        file_name = '%s/results_by_run/run_%d/%s.txt' % (results_folder, cur_run_idx, 'random_name')
        if not os.path.exists(os.path.dirname(file_name)):
            try:
                os.makedirs(os.path.dirname(file_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise


@explicit_serialize
class RunRBPF_Batch(FireTaskBase):   
 #   _fw_name = "Run RBPF Task"
    #modify the element of Q specified by fw_spec['Q_idx'] (e.g. Q[Q_idx//4][Q_idx%4]) by fw_spec['mod_percent'] 
    #percent in the fw_spec['mod_direction'] direction ('inc', 'dec', 'const') and
    #run a batch of RBPF jobs on fw_spec['TRAINING_SEQUENCES'] with fw_spec['NUM_RUNS'] in parallel
    def run_task(self, fw_spec):
        rbpf_batch = []
        if fw_spec['coord_ascent_iter'] > 0:
            assert('mod_direction' in fw_spec)
            fw_spec['results_folder'] = "%s/iterID_%d_dir-%s"%(fw_spec['results_folder'], fw_spec['coord_ascent_iter'],
                                                               fw_spec['mod_direction'])
        else:
            fw_spec['results_folder'] = "%s/iterID_%d"%(fw_spec['results_folder'], fw_spec['coord_ascent_iter'])

        setup_results_folder(fw_spec['results_folder'])
        rbpf_batch = []
        for run_idx in range(1, fw_spec['NUM_RUNS']+1):
            for seq_idx in fw_spec['TRAINING_SEQUENCES']:
                cur_spec = copy.deepcopy(fw_spec)
                cur_spec['run_idx'] = run_idx
                cur_spec['seq_idx'] = seq_idx
#                Q_idx = fw_spec['Q_idx']
#                if fw_spec['mod_direction'] == 'inc':
#                    cur_spec['Q'][Q_idx//4][Q_idx%4] += cur_spec['Q'][Q_idx//4][Q_idx%4]*fw_spec['mod_percent']/100.0
#                elif fw_spec['mod_direction'] == 'dec':
#                    cur_spec['Q'][Q_idx//4][Q_idx%4] -= cur_spec['Q'][Q_idx//4][Q_idx%4]*fw_spec['mod_percent']/100.0
#                else:
#                    assert(fw_spec['mod_direction'] == 'const')
                cur_firework = Firework(RunRBPF(), spec=cur_spec)
                rbpf_batch.append(cur_firework)

        parallel_workflow = Workflow(rbpf_batch)
        return FWAction(detours=parallel_workflow, mod_spec=[{'_set': {"results_folder": fw_spec['results_folder']}}])
#        return FWAction(detours=parallel_workflow)


def cov2corr(cov, return_std=False):
    '''convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.

    Author: Josef Perktold
    License: BSD-3
    '''
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr

def corr2cov(corr, std):
    '''convert correlation matrix to covariance matrix given standard deviation

    Parameters
    ----------
    corr : array_like, 2d
        correlation matrix, see Notes
    std : array_like, 1d
        standard deviation

    Returns
    -------
    cov : ndarray (subclass)
        covariance matrix

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that multiplication is defined elementwise. np.ma.array are allowed, but
    not matrices.

    Author: Josef Perktold
    License: BSD-3
    '''
    corr = np.asanyarray(corr)
    std_ = np.asanyarray(std)
    cov = corr * np.outer(std_, std_)
    return cov


def modify_parameter(spec, direction):
    """
    Inputs:
    - spec: the fireworks spec to modify
    - direction: string, 'inc' or 'dec', whether to increase of decrease the parameter

    Outputs:
    - new_param_value: float, the new value of the parameter we modified
    - Also, we will have modified the spec which was passed by reference (since it
    is a dictionary).  We make the following changes to the spec:
        -spec['mod_direction'] = direction
        -modify the specified parameter as directed
    """
    if not (direction in ['inc', 'dec']):
        raise ValueError('Unknown direction given to modify_parameter: %s' % direction)

    spec['mod_direction'] = direction

    (param_name, det_name, idx) = get_param(spec['param_idx'], spec)
    mod_percent = spec["alpha"][spec['param_idx']]/100.0

    if param_name == 'det_grouping_min_overlap':
        if direction == 'inc':
            spec['det_grouping_min_overlap'][det_name] += spec['det_grouping_min_overlap'][det_name]*mod_percent
        if direction == 'dec':
            spec['det_grouping_min_overlap'][det_name] -= spec['det_grouping_min_overlap'][det_name]*mod_percent
    else:
        assert(param_name == "target_detection_max_dists")
        if direction == 'inc':
            spec['target_detection_max_dists'][idx] += spec['target_detection_max_dists'][idx]*mod_percent
        if direction == 'dec':
            spec['target_detection_max_dists'][idx] -= spec['target_detection_max_dists'][idx]*mod_percent
                
    new_param_value = spec[param_name]
    return new_param_value



#    #construct correlation matrix
#    cor = np.copy(spec[param_name])
#    (num_rows, num_cols) = cor.shape
#    for r in range(num_rows):
#        for c in range(num_cols):
#            cor[r][c] = orig_matrix[r][c]/(math.sqrt(orig_matrix[r][r])*math.sqrt(orig_matrix[c][c]))
#
#    for r in range(num_rows):
#        for c in range(num_cols):
#            assert(abs(cor[r][c] - cor[c][r]) < .0001)
#    if direction == 'inc':
#        cor[row_idx][col_idx] += cor[row_idx][col_idx]*mod_percent
#    elif direction == 'dec':
#        cor[row_idx][col_idx] -= cor[row_idx][col_idx]*mod_percent
#    #make the matrix symmetric
#    cor[col_idx][row_idx] = cor[row_idx][col_idx]
#    #make sure diagonal elements to not go below 0
#
#    if col_idx == row_idx and cor[row_idx][col_idx] < EPSILON:
#        cor[row_idx][col_idx] = EPSILON
#        print "set col/row", col_idx, "to epsilon, proof:"
#        print cor
#    print "spec[param_name]:"
#    print spec[param_name]
#    print cor
#
#    for r in range(num_rows):
#        for c in range(num_cols):
#            assert(abs(cor[r][c] - cor[c][r]) < .0001)
#
#    #reconstruct covariance matrix
#    for r in range(num_rows):
#        for c in range(num_cols):
#                spec[param_name][r][c] = cor[r][c]*\
#                                                    math.sqrt(orig_matrix[r][r])*\
#                                                    math.sqrt(orig_matrix[c][c])
#
#    for r in range(num_rows):
#        for c in range(num_cols):
#            assert(abs(spec[param_name][r][c] - spec[param_name][c][r]) < .0001)
#            if (r==c):
#                assert(spec[param_name][r][r] > 0)
#                
#    new_param_value = spec[param_name][row_idx][col_idx]
#    return new_param_value

#    cov = np.copy(spec[param_name])
#    (num_rows, num_cols) = cov.shape
#    if row_idx == col_idx:
#        for r in range(num_rows):
#            for c in range(num_cols):
#                if r!=c:
#                    cov[r][c] = cov[r][c]/(math.sqrt(cov[r][r])*math.sqrt(cov[c][c]))
#
#    if direction == 'inc':
#        spec[param_name][row_idx][col_idx] += spec[param_name][row_idx][col_idx]*mod_percent
#
#    elif direction == 'dec':
#        spec[param_name][row_idx][col_idx] -= spec[param_name][row_idx][col_idx]*mod_percent
#
#
#    #make the matrix symmetric
#    spec[param_name][col_idx][row_idx] = spec[param_name][row_idx][col_idx]
#    #make sure diagonal elements to not go below 0
#    print "col_idx == row_idx:", col_idx == row_idx
#    print "spec[param_name][row_idx][col_idx] < EPSILON:", spec[param_name][row_idx][col_idx] < EPSILON
#    print "spec[param_name]:"
#    print spec[param_name]
#    if col_idx == row_idx and spec[param_name][row_idx][col_idx] < EPSILON:
#        spec[param_name][row_idx][col_idx] = EPSILON
#        print "set col/row", col_idx, "to zero, proof:"
#        print spec[param_name]
#    print "spec[param_name]:"
#    print spec[param_name]
#    print cov
#
#    #make sure covariances are still valid if changing a diagonal
#    if row_idx == col_idx:
#        for r in range(num_rows):
#            for c in range(num_cols):
#                if r==row_idx or c==col_idx:
#                    assert (spec[param_name][r][r] > 0 and
#                            spec[param_name][c][c] > 0), (spec[param_name], col_idx, row_idx, direction, col_idx == row_idx, spec[param_name][row_idx][col_idx] < EPSILON)
#                    spec[param_name][r][c] = cov[r][c]*\
#                                                        math.sqrt(spec[param_name][r][r])*\
#                                                        math.sqrt(spec[param_name][c][c])
#
#    new_param_value = spec[param_name][row_idx][col_idx]
#    return new_param_value

@explicit_serialize
class Iterate(FireTaskBase):   
    #Run one iteration of coordinate descent.
    #1. Increase a parameter and evaluate on a performance metric
    #2. Decrease a parameter and evaluate on a performance metric
    #3. Compare performance between original parameter, increase and decrease. Iterate
    #coordinate descent on the next parameter.

    def run_task(self, fw_spec):
        (param_name, det_name, idx) = get_param(fw_spec['param_idx'], fw_spec)
        fw_spec['orig_param_val'] = fw_spec[param_name]
        fw_spec['coord_ascent_iter'] += 1

        #run an RBPF batch with the original parameter (avoid randomly getting a
        #great result at some point in the past)
        orig_spec = copy.deepcopy(fw_spec)
        orig_spec['mod_direction'] = 'orig'
        orig_batch_firework = Firework(RunRBPF_Batch(), spec = orig_spec)
        #evaluate the RBPF batch with the original parameter 
        eval_orig_spec = copy.deepcopy(orig_spec)
        eval_orig_spec['seq_idx_to_eval'] = eval_orig_spec['TRAINING_SEQUENCES']
        orig_eval = Firework(RunEval(), spec = eval_orig_spec)

        #run an RBPF batch with the parameter increased
        inc_spec = copy.deepcopy(fw_spec)
        inc_param_value = modify_parameter(inc_spec, 'inc')
        inc_spec['inc_param_val'] = inc_param_value
        fw_spec['inc_param_val'] = inc_param_value
        inc_batch_firework = Firework(RunRBPF_Batch(), spec = inc_spec)
        #evaluate the RBPF batch with the parameter increased
        eval_inc_spec = copy.deepcopy(inc_spec)
        eval_inc_spec['seq_idx_to_eval'] = eval_inc_spec['TRAINING_SEQUENCES']
        inc_eval = Firework(RunEval(), spec = eval_inc_spec)

        #run an RBPF batch with the parameter decreased
        dec_spec = copy.deepcopy(fw_spec)
        dec_param_value = modify_parameter(dec_spec, 'dec')
        dec_spec['dec_param_val'] = dec_param_value
        fw_spec['dec_param_val'] = dec_param_value
        dec_batch_firework = Firework(RunRBPF_Batch(), spec = dec_spec)
        #evaluate the RBPF batch with the parameter decreased
        eval_dec_spec = copy.deepcopy(dec_spec)
        eval_dec_spec['seq_idx_to_eval'] = eval_dec_spec['TRAINING_SEQUENCES']        
        dec_eval = Firework(RunEval(), spec = eval_dec_spec)

        #run a firework that compares the evaluation metric when the parameter is increased, decreased, 
        #and its original value, and runs the next iteration of coordinate descent
        next_iter = Firework(ChooseNextIter(), spec = fw_spec)

        #Chain the fireworks together with proper dependencies and set 'em off!
        iteration_workflow = Workflow([inc_batch_firework, dec_batch_firework, orig_batch_firework, \
                                       inc_eval, dec_eval, orig_eval, next_iter], 
                            {inc_batch_firework: [inc_eval], dec_batch_firework: [dec_eval], \
                             orig_batch_firework: [orig_eval], inc_eval:[next_iter], dec_eval:[next_iter], \
                             orig_eval:[next_iter]})

        return FWAction(additions = iteration_workflow)

@explicit_serialize
class ChooseNextIter(FireTaskBase):   
    #Run one iteration of coordinate descent.
    #Compare performance between original parameter, increase and decrease. Update
    #the parameter with the maximum value and either increase the parameter's alpha if
    #the parameter was changed or decrease the parameter's alpha if it wasn't.  Iterate
    #coordinate descent on the next parameter.

    def run_task(self, fw_spec):
        objective_with_inc = fw_spec["%s_eval_metrics_with_inc" % EVAL_SCRIPT][OBJECTIVE_METRIC]
        objective_with_dec = fw_spec["%s_eval_metrics_with_dec" % EVAL_SCRIPT][OBJECTIVE_METRIC]
        orig_objective = fw_spec["%s_eval_metrics" % EVAL_SCRIPT][OBJECTIVE_METRIC]

        (param_name, row_idx, col_idx) = get_param(fw_spec['param_idx'], fw_spec)

        if orig_objective>=objective_with_inc and orig_objective>=objective_with_dec:
            fw_spec[param_name] = fw_spec['orig_param_val']#update parameter
            fw_spec["alpha"][fw_spec['param_idx']] *= alpha_dec#update parameter's alpha
            best_obj = fw_spec["%s_eval_metrics" % EVAL_SCRIPT][OBJECTIVE_METRIC]
            change_for_best_obj = 'const'
        elif objective_with_dec>=objective_with_inc and objective_with_dec>orig_objective:
            fw_spec[param_name] = fw_spec['dec_param_val']#update parameter
            fw_spec["alpha"][fw_spec['param_idx']] *= alpha_inc#update parameter's alpha
            fw_spec["%s_eval_metrics" % EVAL_SCRIPT] = fw_spec["%s_eval_metrics_with_dec" % EVAL_SCRIPT]#update baseline metrics for next iteration
            best_obj = fw_spec["%s_eval_metrics_with_dec" % EVAL_SCRIPT][OBJECTIVE_METRIC]
            change_for_best_obj = 'dec'

        elif objective_with_inc>objective_with_dec and objective_with_inc>orig_objective:
            fw_spec[param_name] = fw_spec['inc_param_val']#update parameter         
            fw_spec["alpha"][fw_spec['param_idx']] *= alpha_inc#update parameter's alpha
            fw_spec["%s_eval_metrics" % EVAL_SCRIPT] = fw_spec["%s_eval_metrics_with_inc" % EVAL_SCRIPT]#update baseline metrics for next iteration            
            best_obj = fw_spec["%s_eval_metrics_with_inc" % EVAL_SCRIPT][OBJECTIVE_METRIC]
            change_for_best_obj = 'inc'

        else:
            print "Coding Error ChooseNextIter()"
            print (objective_with_inc, objective_with_dec, orig_objective)
            sys.exit(1);               

        fw_spec['param_idx'] = inc_parameter(fw_spec['param_idx'], fw_spec)

        if fw_spec['param_idx'] == 0:
            val_spec = copy.deepcopy(fw_spec)
            val_spec['TRAINING_SEQUENCES'] = VALIDATION_SEQUENCES    
            val_spec['seq_idx_to_eval'] = VALIDATION_SEQUENCES       
            val_spec['validation_eval'] = True   
            val_batch = Firework(RunRBPF_Batch(), spec = val_spec)    
            val_eval = Firework(RunEval(), spec = val_spec)
            storeResultsFW = Firework(StoreResultsInDatabase(), spec=val_spec)

            next_iter_firework = Firework(Iterate(), fw_spec)

            workflow = Workflow([val_spec, val_eval, storeResultsFW, next_iter_firework], 
                                {val_spec: [val_eval], val_eval: [storeResultsFW]})

        else:
            next_iter_firework = Firework(Iterate(), fw_spec)

            workflow = Workflow([next_iter_firework])

        return FWAction(stored_data = {'best_obj': best_obj,
                                       'change_for_best_obj': change_for_best_obj,
                                       'parameter_changed_name': param_name,
                                       'parameter_changed_val': fw_spec[param_name]},
                        additions = workflow)



def get_indices_covMatrix(param_idx, d):
    """
    Since covariance matrices must be symmetric, we only iterate over indices in
    their upper halves (including diagonal)

    Inputs:
    - param_idx: A single integer representing the index in the covariance matrix
    - d: dimension of the covariance matrix

    Outputs:
    - row_idx: The row indicated by param_idx
    - col_idx: The column indicated by param_idx

    For a 3x3 matrix (d=3) we would return (row_idx, col_idx) in the following order
    (where the values in the matrix indicate param_idx):
    [0 1 2]
    [  3 4]
    [    5]
    """
    assert(param_idx <(d+1)*d/2)
    for r in range(d):
        for c in range(r, d):
            if param_idx == 0:
                row_idx = r
                col_idx = c
                return (row_idx, col_idx)
            else:
                param_idx -= 1


def get_param(param_idx, spec):
    """
    Inputs:
    - param_idx: integer, specifying a particular parameter

    Outputs:
    - param_name: string, specifying the name of the parameter matrix (e.g. 'Q' or 'R')
    - row_idx: integer, specifying the row within the parameter matrix of the parameter
    - col_idx: integer, specifying the column within the parameter matrix of the parameter
    """
    if param_idx < len(spec['det_names']):
        param_name = 'det_grouping_min_overlap'
        det_name = spec['det_names'][param_idx] 
        idx = -1

    else:
        param_name = 'target_detection_max_dists'
        idx = param_idx - len(spec['det_names'])
        det_name = None

    return (param_name, det_name, idx)

def inc_parameter(param_idx, spec):
    """
    Inputs:
    - param_idx: The current parameter index

    Outputs:
    - new_idx: The next parameter index.  Equals param_idx+1, wrapped around
    to zero when we have reached the last parameter.
    """
    new_idx = (param_idx + 1) % ((len(spec['det_names'])) + len(spec['target_detection_max_dists']))
    return new_idx

@explicit_serialize
class StoreResultsInDatabase(FireTaskBase):   
    #Does nothing, but the eval firework passes results along, so we can
    #see the evaluation metrics in the database
    def run_task(self, fw_spec):
        return



if __name__ == "__main__":
    # write new launchpad file, not positive if this is necessary
    create_launchpad()
    # set up the LaunchPad and reset it
    launchpad = LaunchPad(host=MONGODB_HOST, port=MONGODB_PORT, name=MONGODB_NAME, username=MONGODB_USERNAME, password=MONGODB_PASSWORD,
                     logdir=None, strm_lvl='INFO', user_indices=None, wf_user_indices=None, ssl_ca_file=None)
    launchpad.reset('', require_password=False)

    det1_name = 'mscnn'
    det2_name = 'regionlets'
    include_ignored_gt=False
    include_dontcare_in_gt=False
    sort_dets_on_intervals=True

    description_of_run = get_description_of_run(include_ignored_gt, include_dontcare_in_gt,
                    sort_dets_on_intervals, det1_name, det2_name)
    results_folder_name = '%s/%d_particles' % (description_of_run, NUM_PARTICLES)
    results_folder = '%s/%s/%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name)
    spec = {
            'det_names': ['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets'],
            'num_particles': NUM_PARTICLES,
            'include_ignored_gt': False,
            'include_dontcare_in_gt': False,
            'sort_dets_on_intervals': True,
            'results_folder': results_folder,
            'use_corrected_eval': True,
            'CHECK_K_NEAREST_TARGETS': CHECK_K_NEAREST_TARGETS,                        
            'K_NEAREST_TARGETS': K_NEAREST_TARGETS,                        
            'RUN_ONLINE': RUN_ONLINE,
            'ONLINE_DELAY': ONLINE_DELAY,
            'MAX_1_MEAS_UPDATE': MAX_1_MEAS_UPDATE,                    
            'UPDATE_MULT_MEAS_SIMUL': UPDATE_MULT_MEAS_SIMUL,
            'TREAT_MEAS_INDEP': TREAT_MEAS_INDEP,                        
            'TREAT_MEAS_INDEP_2': TREAT_MEAS_INDEP_2,
            'USE_CONSTANT_R': USE_CONSTANT_R,
            'P': P_default.tolist(),
            'R': R_default.tolist(),
            'Q': Q_default.tolist(),
            'TRAINING_SEQUENCES': TRAINING_SEQUENCES,
            'NUM_RUNS': NUM_RUNS,
            'param_idx': 0, #index of the parameter to adjust next in coordinate descent
            'alpha': ALPHA_INIT,
            'coord_ascent_iter': 0,
            'derandomize_with_seed': True,
            'use_general_num_dets': True,
            'scale_prior_by_meas_orderings': 'ignore_meas_orderings',
            'set_birth_clutter_prop_equal': False,
            'birth_clutter_likelihood': 'aprox1',
            'proposal_distr': 'min_cost',
            'use_log_probs': 'True',
            'normalize_log_importance_weights': True,
            #the minimum allowed box overlap for each detection source when associating
            #detections into groups
            'det_grouping_min_overlap': {'mscnn':.5, 
                                         '3dop':.5,
                                         'mono3d':.5,
                                         'mv3d':.5,
                                         'regionlets':.5},
            #'distance' or 'box_overlap', metric used when computing min cost measurment
            #target association assignment                             
            'targ_meas_assoc_metric': 'distance',
            #propose target measurement association with these distances as the 
            #maximum allowed distance when finding minimum cost assignment                                     
            'target_detection_max_dists': [15, 50, 150]


#            'det_names': ['mscnn', 'regionlets'],
#            'num_particles': 100,
#            'include_ignored_gt': False,
#            'include_dontcare_in_gt': False,
#            'sort_dets_on_intervals': True,
#            'results_folder': results_folder,
#            'CHECK_K_NEAREST_TARGETS': True,                        
#            'K_NEAREST_TARGETS': 1,                        
#            'RUN_ONLINE': True,
#            'ONLINE_DELAY': online_delay,
#            'MAX_1_MEAS_UPDATE': max_1_meas_update_local,                    
#            'UPDATE_MULT_MEAS_SIMUL': update_simul_local,
#            'TREAT_MEAS_INDEP': TREAT_MEAS_INDEP,                        
#            'TREAT_MEAS_INDEP_2': TREAT_MEAS_INDEP_2,
#            'USE_CONSTANT_R': USE_CONSTANT_R,
#            'P': P_default.tolist(),
#            'R': R_default.tolist(),
#            'Q': Q_default.tolist(),
#            'scale_prior_by_meas_orderings': scale_prior_by_meas_orderings,
#            'derandomize_with_seed': True,
#            'use_general_num_dets': use_general_num_dets,
#            #if true, set the prior probability of birth and clutter equal in
#            #the proposal distribution, using the clutter prior for both
#            'set_birth_clutter_prop_equal': False,
#            'birth_clutter_likelihood': birth_clutter_likelihood,
#            'proposal_distr': proposal_distr,
#            'use_log_probs': 'True',
#            'normalize_log_importance_weights': True,
#            #the minimum allowed box overlap for each detection source when associating
#            #detections into groups
#            'det_grouping_min_overlap': {'mscnn':.5, 
#                                         '3dop':.5,
#                                         'mono3d':.5,
#                                         'mv3d':.5,
#                                         'regionlets':.5},
#            #'distance' or 'box_overlap', metric used when computing min cost measurment
#            #target association assignment                             
#            'targ_meas_assoc_metric': 'distance',
#            #propose target measurement association with these distances as the 
#            #maximum allowed distance when finding minimum cost assignment                                     
#            'target_detection_max_dists': [15, 50, 150]   
            }
            


#    spec['mod_direction'] = 'const'
    init_batch = Firework(RunRBPF_Batch(), spec = spec)
    eval_spec = copy.deepcopy(spec)
    eval_spec['seq_idx_to_eval'] = eval_spec['TRAINING_SEQUENCES']           
    eval_init = Firework(RunEval(), spec = eval_spec)
    storeResultsFW1 = Firework(StoreResultsInDatabase(), spec=eval_spec)


    val_spec = copy.deepcopy(spec)
    val_spec['TRAINING_SEQUENCES'] = VALIDATION_SEQUENCES    
    val_spec['seq_idx_to_eval'] = VALIDATION_SEQUENCES         
    val_spec['validation_eval'] = True   
    val_batch = Firework(RunRBPF_Batch(), spec = val_spec)    
    val_eval = Firework(RunEval(), spec = val_spec)
    storeResultsFW2 = Firework(StoreResultsInDatabase(), spec=val_spec)

    first_iter= Firework(Iterate(), spec = spec)
    workflow = Workflow([init_batch, eval_init, first_iter, val_batch, val_eval, storeResultsFW1, storeResultsFW2], 
                        {init_batch: [eval_init], eval_init: [first_iter], val_batch: [val_eval], eval_init:[storeResultsFW1], val_eval:[storeResultsFW2]})

    launchpad.add_wf(workflow)
    qadapter = CommonAdapter.from_file("%sfireworks_files/my_qadapter.yaml" % RBPF_HOME_DIRECTORY)
#    rapidfire(launchpad, FWorker(), qadapter, launch_dir='.', nlaunches='infinite', njobs_queue=20,
#                  njobs_block=500, sleep_time=None, reserve=False, strm_lvl='INFO', timeout=None,
#                  fill_mode=False)


    fworker = FWorker()
    rapidfire(launchpad, fworker, qadapter, launch_dir='.', nlaunches='infinite', njobs_queue=20,
      njobs_block=500, sleep_time=None, reserve=False, strm_lvl="DEBUG", timeout=None,
      fill_mode=False)










