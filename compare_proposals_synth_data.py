#Note, on Atlas before this script:
# $ PACKAGE_DIR=/atlas/u/jkuck/software
# $ export PATH=$PACKAGE_DIR/anaconda2/bin:$PATH
# $ export LD_LIBRARY_PATH=$PACKAGE_DIR/anaconda2/local:$LD_LIBRARY_PATH
# $ source activate anaconda_venv
# $ cd /atlas/u/jkuck/rbpf_fireworks/
#
# To install anaconda packages run, e.g.:
# $ conda install -c matsci fireworks=1.3.9
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
#$ ml load python/2.7.5
#$ easy_install-2.7 --user pip
#$ export PATH=~/.local/bin:$PATH
# $ pip2.7 install --user fireworks #and others
# $ cd /scratch/users/kuck/rbpf_fireworks/
#
# Add the following line to the file ~/.bashrc on Sherlock:
# export PYTHONPATH="/scratch/users/kuck/rbpf_fireworks:$PYTHONPATH"
# Weird, but to run commands like "lpad -l my_launchpad.yaml get_fws",
# add the following line to the file ~/.bashrc.user on Atlas:
# export PYTHONPATH="${PYTHONPATH}:/scratch/users/kuck/rbpf_fireworks/KITTI_helpers/"
import copy
import os
import errno
import sys
import numpy as np
from fireworks import Firework, Workflow, FWorker, LaunchPad
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FWAction, FireTaskBase

#local:
from fireworks.core.rocket_launcher import rapidfire
#remote:
#from fireworks.queue.queue_launcher import rapidfire

from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from fw_tutorials.dynamic_wf.fibadd_task import FibonacciAdderTask
#from fireworks.core.firework import FWAction, Firework, FiretaskBase
#from fireworks.user_objects.firetasks.script_task import PyTask
from cluster_config import RBPF_HOME_DIRECTORY, MONGODB_USERNAME, MONGODB_PASSWORD
from experiment_config import MONGODB_HOST, MONGODB_PORT, MONGODB_NAME
from rbpf import RunRBPF
sys.path.insert(0, "%sKITTI_helpers" % RBPF_HOME_DIRECTORY)
from jdk_helper_evaluate_results import RunEval
from create_launchpad import create_launchpad
sys.path.insert(0, "%sgeneral_tracking" % RBPF_HOME_DIRECTORY)
from global_params import *
from generate_data import GenData
####################################### Experiment Database ######################################
#MONGODB_HOST = 'ds153609.mlab.com'
#MONGODB_PORT = 53609
#MONGODB_NAME = 'atlas_mult_meas'


#from intermediate import RunRBPF
###################################### Experiment Parameters ######################################
NUM_RUNS=1
NUM_SEQUENCES_TO_GENERATE = 2
NUM_TIME_STEPS = 100 #time steps per sequence
NUM_PARTICLES_TO_TEST = [100]


###################################### Experiment Organization ######################################
GENERATED_DATA_DIR = '%sgenerated_data' % RBPF_HOME_DIRECTORY
CUR_GEN_NAME= 'test'

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

if USE_LEARNED_KF_PARAMS:
    P_default = np.array([[40.64558317, 0,           0, 0],
                          [0,          10,           0, 0],
                          [0,           0, 5.56278505, 0],
                          [0,           0,           0, 3]])
#    R_default = np.array([[ 0.0,   0.0],
#                          [ 0.0,   0.0]])
    R_default = np.array([[ 0.01,   0.0],
                          [ 0.0,   0.01]])    
    
    
    #learned from all GT
    Q_DEFAULT = np.array([[  60.33442497,  102.95992102,   -5.50458177,   -0.22813535],
                          [ 102.95992102,  179.84877761,  -13.37640528,   -9.70601621],
                          [  -5.50458177,  -13.37640528,    4.56034398,    9.48945108],
                          [  -0.22813535,   -9.70601621,    9.48945108,   22.32984314]])
    
    Q_DEFAULT = 4*Q_DEFAULT
#####################replicate ORIG
else:
    if SCALED:
        P_default = np.array([[(NOISE_SD*300)**2,      0,           0,  0],
                              [0,          10*300**2,           0,  0],
                              [0,           0,      (NOISE_SD*90)**2,  0],
                              [0,           0,           0, 10*90**2]])

        R_default = np.array([[ (NOISE_SD*300)**2,             0.0],
                              [          0.0,   (NOISE_SD*90)**2]])
        Q_DEFAULT = np.array([[     (300**2)*0.00003333,    (300**2)*0.0050,         0,         0],
                              [         (300**2)*0.0050,       (300**2)*1.0,         0,         0],
                              [              0,         0,(90**2)*0.00003333,    (90**2)*0.0050],
                              [              0,         0,    (90**2)*0.0050,    (90**2)*1.0000]])
        Q_DEFAULT = Q_DEFAULT*10**(-3)
        if TIME_SCALED:
            Q_DEFAULT = np.array([[     (300**2)*0.00003333,    (300**2)*0.0005,         0,         0],
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
        Q_DEFAULT = np.array([[     0.00003333,    0.0050,         0,         0],
                              [         0.0050,       1.0,         0,         0],
                              [              0,         0,0.00003333,    0.0050],
                              [              0,         0,    0.0050,    1.0000]])
        Q_DEFAULT = Q_DEFAULT*10**(-3)

        if TIME_SCALED:
            Q_DEFAULT = np.array([[     0.00003333,    0.0005,         0,         0],
                                  [         0.0005,       .01,         0,         0],
                                  [              0,         0,0.00003333,    0.0005],
                                  [              0,         0,    0.0005,    .01]])  
#####################end replicate ORIG




def get_description_of_run_gen_detections(include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals,
                           det_names):
    det_names_string = ''
    for det_name in det_names:
        det_names_string = det_names_string + det_name
    if (not include_ignored_gt) and (not include_dontcare_in_gt)\
        and sort_dets_on_intervals:
        description_of_run = "%s_with_score_intervals" % det_names_string
    elif (not include_ignored_gt) and (not include_dontcare_in_gt)\
        and (not sort_dets_on_intervals):
        description_of_run = "%s_no_score_intervals" % det_names_string
    else:
        print "Unexpected combination of boolean arguments"
        print include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals, use_mscnn
        sys.exit(1);


    return description_of_run 

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

def submit_single_experiment(det1_name, det2_name, num_particles, include_ignored_gt=False, include_dontcare_in_gt=False, 
    sort_dets_on_intervals=True):
    setup_results_folder(num_particles, include_ignored_gt, include_dontcare_in_gt, \
                         sort_dets_on_intervals, det1_name, det2_name)
    for run_idx in range(1, NUM_RUNS+1):
        for seq_idx in SEQUENCES_TO_PROCESS:
            submit_single_qsub_job(det1_name, det2_name, num_particles=num_particles, include_ignored_gt=include_ignored_gt, 
                include_dontcare_in_gt=include_dontcare_in_gt,
                sort_dets_on_intervals=sort_dets_on_intervals, run_idx=run_idx, seq_idx=seq_idx)


@explicit_serialize
class StoreResultsInDatabase(FireTaskBase):   
    #Does nothing, but the eval firework passes results along, so we can
    #see the evaluation metrics in the database
    def run_task(self, fw_spec):
        return

if __name__ == "__main__":
    # write new launchpad file, not positive if this is necessary
    # create_launchpad()

    # set up the LaunchPad and reset it
    launchpad = LaunchPad(host=MONGODB_HOST, port=MONGODB_PORT, name=MONGODB_NAME, username=MONGODB_USERNAME, password=MONGODB_PASSWORD,
                     logdir=None, strm_lvl='INFO', user_indices=None, wf_user_indices=None, ssl_ca_file=None)
    launchpad.reset('', require_password=False)

    det1_name = 'mscnn'
    det2_name = 'regionlets'
    include_ignored_gt=False
    include_dontcare_in_gt=False
    sort_dets_on_intervals=True

    all_fireworks = []
    firework_dependencies = {}

    train_test = 'generated_data'
    targ_meas_assoc_metric = 'distance'
    online_delay = 0
    birth_clutter_likelihood = 'aprox1'
    scale_prior_by_meas_orderings = 'count_multi_src_orderings'
    use_general_num_dets = True
    max_1_meas_update_local = True
    update_simul_local = False
    
    

    for proposal_distr in ['min_cost']:
#    for proposal_distr in ['min_cost', 'sequential']:
        for num_particles in NUM_PARTICLES_TO_TEST:
            data_folder = "%s/%s" % (GENERATED_DATA_DIR, CUR_GEN_NAME)
            results_folder_name = '%d_particles' % (num_particles)
            results_folder = '%s/%s_proposal_distr=%s' % \
                (data_folder, results_folder_name, proposal_distr)
            setup_results_folder(results_folder)
            run_rbpf_fireworks = []  
            data_generation_spec = \
                {#if True calculate data generation parameters on KITTI training data with measurements of type
                #fw_spec['det_names'], otherwise use data generation parameters we provide in 'data_gen_params'
                'use_KITTI_data_gen_params' : True,
                'det_names' : ['regionlets'], #used if 'use_KITTI_data_gen_params' is True
                'SCORE_INTERVALS' : {'regionlets': [2]}, #used if 'use_KITTI_data_gen_params' is True
                'training_sequences' : [i for i in range(21)], #used if 'use_KITTI_data_gen_params' is True
                #supply data generation parameters if 'use_KITTI_data_gen_params' is False, otherwise set to None
                'data_gen_params' : None,
####                'data_gen_params' : 
####                    {'lamda_c': 1,
####                    'lamda_b': .1,
####                    'p_emission': 1.0,
####                    'process_noise': Q_DEFAULT,
####                    'meas_noise_target_state': np.array([[40,0],[0,5]]),
####                    'avg_bb_birth': np.array([60, 60]),
####                    'var_bb_birth': np.array([[0,0],[0,0]]),
####                    'avg_bb_clutter': np.array([60, 60]),
####                    'var_bb_clutter': np.array([[0,0],[0,0]]),
####                    'BORDER_DEATH_PROBABILITIES': [-99, 0.9426605504587156, 0.6785714285714286, 0.4444444444444444],
####                    'NOT_BORDER_DEATH_PROBABILITIES': [-99, 0.04229195088676671, 0.02284263959390863, 0.03787878787878788]},
                #always supply these parameters, whether 'use_KITTI_data_gen_params' is True or False
                'data_file_path': data_folder,
                'num_seq_to_generate': NUM_SEQUENCES_TO_GENERATE, #how many sequences with these params do we generate?
                'num_time_steps': NUM_TIME_STEPS, #time steps per sequence
                'time_per_time_step': DEFAULT_TIME_STEP,
                'init_vel_cov': np.array([[10, 0],
                                          [0, 10]])}   
            data_gen_firework = Firework(GenData(), spec = data_generation_spec)
            all_fireworks.append(data_gen_firework)

            for run_idx in range(1, NUM_RUNS+1):
                for seq_idx in range(NUM_SEQUENCES_TO_GENERATE):
                    cur_spec = \
                        {'num_particles': num_particles,
                        'include_ignored_gt': False,
                        'include_dontcare_in_gt': False,
                        'sort_dets_on_intervals': True,
                        'det_names': ['regionlets'],
                        'run_idx': run_idx,
                        'seq_idx': seq_idx,
                        'results_folder': results_folder,
                        'CHECK_K_NEAREST_TARGETS': True,                        
                        'K_NEAREST_TARGETS': 1,                        
                        'RUN_ONLINE': RUN_ONLINE,
                        'ONLINE_DELAY': online_delay,
                        'MAX_1_MEAS_UPDATE': max_1_meas_update_local,                    
                        'UPDATE_MULT_MEAS_SIMUL': update_simul_local,
                        'TREAT_MEAS_INDEP': TREAT_MEAS_INDEP,                        
                        'TREAT_MEAS_INDEP_2': TREAT_MEAS_INDEP_2,
                        'USE_CONSTANT_R': USE_CONSTANT_R,
                        'P': P_default.tolist(),
                        'R': R_default.tolist(),
                        'Q': Q_DEFAULT.tolist(),
                        'scale_prior_by_meas_orderings': scale_prior_by_meas_orderings,
                        'derandomize_with_seed': False,
                        'use_general_num_dets': use_general_num_dets,
                        #if true, set the prior probability of birth and clutter equal in
                        #the proposal distribution, using the clutter prior for both
                        'set_birth_clutter_prop_equal': False,
                        'birth_clutter_likelihood': birth_clutter_likelihood,
                        'proposal_distr': proposal_distr,
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
                        'targ_meas_assoc_metric': targ_meas_assoc_metric,
                        #propose target measurement association with these distances as the 
                        #maximum allowed distance when finding minimum cost assignment                                     
                        'target_detection_max_dists': [15, 50, 150],
                        'coord_ascent_params':{ #first entry in each list is the parameter value, second is the parameter's alpha value
                            'birth_proposal_prior_const': [1.0, 2.0],
                            'clutter_proposal_prior_const': [1.0, 2.0],
                            'birth_model_prior_const': [1.0, 2.0],
                            'clutter_model_prior_const': [1.0, 2.0],
                            'det_grouping_min_overlap_mscnn': [.5, 0, 1],
                            'det_grouping_min_overlap_3dop': [.5, 0, 1],
                            'det_grouping_min_overlap_mono3d': [.5, 0, 1],
                            'det_grouping_min_overlap_mv3d': [.5, 0, 1],
                            'det_grouping_min_overlap_regionlets': [.5, 0, 1],
                            'target_detection_max_dists_0': [15, 1.4],
                            'target_detection_max_dists_1': [50, 1.4],
                            'target_detection_max_dists_2': [150, 1.4]
                            },
                        'train_test': train_test,  #should be 'train', 'test', or 'generated_data'                                                                        
#                        'gt_path': None #None for KITTI data, file path (string) for synthetic data
                        'gt_path': "%sground_truth" % data_folder, #None for KITTI data, file path (string) for synthetic data
                        'data_generation_spec': data_generation_spec}

                    cur_firework = Firework(RunRBPF(), spec=cur_spec)
    #                cur_firework = Firework(PyTask(func='rbpf.run_rbpf', auto_kwargs=False, kwargs=cur_spec))

                    run_rbpf_fireworks.append(cur_firework)

#                   seq_idx_to_eval = [i for i in range(21)]
                seq_idx_to_eval = range(NUM_SEQUENCES_TO_GENERATE)
                eval_old_spec = copy.deepcopy(cur_spec)
                eval_old_spec['seq_idx_to_eval'] = seq_idx_to_eval 
                eval_old_spec['use_corrected_eval'] = False
                eval_old_firework = Firework(RunEval(), spec=eval_old_spec)

                eval_new_spec = copy.deepcopy(cur_spec)
                eval_new_spec['seq_idx_to_eval'] = seq_idx_to_eval 
                eval_new_spec['use_corrected_eval'] = True
                eval_new_firework = Firework(RunEval(), spec=eval_new_spec)

                eval_fireworks = [eval_old_firework, eval_new_firework]
                all_fireworks.extend(run_rbpf_fireworks)
                all_fireworks.extend(eval_fireworks)
                
                firework_dependencies[data_gen_firework] = run_rbpf_fireworks
                for fw in run_rbpf_fireworks: 
                    firework_dependencies[fw] = eval_fireworks

                storeResultsFW = Firework(StoreResultsInDatabase(), spec=eval_new_spec)
                all_fireworks.append(storeResultsFW)
                firework_dependencies[eval_old_firework] = storeResultsFW
                firework_dependencies[eval_new_firework] = storeResultsFW



    # store workflow and launch it
    workflow = Workflow(all_fireworks, firework_dependencies)
    #local
    launchpad.add_wf(workflow)
    rapidfire(launchpad, FWorker())
    #remote
#    launchpad.add_wf(workflow)
#    qadapter = CommonAdapter.from_file("%sfireworks_files/my_qadapter.yaml" % RBPF_HOME_DIRECTORY)
#    rapidfire(launchpad, FWorker(), qadapter, launch_dir='.', nlaunches='infinite', njobs_queue=81,
#                  njobs_block=500, sleep_time=None, reserve=False, strm_lvl='INFO', timeout=None,
#                  fill_mode=False)
#
#
#

