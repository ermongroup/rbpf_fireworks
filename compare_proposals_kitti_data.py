# If the database thinks a firework is still running, but no jobs are running on the cluster, try:
# $ lpad detect_lostruns --time 1 --refresh
#
#
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
# To install cvxpy on atlas run (hopefully):
#
#$ export PATH=/opt/rh/python27/root/usr/bin:$PATH
#$ export LD_LIBRARY_PATH=/opt/rh/python27/root/usr/lib64/:$LD_LIBRARY_PATH
#$ pip install --user numpy
#$ pip install --user cvxpy
##########################################################################################
#
#Note, on Sherlock before this script:
#$ ml load python/2.7.5
#$ easy_install-2.7 --user pip
#$ export PATH=~/.local/bin:$PATH
#$ pip2.7 install --user fireworks #and others
#$ pip2.7 install --user filterpy
#$ pip2.7 install --user scipy --upgrade
#$ pip2.7 install --user munkres
#$ cd /scratch/users/kuck/rbpf_fireworks/
#
# Add the following line to the file ~/.bashrc on Sherlock:
# export PYTHONPATH="/scratch/users/kuck/rbpf_fireworks:$PYTHONPATH"
# Weird, but to run commands like "lpad -l my_launchpad.yaml get_fws",
# add the following line to the file ~/.bashrc.user on Atlas:
# export PYTHONPATH="${PYTHONPATH}:/scratch/users/kuck/rbpf_fireworks/KITTI_helpers/"
#
#
# When setting up:
# - make cluster_config.py file
# - make my_qadapter.yaml file (look at fireworks workflow manager website for info)
#
# To install cvxpy on sherlock run:
# $ pip2.7 install --user cvxpy
import copy
import os
import errno
import sys
import numpy as np
from fireworks import Firework, Workflow, FWorker, LaunchPad
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FWAction, FireTaskBase

#local:
#from fireworks.core.rocket_launcher import rapidfire
#remote:
from fireworks.queue.queue_launcher import rapidfire

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
SEQUENCES_TO_PROCESS = [i for i in range(21)]
#SEQUENCES_TO_PROCESS = [0]
#SEQUENCES_TO_PROCESS = [11]
#SEQUENCES_TO_PROCESS = [13,14,15]
#SEQUENCES_TO_PROCESS = [13]
#NUM_PARTICLES_TO_TEST = [20, 50, 125]
NUM_PARTICLES_TO_TEST = [100]


###################################### Experiment Organization ######################################
DIRECTORY_OF_ALL_RESULTS = '%sNIPS_prep/' % RBPF_HOME_DIRECTORY
CUR_EXPERIMENT_BATCH_NAME = 'compare_proposals_kitti/'

###################################### RBPF Parameters ######################################
#Specify how the proposal distribution should be pruned
#CHECK_K_NEAREST_TARGETS = True
#K_NEAREST_TARGETS = 1

#If False, ONLINE_DELAY is maximized (we wait until the video ends before picking a particle)
RUN_ONLINE = True #save near online results 
#near online mode, wait this many frames before picking max weight particle 
#ONLINE_DELAY = 3


USE_CONSTANT_R = True

###################################### Kalman Filter Parameters ######################################

P_default = np.array([[40.64558317, 0,           0, 0],
                      [0,          10,           0, 0],
                      [0,           0, 5.56278505, 0],
                      [0,           0,           0, 3]])
#    R_default = np.array([[ 0.0,   0.0],
#                          [ 0.0,   0.0]])
#    R_default = np.array([[ 0.01,   0.0],
#                          [ 0.0,   0.01]])    

R_DEFAULT = np.array([[40,0],[0,5]])
#learned from all GT
Q_DEFAULT = np.array([[  60.33442497,  102.95992102,   -5.50458177,   -0.22813535],
                      [ 102.95992102,  179.84877761,  -13.37640528,   -9.70601621],
                      [  -5.50458177,  -13.37640528,    4.56034398,    9.48945108],
                      [  -0.22813535,   -9.70601621,    9.48945108,   22.32984314]])

Q_DEFAULT = 4*Q_DEFAULT

INIT_VEL_COV = np.array([[10, 0],
                         [0, 10]])

BB_SIZE = np.array([60, 60])

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
    create_launchpad()

    # set up the LaunchPad and reset it
    launchpad = LaunchPad(host=MONGODB_HOST, port=MONGODB_PORT, name=MONGODB_NAME, username=MONGODB_USERNAME, password=MONGODB_PASSWORD,
                     logdir=None, strm_lvl='INFO', user_indices=None, wf_user_indices=None, ssl_ca_file=None)
    launchpad.reset('', require_password=False)

    include_ignored_gt=False
    include_dontcare_in_gt=False
    sort_dets_on_intervals=True

    all_fireworks = []
    firework_dependencies = {}

    train_test = 'train'
    targ_meas_assoc_metric = 'distance'
    online_delay = 0
    birth_clutter_likelihood = 'aprox1'
    birth_clutter_model = 'poisson'# 'poisson' or 'training_counts'
    scale_prior_by_meas_orderings = 'count_multi_src_orderings'
    use_general_num_dets = True

    birth_clutter_model = 'poisson'
    birth_clutter_likelihood = 'aprox1'
    scale_prior_by_meas_orderings = 'count_multi_src_orderings'
    for train_test in ['train']:
        for online_delay in [0, 3]:
            for (proposal_distr, targ_meas_assoc_metric, check_k_nearest) in \
            [('min_cost', 'distance', None),
            ('min_cost', 'box_overlap', None),
            ('sequential', None, True),
            ('sequential', None, False)]:
                for det_names in [['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]:
#                    for det_names in [['regionlets']]:
#                        for det_names in [['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets'], ['mscnn', '3dop', 'mono3d', 'mv3d'], \
                    for num_particles in NUM_PARTICLES_TO_TEST:
                        description_of_run = get_description_of_run_gen_detections(include_ignored_gt, include_dontcare_in_gt,
                                        sort_dets_on_intervals, det_names)
                        results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
                        results_folder = '%s/%s/%s_onine_delay=%d,proposal_distr=%s,targ_meas_assoc_metric=%s,check_k_nearest=%s' % \
                            (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name, online_delay,
                            proposal_distr,targ_meas_assoc_metric,check_k_nearest)
                                                                        
                        setup_results_folder(results_folder)
                        run_rbpf_fireworks = []            
                        for run_idx in range(1, NUM_RUNS+1):
                            for seq_idx in SEQUENCES_TO_PROCESS:
                                cur_spec = \
                                {'num_particles': num_particles,
                                'include_ignored_gt': False,
                                'include_dontcare_in_gt': False,
                                'sort_dets_on_intervals': True,
                                'det_names': det_names,
                                'run_idx': run_idx,
                                'seq_idx': seq_idx,
                                'results_folder': results_folder,
                                'CHECK_K_NEAREST_TARGETS': check_k_nearest,                        
                                'K_NEAREST_TARGETS': 1,                        
                                'RUN_ONLINE': RUN_ONLINE,
                                'ONLINE_DELAY': online_delay,
                                'USE_CONSTANT_R': USE_CONSTANT_R,
                                'P': P_default.tolist(),
                                'R': R_DEFAULT.tolist(),
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
                                #and 'targ_meas_assoc_metric' = 'distance'                                 
                                'target_detection_max_dists': [15, 50, 150],
                                #propose target measurement association with these box overlaps as the 
                                #maximum allowed box overlap when finding minimum cost assignment  
                                #and 'targ_meas_assoc_metric' = 'box_overlap'                                 
                                'target_detection_max_overlaps': [.25, .5, .75],
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
                                'gt_path': None, #None for KITTI data, file path (string) for synthetic data
                                'data_generation_spec': None,
                                'birth_clutter_model':birth_clutter_model,
                                #the number of samples we will use to compute the expected value of the partition function 
                                #using an approximation to the Gumbel max trick
                                'num_gumbel_partition_samples': 20 }                                                    
                                cur_firework = Firework(RunRBPF(), spec=cur_spec)
        #                       cur_firework = Firework(PyTask(func='rbpf.run_rbpf', auto_kwargs=False, kwargs=cur_spec))

                                run_rbpf_fireworks.append(cur_firework)


    #                   seq_idx_to_eval = [i for i in range(21)]
                        seq_idx_to_eval = SEQUENCES_TO_PROCESS
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
                        for fw in run_rbpf_fireworks:
                            firework_dependencies[fw] = eval_fireworks

                        storeResultsFW = Firework(StoreResultsInDatabase(), spec=eval_new_spec)
                        all_fireworks.append(storeResultsFW)
                        firework_dependencies[eval_old_firework] = storeResultsFW
                        firework_dependencies[eval_new_firework] = storeResultsFW




    # store workflow and launch it
    workflow = Workflow(all_fireworks, firework_dependencies)
    #local
#    launchpad.add_wf(workflow)
#    rapidfire(launchpad, FWorker())
    #remote
    launchpad.add_wf(workflow)
    qadapter = CommonAdapter.from_file("%sfireworks_files/my_qadapter.yaml" % RBPF_HOME_DIRECTORY)
    rapidfire(launchpad, FWorker(), qadapter, launch_dir='.', nlaunches='infinite', njobs_queue=81,
                  njobs_block=500, sleep_time=None, reserve=False, strm_lvl='INFO', timeout=None,
                  fill_mode=False)




