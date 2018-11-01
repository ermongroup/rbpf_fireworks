# If the database thinks a firework is still running, but no jobs are running on the cluster, try:
# $ lpad detect_lostruns --time 1 --refresh
#
# If fireworks are stuck at RESERVED (or if you want to rerun FIZZLED, change RESERVED to FIZZLED)
# $ lpad rerun_fws -s RESERVED
# $ qlaunch rapidfire -m 20 --nlaunches infinite
#
# If a firework fizzles and you are trying to find the error/output, note the fireworks fw_id
# in the online database, then search for this fw_id in the launcher block, e.g.:
# $ cd block_2017-11-01-07-30-53-457640
# $ pt 'fw_id: 34'
# or on atlas-ws-6 use silver searcher:
# $ ag 'fw_id: 34'
#
#Note, on Atlas before this script:
# start a krbscreen session:
# $ krbscreen #reattach using $ screen -rx
# $ reauth #important so that jobs can be submitted after logging out, enter password
#
# $ export PATH=/opt/rh/python27/root/usr/bin:$PATH
# $ export LD_LIBRARY_PATH=/opt/rh/python27/root/usr/lib64/:$LD_LIBRARY_PATH
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
#
# Install pymatgen:
#$ pip install --user pymatgen
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
#$ pip2.7 install --user pymatgen
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

LOCAL_TESTING = False
if LOCAL_TESTING:
    from fireworks.core.rocket_launcher import rapidfire
else:
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
SEQUENCES_TO_PROCESS = [i for i in reversed([i for i in range(21)])]
# SEQUENCES_TO_PROCESS = [0]
#SEQUENCES_TO_PROCESS = [4]
# SEQUENCES_TO_PROCESS = [0,1,2,3,4,5,6]
# SEQUENCES_TO_PROCESS = range(56)
# SEQUENCES_TO_PROCESS = [34]
#SEQUENCES_TO_PROCESS = [0,2,3,4,5,6,10]
#SEQUENCES_TO_PROCESS = [0]
#SEQUENCES_TO_PROCESS = [11]
#SEQUENCES_TO_PROCESS = [13,14,15]
#SEQUENCES_TO_PROCESS = [7]
#NUM_PARTICLES_TO_TEST = [5, 10, 20, 50, 100]
#NUM_PARTICLES_TO_TEST = [1]
# NUM_PARTICLES_TO_TEST = [10]#, 20, 50, 100]
NUM_PARTICLES_TO_TEST = [10, 100]
# NUM_PARTICLES_TO_TEST = [10]
#NUM_PARTICLES_TO_TEST = [5, 20, 50, 100]


###################################### Experiment Organization ######################################
# DATA_SET_NAME = 'MOT17'
DATA_SET_NAME = 'KITTI_split'
# DATA_SET_NAME = 'MOT17_split'
# DATA_SET_NAME = 'KITTI'
#DIRECTORY_OF_ALL_RESULTS = '%sSUMMER_2018/reproduce3_%s/' % (RBPF_HOME_DIRECTORY, DATA_SET_NAME)
# DIRECTORY_OF_ALL_RESULTS = '%sSUMMER_2018/save_MAP_particle_weights%s/' % (RBPF_HOME_DIRECTORY, DATA_SET_NAME)

# DIRECTORY_OF_ALL_RESULTS = '%sFALL_2018/get_prob_matrix%s/' % (RBPF_HOME_DIRECTORY, DATA_SET_NAME)
# DIRECTORY_OF_ALL_RESULTS = '%sFALL_2018/exact_sampling114%s/' % (RBPF_HOME_DIRECTORY, DATA_SET_NAME)
DIRECTORY_OF_ALL_RESULTS = '%sFALL_2018/exact_sampling_compareSUBCNN%s/' % (RBPF_HOME_DIRECTORY, DATA_SET_NAME)
# DIRECTORY_OF_ALL_RESULTS = '%sFALL_2018/try_current_again_resampleRatio4%s' % (RBPF_HOME_DIRECTORY, DATA_SET_NAME)
#CUR_EXPERIMENT_BATCH_NAME = 'full_support_proposal/'
#CUR_EXPERIMENT_BATCH_NAME = 'imgFeat_killAllUnassoc/'
#CUR_EXPERIMENT_BATCH_NAME = 'gt_assoc_sourcesToGT_beforeGrping/'
CUR_EXPERIMENT_BATCH_NAME = 'no_img_features/'
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

# R_DEFAULT = np.array([[40,0],[0,5]])
R_DEFAULT = np.array([[ 0.01,   0.0],
                      [ 0.0,   0.01]])    

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
                           det_names, train_test='train'):
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
    description_of_run = description_of_run + '_' + train_test

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
    # write new launchpad file
    create_launchpad()

    # set up the LaunchPad and reset it
    launchpad = LaunchPad(host=MONGODB_HOST, port=MONGODB_PORT, name=MONGODB_NAME, username=MONGODB_USERNAME, password=MONGODB_PASSWORD,
                     logdir=None, strm_lvl='INFO', user_indices=None, wf_user_indices=None)#, ssl_ca_file=None)
                     # logdir=None, strm_lvl='INFO', user_indices=None, wf_user_indices=None, ssl_ca_file=None)
    launchpad.reset('', require_password=False)
           
    #the number of training sequences (separate videos) for different datasets
    training_seq_count = {\
        'KITTI': 21,
        'KITTI_split': 90,        
        '2DMOT2015': 11,
        'MOT16': 7,
        'MOT17': 7,
        'MOT17_split': 56
    }

    #the number of test sequences (separate videos) for different datasets
    test_seq_count = {\
        'KITTI': 29,
        'KITTI_split': 127,
        '2DMOT2015': 11,
        'MOT16': 7,
        'MOT17': 7,
        'MOT17_split': 61
    }

    if DATA_SET_NAME == 'MOT16':
        obj_class = 'pedestrian'
        data_path = '/atlas/u/jkuck/%s/kitti_format' % DATA_SET_NAME
        pickled_data_dir = "%s/learn_params1_pickled_data" % data_path
        #list of image widths (in pixels) by sequence
        image_widths = [1920, 1920, 640, 1920, 1920, 1920, 1920]
        #list of camera pixel heights by sequence
        image_heights = [1080, 1080, 480, 1080, 1080, 1080, 1080]
        det_sets_to_run = [['single_det_src']]

    if DATA_SET_NAME == 'MOT17':
        obj_class = 'pedestrian'
        data_path = '/atlas/u/jkuck/%s/kitti_format' % DATA_SET_NAME
        pickled_data_dir = "%s/learn_params1_pickled_data" % data_path
        #list of image widths (in pixels) by sequence, for train and test sets
        image_widths = [1920, 1920, 640, 1920, 1920, 1920, 1920]
        #list of image heights (in pixels) by sequence, for train and test sets
        image_heights = [1080, 1080, 480, 1080, 1080, 1080, 1080]
#        det_sets_to_run = [['DPM', 'FRCNN', 'SDP']]
        # det_sets_to_run = [['DPM'], ['DPM', 'FRCNN', 'SDP'], ['FRCNN', 'SDP'], ['DPM', 'SDP'], ['DPM', 'FRCNN'], ['FRCNN'], ['SDP']]
        # det_sets_to_run = [['DPM']]
        # det_sets_to_run = [['DPM', 'FRCNN', 'SDP']]

        det_sets_to_run = [['SDP']]

    if DATA_SET_NAME == 'MOT17_split':
        obj_class = 'pedestrian'
        data_path = '/atlas/u/jkuck/%s/kitti_format' % DATA_SET_NAME
        pickled_data_dir = "%s/learn_params1_pickled_data" % data_path
        #list of image widths (in pixels) by sequence
        training_image_widths = [1920 for i in range(17)] + [640 for i in range(9)] + [1920 for i in range(30)] 
        testing_image_widths = [1920 for i in range(20)] + [640 for i in range(12)] + [1920 for i in range(29)] 

        #list of image heights (in pixels) by sequence
        training_image_heights = [1080 for i in range(17)] + [480 for i in range(9)] + [1080 for i in range(30)] 
        testing_image_heights = [1080 for i in range(20)] + [480 for i in range(12)] + [1080 for i in range(29)] 

        det_sets_to_run = [['DPM', 'FRCNN', 'SDP']]
        # det_sets_to_run = [['DPM'], ['DPM', 'FRCNN', 'SDP'], ['FRCNN', 'SDP'], ['DPM', 'SDP'], ['DPM', 'FRCNN'], ['FRCNN'], ['SDP']]
        # det_sets_to_run = [['DPM', 'FRCNN', 'SDP']]
        # det_sets_to_run = [['DPM'], ['DPM', 'FRCNN', 'SDP'], ['FRCNN'], ['SDP']]
        # det_sets_to_run = [['DPM']]

       # det_sets_to_run = [['SDP']]


    #DATA_SET_NAME = 'KITTI'
    elif DATA_SET_NAME == 'KITTI':
        obj_class = 'car'
        pickled_data_dir = "%sKITTI_helpers/learn_params1_pickled_data" % RBPF_HOME_DIRECTORY
        data_path = "%sKITTI_helpers/data" % RBPF_HOME_DIRECTORY
        # det_sets_to_run = [['regionlets'], ['mscnn'], ['3dop'], ['mono3d'], ['mv3d'],['subcnn'],\
        #                     ['mscnn', '3dop'],\
        #                     ['mscnn', '3dop', 'mono3d'],\
        #                     ['mscnn', '3dop', 'mono3d', 'mv3d'],\
        #                     ['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets'],\
        #                     ['subcnn', 'mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]
        det_sets_to_run = [['regionlets'], ['mscnn']]#,\
        #                     ['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]

    elif DATA_SET_NAME == 'KITTI_split':
        obj_class = 'car'
        pickled_data_dir = "%sKITTI_helpers/data_split/learn_params1_pickled_data" % RBPF_HOME_DIRECTORY
        data_path = "%sKITTI_helpers/data_split" % RBPF_HOME_DIRECTORY
        # det_sets_to_run = [['mscnn', 'mv3d'],\
        #                     ['mv3d'], ['mscnn']]
        # det_sets_to_run = [['mscnn']]
        # det_sets_to_run = [['mscnn'], ['subcnn', 'mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]
        # det_sets_to_run = [['mscnn'], ['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]
        det_sets_to_run = [['subcnn', 'mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets'], ['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]
        # det_sets_to_run = [['subcnn', 'mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets'],\
        #                     ['subcnn'], ['mscnn']]

# [['regionlets'], ['mscnn'], ['3dop'], ['mono3d'], ['mv3d'],\
#                             ['mscnn', '3dop'],\
#                             ['mscnn', '3dop', 'mono3d'],\
#                             ['mscnn', '3dop', 'mono3d', 'mv3d'],\
#                             ['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]


        #lists of image heights and widths (in pixels) by sequence
#        if train_test == 'train':
#            image_widths = [1242 for i in range(training_seq_count['KITTI'])]
#            image_heights = [375 for i in range(training_seq_count['KITTI'])]
#        else:
#            assert(train_test == 'test')
#            image_widths = [1242 for i in range(test_seq_count['KITTI'])]
#            image_heights = [375 for i in range(test_seq_count['KITTI'])]

    include_ignored_gt=False
    include_dontcare_in_gt=False
    sort_dets_on_intervals=True

    all_fireworks = []
    firework_dependencies = {}

    #targ_meas_assoc_metric = 'distance'
    online_delay = 0
    birth_clutter_likelihood = 'aprox1'
    birth_clutter_model = 'poisson'# 'poisson' or 'training_counts'
    use_general_num_dets = True

    birth_clutter_model = 'poisson'
    birth_clutter_likelihood = 'aprox1'
    scale_prior_by_meas_orderings = 'count_multi_src_orderings'

    check_k_nearest = False

    gumbel_scale = 0

    print 'HOWDY!!'
    debug_counter = 0
    for train_test in ['train']:
        for online_delay in [0]:
            # for (proposal_distr, targ_meas_assoc_metric) in [('modified_SIS_gumbel', 'box_overlap')]:
            # for (proposal_distr, targ_meas_assoc_metric) in [('exact_sampling', 'distance'), ('min_cost_corrected', 'distance')]:
            for (proposal_distr, targ_meas_assoc_metric) in [('modified_SIS_gumbel', 'distance')]:
            # for (proposal_distr, targ_meas_assoc_metric) in [('min_cost_corrected', 'box_overlap'), ('min_cost_corrected', 'distance')]:
            # for (proposal_distr, targ_meas_assoc_metric) in [('min_cost_corrected', 'box_overlap'), ('exact_sampling', 'box_overlap')]:
            #for (proposal_distr, targ_meas_assoc_metric) in [('ground_truth_assoc', 'box_overlap')]:
            #for (proposal_distr, targ_meas_assoc_metric) in [('ground_truth_assoc', 'box_overlap'), ('min_cost', 'box_overlap')]:
#            for (proposal_distr, targ_meas_assoc_metric) in [('ground_truth_assoc', 'distance')]:
#            for proposal_distr in ['modified_SIS_gumbel']:
#            for proposal_distr in ['min_cost', 'sequential', 'modified_SIS_w_replacement']:
#            for proposal_distr in ['modified_SIS_w_replacement']:
#            for (proposal_distr, gumbel_scale) in [('modified_SIS_gumbel', 0)]:#, ('modified_SIS_gumbel', 1)]:
#            for (proposal_distr, gumbel_scale) in [('modified_SIS_gumbel', 0), ('modified_SIS_gumbel', .25), \
#            ('modified_SIS_gumbel', 1), ('modified_SIS_gumbel', 4)]:
#            ('modified_SIS_gumbel', .5), ('modified_SIS_gumbel', 1), ('modified_SIS_gumbel', 2), ('modified_SIS_gumbel', 4)]:
#            for (proposal_distr, gumbel_scale) in [('modified_SIS_gumbel', 1)]:

#            for (proposal_distr, gumbel_scale, num_particles) in [('modified_SIS_gumbel', 0, 5)]:
        
            

#            for (proposal_distr, gumbel_scale, num_particles) in [('modified_SIS_gumbel', 0, 4)]:
#            for (proposal_distr, gumbel_scale, num_particles) in [('modified_SIS_exact', -9999, 4)]:
#Run me tonight:
#            for (proposal_distr, gumbel_scale, num_particles) in \
#                [('modified_SIS_exact', None, 4),
#                 ('modified_SIS_gumbel', 0, 4),
#                 ('modified_SIS_gumbel', 0, 240)]:


#            for (proposal_distr, gumbel_scale) in [('modified_SIS_gumbel', 0), ('modified_SIS_gumbel', .25), ('modified_SIS_gumbel', 1)]:
#            for (proposal_distr, targ_meas_assoc_metric, check_k_nearest) in \
#            [('modified_SIS_gumbel', 'distance', None)]:  
#            [('modified_SIS_min_cost', 'distance', None),
#             ('min_cost', 'distance', None),
#             ('min_cost_corrected', 'distance', None)]:   
#            
#            [('modified_SIS_min_cost', 'distance', None)]:
#            [('modified_SIS_min_cost', 'distance', None),
#             ('modified_SIS_min_cost', 'box_overlap', None),
#             ('min_cost', 'distance', None),
#             ('min_cost', 'box_overlap', None),
#             ('min_cost_corrected', 'distance', None),
#             ('min_cost_corrected', 'box_overlap', None)]:   
#
#             ('sequential', None, True),
#             ('sequential', None, False)]:
#                for det_names in [['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]:
#                for det_names in [['regionlets'], ['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]:
                for det_names in det_sets_to_run:
#                for det_names in [['regionlets']]:
#                for det_names in [['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]:
#                for det_names in [['regionlets']]:
#                        for det_names in [['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets'], ['mscnn', '3dop', 'mono3d', 'mv3d'], \
                    for num_particles in NUM_PARTICLES_TO_TEST:
                        if proposal_distr == 'ground_truth_assoc':
                            num_particles = 1

                        if proposal_distr == 'exact_sampling':
                            normalize_log_importance_weights = False
                        else:
                            normalize_log_importance_weights = True

                        description_of_run = get_description_of_run_gen_detections(include_ignored_gt, include_dontcare_in_gt,
                                        sort_dets_on_intervals, det_names, train_test=train_test)
                        results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
                        results_folder = '%s/%s/%s_online_delay=%d,proposal_distr=%s,targ_meas_assoc_metric=%s,check_k_nearest=%s,gumbel_scale=%f' % \
                            (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name, online_delay,
                            proposal_distr,targ_meas_assoc_metric,check_k_nearest, gumbel_scale)
                                                                        
                        setup_results_folder(results_folder)
                        run_rbpf_fireworks = []
                        if DATA_SET_NAME == 'KITTI_split' and train_test == 'train':
                            SEQUENCES_TO_PROCESS = range(training_seq_count['KITTI_split'])
                            #SEQUENCES_TO_PROCESS = range(56)
                        elif DATA_SET_NAME == 'KITTI_split' and train_test == 'test':
                            SEQUENCES_TO_PROCESS = range(61)

                        print 'NUM_RUNS =', NUM_RUNS
                        print 'SEQUENCES_TO_PROCESS:', SEQUENCES_TO_PROCESS            
                        for run_idx in range(1, NUM_RUNS+1):
                            for seq_idx in SEQUENCES_TO_PROCESS:
                                if DATA_SET_NAME == 'KITTI' and train_test == 'train':
                                    image_widths = [1242 for i in range(training_seq_count['KITTI'])]
                                    image_heights = [375 for i in range(training_seq_count['KITTI'])]
                                elif DATA_SET_NAME == 'KITTI' and train_test == 'test':
                                    image_widths = [1242 for i in range(test_seq_count['KITTI'])]
                                    image_heights = [375 for i in range(test_seq_count['KITTI'])]
                                elif DATA_SET_NAME == 'KITTI_split' and train_test == 'train':
                                    image_widths = [1242 for i in range(training_seq_count['KITTI_split'])]
                                    image_heights = [375 for i in range(training_seq_count['KITTI_split'])]
                                elif DATA_SET_NAME == 'KITTI_split' and train_test == 'test':
                                    image_widths = [1242 for i in range(test_seq_count['KITTI_split'])]
                                    image_heights = [375 for i in range(test_seq_count['KITTI_split'])]

                                elif DATA_SET_NAME == 'MOT17_split' and train_test == 'train':
                                    image_widths = training_image_widths
                                    image_heights = training_image_heights
                                elif DATA_SET_NAME == 'MOT17_split' and train_test == 'test':
                                    image_widths = testing_image_widths
                                    image_heights = testing_image_heights
                                cur_spec = \
                                {'_dupefinder': {'_fw_name': 'DupeFinderExact'}, #enable duplicate cheking
                                'obj_class': obj_class,
                                'data_path': data_path,
                                'pickled_data_dir': pickled_data_dir,
                                'DATA_SET_NAME': DATA_SET_NAME,
                                'image_widths': image_widths,
                                'image_heights': image_heights,
                                'training_seq_count': training_seq_count[DATA_SET_NAME],
                                'num_particles': num_particles,
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
                                #for 'min_cost' proposal distribution, sample measurement associations
                                #uniformly at random with probability 'prob_sample_assoc_uniform' so 
                                #that proposal distribution has full support
                                'prob_sample_assoc_uniform': .05,
                                'use_log_probs': 'True',
                                'normalize_log_importance_weights': normalize_log_importance_weights,                                    
                                #the minimum allowed box overlap for each detection source when associating
                                #detections into groups
                                'det_grouping_min_overlap': {'mscnn':.5, 
                                                             '3dop':.5,
                                                             'mono3d':.5,
                                                             'mv3d':.5,
                                                             'regionlets':.5,
                                                             'DPM':.5, 
                                                             'FRCNN':.5, 
                                                             'SDP':.5},
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
                                    'det_grouping_min_overlap_DPM':[.5, 0, 1],
                                    'det_grouping_min_overlap_FRCNN':[.5, 0, 1],
                                    'det_grouping_min_overlap_SDP':[.5, 0, 1],                                    
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
                                'num_gumbel_partition_samples': 20,
                                'gumbel_scale': gumbel_scale,
                                #compute death probabilities for targets that have been unassociated
                                #for up to death_prob_markov_order time instances, we will assume death probability is unchanged after
                                #this number of time instances in our model                                
                                #'death_prob_markov_order':2,
                                'death_prob_markov_order':3,
                                #if params.SPEC['proposal_distr'] == 'modified_SIS_exact'
                                #we sample from this many of the most likely hypotheses
                                #this number should be large enough that we never draw a sample
                                #from outside these choices, check whether this is the case or increase
                                #dynamically if we sample from outside these choices
                                'num_top_hypotheses_to_sample_from': 200,
                                #compute target emission priors conditioned on target image features
                                #and image features of the target's nearest neighbor detections
                                'condition_emission_prior_img_feat': False,
                                #number of target's nearest neighbor detections to consider
                                'emission_prior_k_NN': 5,
                                #either 'euclidean' or 'bb_overlap', used to find nearest neighbors
                                'emission_prior_distance_metric': 'bb_overlap'}

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

                        print "howdy!", debug_counter, 'len(all_fireworks) =', len(all_fireworks), 'len(run_rbpf_fireworks) =', len(run_rbpf_fireworks)
                        debug_counter += 1                       
                         
                        all_fireworks.extend(eval_fireworks)
                        for fw in run_rbpf_fireworks:
                            firework_dependencies[fw] = eval_fireworks

                        storeResultsFW = Firework(StoreResultsInDatabase(), spec=eval_new_spec)
                        all_fireworks.append(storeResultsFW)
                        firework_dependencies[eval_old_firework] = storeResultsFW
                        firework_dependencies[eval_new_firework] = storeResultsFW


###    NUM_PARTICLES_TO_TEST = [20, 50, 80]#[5, 20, 80, 240, 960]
##    NUM_PARTICLES_TO_TEST = [5, 10]#[5, 20, 80, 240, 960]
##    det_sets_to_run = [['DPM', 'FRCNN', 'SDP'], ['FRCNN', 'SDP']]
##    for train_test in ['test']:
##        for online_delay in [0]:
###            for (proposal_distr) in ['modified_SIS_w_replacement_unique']:
##            #for proposal_distr in ['modified_SIS_wo_replacement_approx', 'modified_SIS_w_replacement', 'modified_SIS_w_replacement_unique']:
##            for proposal_distr in ['min_cost']:
###            for proposal_distr in ['modified_SIS_gumbel']:
###            for proposal_distr in ['min_cost', 'sequential', 'modified_SIS_w_replacement']:
###            for proposal_distr in ['modified_SIS_w_replacement']:
###            for (proposal_distr, gumbel_scale) in [('modified_SIS_gumbel', 0)]:#, ('modified_SIS_gumbel', 1)]:
###            for (proposal_distr, gumbel_scale) in [('modified_SIS_gumbel', 0), ('modified_SIS_gumbel', .25), \
###            ('modified_SIS_gumbel', 1), ('modified_SIS_gumbel', 4)]:
###            ('modified_SIS_gumbel', .5), ('modified_SIS_gumbel', 1), ('modified_SIS_gumbel', 2), ('modified_SIS_gumbel', 4)]:
###            for (proposal_distr, gumbel_scale) in [('modified_SIS_gumbel', 1)]:
##
###            for (proposal_distr, gumbel_scale, num_particles) in [('modified_SIS_gumbel', 0, 5)]:
##        
##            
##
###            for (proposal_distr, gumbel_scale, num_particles) in [('modified_SIS_gumbel', 0, 4)]:
###            for (proposal_distr, gumbel_scale, num_particles) in [('modified_SIS_exact', -9999, 4)]:
###Run me tonight:
###            for (proposal_distr, gumbel_scale, num_particles) in \
###                [('modified_SIS_exact', None, 4),
###                 ('modified_SIS_gumbel', 0, 4),
###                 ('modified_SIS_gumbel', 0, 240)]:
##
##
###            for (proposal_distr, gumbel_scale) in [('modified_SIS_gumbel', 0), ('modified_SIS_gumbel', .25), ('modified_SIS_gumbel', 1)]:
###            for (proposal_distr, targ_meas_assoc_metric, check_k_nearest) in \
###            [('modified_SIS_gumbel', 'distance', None)]:  
###            [('modified_SIS_min_cost', 'distance', None),
###             ('min_cost', 'distance', None),
###             ('min_cost_corrected', 'distance', None)]:   
###            
###            [('modified_SIS_min_cost', 'distance', None)]:
###            [('modified_SIS_min_cost', 'distance', None),
###             ('modified_SIS_min_cost', 'box_overlap', None),
###             ('min_cost', 'distance', None),
###             ('min_cost', 'box_overlap', None),
###             ('min_cost_corrected', 'distance', None),
###             ('min_cost_corrected', 'box_overlap', None)]:   
###
###             ('sequential', None, True),
###             ('sequential', None, False)]:
###                for det_names in [['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]:
###                for det_names in [['regionlets'], ['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]:
##                for det_names in det_sets_to_run:
###                for det_names in [['regionlets']]:
###                for det_names in [['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']]:
###                for det_names in [['regionlets']]:
###                        for det_names in [['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets'], ['mscnn', '3dop', 'mono3d', 'mv3d'], \
##                    for num_particles in NUM_PARTICLES_TO_TEST:
##                        description_of_run = get_description_of_run_gen_detections(include_ignored_gt, include_dontcare_in_gt,
##                                        sort_dets_on_intervals, det_names)
##                        results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
##                        results_folder = '%s/%s/%s_online_delay=%d,proposal_distr=%s,targ_meas_assoc_metric=%s,check_k_nearest=%s,gumbel_scale=%f' % \
##                            (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name, online_delay,
##                            proposal_distr,targ_meas_assoc_metric,check_k_nearest, gumbel_scale)
##                                                                        
##                        setup_results_folder(results_folder)
##                        run_rbpf_fireworks = []
##                        print 'NUM_RUNS =', NUM_RUNS
##                        print 'SEQUENCES_TO_PROCESS:', SEQUENCES_TO_PROCESS            
##                        for run_idx in range(1, NUM_RUNS+1):
##                            for seq_idx in SEQUENCES_TO_PROCESS:
##                                if DATA_SET_NAME == 'KITTI' and train_test == 'train':
##                                    image_widths = [1242 for i in range(training_seq_count['KITTI'])]
##                                    image_heights = [375 for i in range(training_seq_count['KITTI'])]
##                                elif DATA_SET_NAME == 'KITTI' and train_test == 'test':
##                                    image_widths = [1242 for i in range(test_seq_count['KITTI'])]
##                                    image_heights = [375 for i in range(test_seq_count['KITTI'])]
##                                cur_spec = \
##                                {'_dupefinder': {'_fw_name': 'DupeFinderExact'}, #enable duplicate cheking
##                                'obj_class': obj_class,
##                                'data_path': data_path,
##                                'pickled_data_dir': pickled_data_dir,
##                                'DATA_SET_NAME': DATA_SET_NAME,
##                                'image_widths': image_widths,
##                                'image_heights': image_heights,
##                                'training_seq_count': training_seq_count[DATA_SET_NAME],
##                                'num_particles': num_particles,
##                                'include_ignored_gt': False,
##                                'include_dontcare_in_gt': False,
##                                'sort_dets_on_intervals': True,
##                                'det_names': det_names,
##                                'run_idx': run_idx,
##                                'seq_idx': seq_idx,
##                                'results_folder': results_folder,
##                                'CHECK_K_NEAREST_TARGETS': check_k_nearest,                        
##                                'K_NEAREST_TARGETS': 1,                        
##                                'RUN_ONLINE': RUN_ONLINE,
##                                'ONLINE_DELAY': online_delay,
##                                'USE_CONSTANT_R': USE_CONSTANT_R,
##                                'P': P_default.tolist(),
##                                'R': R_DEFAULT.tolist(),
##                                'Q': Q_DEFAULT.tolist(),
##                                'scale_prior_by_meas_orderings': scale_prior_by_meas_orderings,
##                                'derandomize_with_seed': False,
##                                'use_general_num_dets': use_general_num_dets,
##                                #if true, set the prior probability of birth and clutter equal in
##                                #the proposal distribution, using the clutter prior for both
##                                'set_birth_clutter_prop_equal': False,
##                                'birth_clutter_likelihood': birth_clutter_likelihood,
##                                'proposal_distr': proposal_distr,
##                                #for 'min_cost' proposal distribution, sample measurement associations
##                                #uniformly at random with probability 'prob_sample_assoc_uniform' so 
##                                #that proposal distribution has full support
##                                'prob_sample_assoc_uniform': .05,
##                                'use_log_probs': 'True',
##                                'normalize_log_importance_weights': True,                                    
##                                #the minimum allowed box overlap for each detection source when associating
##                                #detections into groups
##                                'det_grouping_min_overlap': {'mscnn':.5, 
##                                                             '3dop':.5,
##                                                             'mono3d':.5,
##                                                             'mv3d':.5,
##                                                             'regionlets':.5,
##                                                             'DPM':.5, 
##                                                             'FRCNN':.5, 
##                                                             'SDP':.5},
##                                #'distance' or 'box_overlap', metric used when computing min cost measurment
##                                #target association assignment                             
##                                'targ_meas_assoc_metric': targ_meas_assoc_metric,
##                                #propose target measurement association with these distances as the 
##                                #maximum allowed distance when finding minimum cost assignment  
##                                #and 'targ_meas_assoc_metric' = 'distance'                                 
##                                'target_detection_max_dists': [15, 50, 150],
##                                #propose target measurement association with these box overlaps as the 
##                                #maximum allowed box overlap when finding minimum cost assignment  
##                                #and 'targ_meas_assoc_metric' = 'box_overlap'                                 
##                                'target_detection_max_overlaps': [.25, .5, .75],
##                                'coord_ascent_params':{ #first entry in each list is the parameter value, second is the parameter's alpha value
##                                    'birth_proposal_prior_const': [1.0, 2.0],
##                                    'clutter_proposal_prior_const': [1.0, 2.0],
##                                    'birth_model_prior_const': [1.0, 2.0],
##                                    'clutter_model_prior_const': [1.0, 2.0],
##                                    'det_grouping_min_overlap_mscnn': [.5, 0, 1],
##                                    'det_grouping_min_overlap_3dop': [.5, 0, 1],
##                                    'det_grouping_min_overlap_mono3d': [.5, 0, 1],
##                                    'det_grouping_min_overlap_mv3d': [.5, 0, 1],
##                                    'det_grouping_min_overlap_DPM':[.5, 0, 1],
##                                    'det_grouping_min_overlap_FRCNN':[.5, 0, 1],
##                                    'det_grouping_min_overlap_SDP':[.5, 0, 1],                                    
##                                    'det_grouping_min_overlap_regionlets': [.5, 0, 1],
##                                    'target_detection_max_dists_0': [15, 1.4],
##                                    'target_detection_max_dists_1': [50, 1.4],
##                                    'target_detection_max_dists_2': [150, 1.4]
##                                    },
##                                'train_test': train_test,  #should be 'train', 'test', or 'generated_data'                                                                        
##        #                        'gt_path': None #None for KITTI data, file path (string) for synthetic data
##                                'gt_path': None, #None for KITTI data, file path (string) for synthetic data
##                                'data_generation_spec': None,
##                                'birth_clutter_model':birth_clutter_model,
##                                #the number of samples we will use to compute the expected value of the partition function 
##                                #using an approximation to the Gumbel max trick
##                                'num_gumbel_partition_samples': 20,
##                                'gumbel_scale': gumbel_scale,
##                                #compute death probabilities for targets that have been unassociated
##                                #for up to death_prob_markov_order time instances, we will assume death probability is unchanged after
##                                #this number of time instances in our model                                
##                                #'death_prob_markov_order':2,
##                                'death_prob_markov_order':3,
##                                #if params.SPEC['proposal_distr'] == 'modified_SIS_exact'
##                                #we sample from this many of the most likely hypotheses
##                                #this number should be large enough that we never draw a sample
##                                #from outside these choices, check whether this is the case or increase
##                                #dynamically if we sample from outside these choices
##                                'num_top_hypotheses_to_sample_from': 200,
##                                #compute target emission priors conditioned on target image features
##                                #and image features of the target's nearest neighbor detections
##                                'condition_emission_prior_img_feat': True,
##                                #number of target's nearest neighbor detections to consider
##                                'emission_prior_k_NN': 5,
##                                #either 'euclidean' or 'bb_overlap', used to find nearest neighbors
##                                'emission_prior_distance_metric': 'bb_overlap'}
##
##                                cur_firework = Firework(RunRBPF(), spec=cur_spec)
##        #                       cur_firework = Firework(PyTask(func='rbpf.run_rbpf', auto_kwargs=False, kwargs=cur_spec))
##
##                                run_rbpf_fireworks.append(cur_firework)
##
##
##    #                   seq_idx_to_eval = [i for i in range(21)]
##                        seq_idx_to_eval = SEQUENCES_TO_PROCESS
##                        eval_old_spec = copy.deepcopy(cur_spec)
##                        eval_old_spec['seq_idx_to_eval'] = seq_idx_to_eval 
##                        eval_old_spec['use_corrected_eval'] = False
##                        eval_old_firework = Firework(RunEval(), spec=eval_old_spec)
##
##                        eval_new_spec = copy.deepcopy(cur_spec)
##                        eval_new_spec['seq_idx_to_eval'] = seq_idx_to_eval 
##                        eval_new_spec['use_corrected_eval'] = True
##                        eval_new_firework = Firework(RunEval(), spec=eval_new_spec)
##
##                        eval_fireworks = [eval_old_firework, eval_new_firework]
##                        all_fireworks.extend(run_rbpf_fireworks)
##
##                        print "howdy!", debug_counter, 'len(all_fireworks) =', len(all_fireworks), 'len(run_rbpf_fireworks) =', len(run_rbpf_fireworks)
##                        debug_counter += 1                       
##                         
##                        all_fireworks.extend(eval_fireworks)
##                        for fw in run_rbpf_fireworks:
##                            firework_dependencies[fw] = eval_fireworks
##
##                        storeResultsFW = Firework(StoreResultsInDatabase(), spec=eval_new_spec)
##                        all_fireworks.append(storeResultsFW)
##                        firework_dependencies[eval_old_firework] = storeResultsFW
##                        firework_dependencies[eval_new_firework] = storeResultsFW


    # store workflow and launch it
    workflow = Workflow(all_fireworks, firework_dependencies)
    if LOCAL_TESTING:
        launchpad.add_wf(workflow)
        rapidfire(launchpad, FWorker())
    else:
        launchpad.add_wf(workflow)
        qadapter = CommonAdapter.from_file("%sfireworks_files/my_qadapter.yaml" % RBPF_HOME_DIRECTORY)
        rapidfire(launchpad, FWorker(), qadapter, launch_dir='.', nlaunches='infinite', njobs_queue=200,
                      njobs_block=500, sleep_time=None, reserve=False, strm_lvl='INFO', timeout=None,
                      fill_mode=False)





