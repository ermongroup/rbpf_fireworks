#Note, on Atlas before this script:
# $ PACKAGE_DIR=/atlas/u/jkuck/software
# $ export PATH=$PACKAGE_DIR/anaconda2/bin:$PATH
# $ export LD_LIBRARY_PATH=$PACKAGE_DIR/anaconda2/local:$LD_LIBRARY_PATH
# $ source activate anaconda_venv
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

#from fireworks.core.rocket_launcher import rapidfire
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

#from intermediate import RunRBPF
###################################### Experiment Parameters ######################################
NUM_RUNS=1
SEQUENCES_TO_PROCESS = [i for i in range(21)]
#SEQUENCES_TO_PROCESS = [0]
#SEQUENCES_TO_PROCESS = [11]
#SEQUENCES_TO_PROCESS = [13,14,15]
#SEQUENCES_TO_PROCESS = [13]
#NUM_PARTICLES_TO_TEST = [25, 100]
NUM_PARTICLES_TO_TEST = [100]


###################################### Experiment Organization ######################################
#DIRECTORY_OF_ALL_RESULTS = './ICML_prep_correctedOnline/propose_k=1_nearest_targets'
DIRECTORY_OF_ALL_RESULTS = '%sICML_prep' % RBPF_HOME_DIRECTORY
#CUR_EXPERIMENT_BATCH_NAME = 'genData_origRBPF_multMeas_probDet95_clutLambdaPoint1_noise05_noShuffle_beta1'
#CUR_EXPERIMENT_BATCH_NAME = 'genData_origRBPF_multMeas_fixedRounding_resampleRatio4_scaled_ShuffleMeas_timeScaled_PQdiv100'
#CUR_EXPERIMENT_BATCH_NAME = 'Rto0_4xQ_multMeas1update_online3frameDelay2'
CUR_EXPERIMENT_BATCH_NAME = 'CHECK_1_NEAREST_TARGETS/Rto0_4xQ_max1MeasUpdate_online3frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = 'CHECK_K_NEAREST_TARGETS=False/Reference/Rto0_4xQ_max1MeasUpdate_online3frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = '/Reference/Rto0_4xQ_max1MeasUpdate_online0frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = 'measuredR_1xQ_max1MeasUpdate_online3frameDelay'

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

    det1_name = 'mscnn'
    det2_name = 'regionlets'
    include_ignored_gt=False
    include_dontcare_in_gt=False
    sort_dets_on_intervals=True

    all_fireworks = []
    firework_dependencies = {}

#####    for det2_name in ['3dop', 'mono3d', 'mv3d', 'mscnn', 'regionlets']:
####    for det2_name in ['regionlets']:
####        for (check_k_near_targets_local, k_nearest_targets_local) in [(False, -1), (False, 1), (False, 2), (False, 3)]:
####            for online_delay_local in [0, 1, 3]:
####                for (max_1_meas_update_local, update_simul_local) in [(True, False), (False, True), (False, False)]:
####                    run_rbpf_fireworks = []
####                    for num_particles in NUM_PARTICLES_TO_TEST:
####                        description_of_run = get_description_of_run(include_ignored_gt, include_dontcare_in_gt,
####                                        sort_dets_on_intervals, det1_name, det2_name)
####                        results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
#####                        results_folder = '%s/%s/%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name)
####                        results_folder = '%s/%s/%s_%s_%s_%s_%s_%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name,
####                            check_k_near_targets_local, k_nearest_targets_local, online_delay_local, max_1_meas_update_local, update_simul_local)
####                        setup_results_folder(results_folder)
####                        for run_idx in range(1, NUM_RUNS+1):
####                            for seq_idx in SEQUENCES_TO_PROCESS:
####                                cur_spec = {'det1_name': 'mscnn',
####                                        'det2_name': det2_name,
####                                        'num_particles': num_particles,
####                                        'include_ignored_gt': False,
####                                        'include_dontcare_in_gt': False,
####                                        'sort_dets_on_intervals': True,
####                                        'run_idx': run_idx,
####                                        'seq_idx': seq_idx,
####                                        'results_folder': results_folder,
####                                        'CHECK_K_NEAREST_TARGETS': check_k_near_targets_local,                        
####                                        'K_NEAREST_TARGETS': k_nearest_targets_local,                        
####                                        'RUN_ONLINE': RUN_ONLINE,
####                                        'ONLINE_DELAY': online_delay_local,
####                                        'MAX_1_MEAS_UPDATE': max_1_meas_update_local,                    
####                                        'UPDATE_MULT_MEAS_SIMUL': update_simul_local,
####                                        'TREAT_MEAS_INDEP': TREAT_MEAS_INDEP,                        
####                                        'TREAT_MEAS_INDEP_2': TREAT_MEAS_INDEP_2,
####                                        'USE_CONSTANT_R': USE_CONSTANT_R,
####                                        'P': P_default.tolist(),
####                                        'R': R_default.tolist(),
####                                        'Q': Q_default.tolist()}
####                                cur_firework = Firework(RunRBPF(), spec=cur_spec)
####                #                cur_firework = Firework(PyTask(func='rbpf.run_rbpf', auto_kwargs=False, kwargs=cur_spec))
####                
####                                run_rbpf_fireworks.append(cur_firework)
####                
####                
####                    seq_idx_to_eval = [i for i in range(21)]
####                    eval_old_firework = Firework(RunEval(), spec={'results_folder': results_folder,
####                                                                  'use_corrected_eval': False,
####                                                                  'seq_idx_to_eval': seq_idx_to_eval})
####                    eval_new_firework = Firework(RunEval(), spec={'results_folder': results_folder,
####                                                                  'use_corrected_eval': True,
####                                                                  'seq_idx_to_eval': seq_idx_to_eval})
####                    eval_fireworks = [eval_old_firework, eval_new_firework]
####                    all_fireworks.extend(run_rbpf_fireworks)
####                    all_fireworks.extend(eval_fireworks)
####                    for fw in run_rbpf_fireworks:
####                        firework_dependencies[fw] = eval_fireworks
#    for det2_name in ['3dop', 'mono3d', 'mv3d', 'mscnn', 'regionlets']:
    for det2_name in ['regionlets']:
        for scale_prior_by_meas_orderings in ['original', 'corrected_with_score_intervals', 'ignore_meas_orderings']:
            for num_particles in NUM_PARTICLES_TO_TEST:
                for (max_1_meas_update_local, update_simul_local) in [(True, False), (False, True), (False, False)]:
	                description_of_run = get_description_of_run(include_ignored_gt, include_dontcare_in_gt,
	                                sort_dets_on_intervals, det1_name, det2_name)
	                results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
	                results_folder = '%s/%s/%s_measOrder=%s,max1_meas=%s,update_simul=%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name, scale_prior_by_meas_orderings,
	                																	max_1_meas_update_local, update_simul_local)
	                setup_results_folder(results_folder)
	                run_rbpf_fireworks = []            
	                for run_idx in range(1, NUM_RUNS+1):
	                    for seq_idx in SEQUENCES_TO_PROCESS:
	                        cur_spec = {'det1_name': 'mscnn',
	                                'det2_name': det2_name,
	                                'num_particles': num_particles,
	                                'include_ignored_gt': False,
	                                'include_dontcare_in_gt': False,
	                                'sort_dets_on_intervals': True,
	                                'run_idx': run_idx,
	                                'seq_idx': seq_idx,
	                                'results_folder': results_folder,
	                                'CHECK_K_NEAREST_TARGETS': True,                        
	                                'K_NEAREST_TARGETS': 1,                        
	                                'RUN_ONLINE': RUN_ONLINE,
	                                'ONLINE_DELAY': 3,
	                                'MAX_1_MEAS_UPDATE': max_1_meas_update_local,                    
	                                'UPDATE_MULT_MEAS_SIMUL': update_simul_local,
	                                'TREAT_MEAS_INDEP': TREAT_MEAS_INDEP,                        
	                                'TREAT_MEAS_INDEP_2': TREAT_MEAS_INDEP_2,
	                                'USE_CONSTANT_R': USE_CONSTANT_R,
	                                'P': P_default.tolist(),
	                                'R': R_default.tolist(),
	                                'Q': Q_default.tolist(),
	                                'scale_prior_by_meas_orderings': scale_prior_by_meas_orderings}
	                        cur_firework = Firework(RunRBPF(), spec=cur_spec)
	        #                cur_firework = Firework(PyTask(func='rbpf.run_rbpf', auto_kwargs=False, kwargs=cur_spec))

	                        run_rbpf_fireworks.append(cur_firework)


	                seq_idx_to_eval = [i for i in range(21)]
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


    # store workflow and launch it locally
    workflow = Workflow(all_fireworks, firework_dependencies)
    launchpad.add_wf(workflow)
    qadapter = CommonAdapter.from_file("%sfireworks_files/my_qadapter.yaml" % RBPF_HOME_DIRECTORY)
    rapidfire(launchpad, FWorker(), qadapter, launch_dir='.', nlaunches='infinite', njobs_queue=81,
                  njobs_block=500, sleep_time=None, reserve=False, strm_lvl='INFO', timeout=None,
                  fill_mode=False)




