#Note, run the following on atlas before this script:
# $ export PATH=/opt/rh/python27/root/usr/bin:$PATH
# $ export LD_LIBRARY_PATH=/opt/rh/python27/root/usr/lib64/:$LD_LIBRARY_PATH
#
#May need to run $ kinit -r 30d
#
# Add the following line to the file ~/.bashrc.user on Atlas:
# export PYTHONPATH="/atlas/u/jkuck/rbpf_fireworks:$PYTHONPATH"

import os
import errno
import sys
from fireworks import Firework, Workflow, FWorker, LaunchPad
#from fireworks.core.rocket_launcher import rapidfire
from fireworks.queue.queue_launcher import rapidfire
from fireworks.user_objects.queue_adapters.common_adapter import CommonAdapter
from fw_tutorials.dynamic_wf.fibadd_task import FibonacciAdderTask
#from fireworks.core.firework import FWAction, Firework, FiretaskBase
from fireworks.user_objects.firetasks.script_task import PyTask

#from rbpf import RunRBPF
#from intermediate import RunRBPF
###################################### Experiment Parameters ######################################
NUM_RUNS=1
SEQUENCES_TO_PROCESS = [i for i in range(21)]
#SEQUENCES_TO_PROCESS = [0]
#SEQUENCES_TO_PROCESS = [11]
#SEQUENCES_TO_PROCESS = [13]
#NUM_PARTICLES_TO_TEST = [25, 100]
NUM_PARTICLES_TO_TEST = [100]


###################################### Experiment Organization ######################################
#DIRECTORY_OF_ALL_RESULTS = './ICML_prep_correctedOnline/propose_k=1_nearest_targets'
DIRECTORY_OF_ALL_RESULTS = '/atlas/u/jkuck/rbpf_fireworks/ICML_prep_debug2'
#CUR_EXPERIMENT_BATCH_NAME = 'genData_origRBPF_multMeas_probDet95_clutLambdaPoint1_noise05_noShuffle_beta1'
#CUR_EXPERIMENT_BATCH_NAME = 'genData_origRBPF_multMeas_fixedRounding_resampleRatio4_scaled_ShuffleMeas_timeScaled_PQdiv100'
#CUR_EXPERIMENT_BATCH_NAME = 'Rto0_4xQ_multMeas1update_online3frameDelay2'
CUR_EXPERIMENT_BATCH_NAME = 'CHECK_1_NEAREST_TARGETS/Rto0_4xQ_max1MeasUpdate_online3frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = 'CHECK_K_NEAREST_TARGETS=False/Reference/Rto0_4xQ_max1MeasUpdate_online3frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = '/Reference/Rto0_4xQ_max1MeasUpdate_online0frameDelay'
#CUR_EXPERIMENT_BATCH_NAME = 'measuredR_1xQ_max1MeasUpdate_online3frameDelay'

###################################### RBPF Parameters ######################################


def get_description_of_run(include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals,
                           use_regionlets, det1_name, det2_name):

    if det2_name == 'None' or det2_name == None:
        if (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and sort_dets_on_intervals:
            description_of_run = "%s_with_score_intervals" % (det1_name)
        elif (not include_ignored_gt) and (not include_dontcare_in_gt)\
            and (not sort_dets_on_intervals):
            description_of_run = "%s_no_score_intervals" % (det1_name)
        else:
            print "Unexpected combination of boolean arguments"
            print include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals, use_regionlets, use_mscnn
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
            print include_ignored_gt, include_dontcare_in_gt, sort_dets_on_intervals, use_regionlets, use_mscnn
            sys.exit(1);

def setup_results_folder(num_particles, include_ignored_gt, include_dontcare_in_gt, \
                         sort_dets_on_intervals, use_regionlets, det1_name, det2_name):
    description_of_run = get_description_of_run(include_ignored_gt, include_dontcare_in_gt, \
                                                sort_dets_on_intervals, use_regionlets, det1_name, det2_name)
    results_folder_name = '%s/%d_particles' % (description_of_run, num_particles)
    results_folder = '%s/%s/%s' % (DIRECTORY_OF_ALL_RESULTS, CUR_EXPERIMENT_BATCH_NAME, results_folder_name)

    for cur_run_idx in range(1, NUM_RUNS + 1):
        file_name = '%s/results_by_run/run_%d/%s.txt' % (results_folder, cur_run_idx, 'random_name')
        if not os.path.exists(os.path.dirname(file_name)):
            try:
                os.makedirs(os.path.dirname(file_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

def submit_single_experiment(use_regionlets, det1_name, det2_name, num_particles, include_ignored_gt=False, include_dontcare_in_gt=False, 
    sort_dets_on_intervals=True):
    setup_results_folder(num_particles, include_ignored_gt, include_dontcare_in_gt, \
                         sort_dets_on_intervals, use_regionlets, det1_name, det2_name)
    for run_idx in range(1, NUM_RUNS+1):
        for seq_idx in SEQUENCES_TO_PROCESS:
            submit_single_qsub_job(use_regionlets, det1_name, det2_name, num_particles=num_particles, include_ignored_gt=include_ignored_gt, 
                include_dontcare_in_gt=include_dontcare_in_gt,
                sort_dets_on_intervals=sort_dets_on_intervals, run_idx=run_idx, seq_idx=seq_idx)




if __name__ == "__main__":

    # set up the LaunchPad and reset it
    launchpad = LaunchPad(host='ds050869.mlab.com', port=50869, name='rbpf', username='jkuck', password='YH9uF643uLXJctBsCfau',
                     logdir=None, strm_lvl='INFO', user_indices=None, wf_user_indices=None, ssl_ca_file=None)
    launchpad.reset('', require_password=False)

    use_regionlets=False
    det1_name = 'mscnn'
    det2_name = 'regionlets'
    include_ignored_gt=False
    include_dontcare_in_gt=False
    sort_dets_on_intervals=True

    run_rbpf_fireworks = []
    for num_particles in NUM_PARTICLES_TO_TEST:
        setup_results_folder(num_particles, include_ignored_gt, include_dontcare_in_gt, \
                             sort_dets_on_intervals, use_regionlets, det1_name, det2_name)
        for run_idx in range(1, NUM_RUNS+1):
            for seq_idx in SEQUENCES_TO_PROCESS:
                cur_spec = {'use_regionlets': False,
                        'det1_name': 'mscnn',
                        'det2_name': 'regionlets',
                        'num_particles': num_particles,
                        'include_ignored_gt': False,
                        'include_dontcare_in_gt': False,
                        'sort_dets_on_intervals': True,
                        'run_idx': run_idx,
                        'seq_idx': seq_idx}
                # create the Firework consisting of a custom "Fibonacci" task
#                cur_firework = Firework(RunRBPF(), spec=cur_spec)
                cur_firework = Firework(PyTask(func='rbpf.run_rbpf', auto_kwargs=False, kwargs=cur_spec))

                run_rbpf_fireworks.append(cur_firework)

    # store workflow and launch it locally
    workflow = Workflow(run_rbpf_fireworks)
    launchpad.add_wf(workflow)
    qadapter = CommonAdapter.from_file("/atlas/u/jkuck/fireworks/fw_tutorials/queue/queue_tests/my_qadapter.yaml")
    rapidfire(launchpad, FWorker(), qadapter, launch_dir='.', nlaunches='infinite', njobs_queue=21,
                  njobs_block=500, sleep_time=None, reserve=False, strm_lvl='INFO', timeout=None,
                  fill_mode=False)





