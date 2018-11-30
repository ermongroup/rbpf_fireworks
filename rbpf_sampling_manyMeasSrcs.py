from __future__ import division
from scipy.stats import multivariate_normal
from scipy.stats import poisson
from scipy import spatial
import numpy as np
from numpy.linalg import inv
import numpy.linalg
import random
from sets import ImmutableSet
from collections import defaultdict
from itertools import combinations
from itertools import permutations
from operator import itemgetter
from pymatgen.optimization import linear_assignment
from munkres import Munkres
# import cvxpy as cvx
import sys
import math
from cluster_config import RBPF_HOME_DIRECTORY
sys.path.insert(0, "%sgeneral_tracking" % RBPF_HOME_DIRECTORY)
from global_params import INFEASIBLE_COST
from global_params import DEFAULT_TIME_STEP


#if we have prior of 0, return PRIOR_EPSILON
PRIOR_EPSILON = .000000001

class cvx:
    def __init__(self, x):
        self.x = x

class Parameters:
    def __init__(self, det_names, target_groupEmission_priors, clutter_grpCountByFrame_priors,\
                 clutter_group_priors, clutter_lambdas_by_group, birth_count_priors, birth_lambdas_by_group, posOnly_covariance_blocks, \
                 meas_noise_mean, posAndSize_inv_covariance_blocks, R_default, H,\
                 USE_PYTHON_GAUSSIAN, USE_CONSTANT_R, score_intervals,\
                 p_birth_likelihood, p_clutter_likelihood, CHECK_K_NEAREST_TARGETS,
                 K_NEAREST_TARGETS, scale_prior_by_meas_orderings, SPEC,
                 clutter_posAndSize_inv_covariance_blocks, clutter_posOnly_covariance_blocks, clutter_meas_noise_mean_posAndSize,\
                 fraction_of_group_type_that_is_clutter):
        '''
        Inputs:
        - det_names: list of detection source names
        -  score_intervals: list of lists, where score_intervals[i] is a list
            specifying the score intervals for measurement source i.  
            score_intervals[i][j] specifies the lower bound for the jth score
            interval corresponding to measurement source i (0 indexed).
        - CHECK_K_NEAREST_TARGETS: If true only possibly associate each measurement with
            one of its K_NEAREST_TARGETS.  If false measurements may be associated
            with any target.
        '''
        self.det_names = det_names

        #dictionary where target_groupEmission_priors[det_set] is the prior probability
        #that a ground truth object will emit the set of measurements specified by the immutable set det_set.
        self.target_groupEmission_priors = target_groupEmission_priors
        check_prior_emission_distr = 0.0
        for grp, prob in self.target_groupEmission_priors.iteritems():
            check_prior_emission_distr+=prob
        assert(np.isclose(check_prior_emission_distr, 1.0, rtol=1e-04, atol=1e-04))


        self.clutter_grpCountByFrame_priors = clutter_grpCountByFrame_priors
        self.clutter_group_priors = clutter_group_priors

        #dictionary where clutter_by_group_count[det_set] is the MLE estimate of the
        #    poisson parameter lambda for clutter emissions of this detection set.  This is used for a different
        #    clutter model than the above two.  Now we model clutter as occurring as indendent poisson distributions
        #    for every detection set.
        self.clutter_lambdas_by_group = clutter_lambdas_by_group

        self.birth_count_priors = birth_count_priors #dictionary, where birth_count_priors[n] is the prior probability of observing n births in a frame.

        #birth_lambdas_by_group: dictionary where birth_lambdas_by_group[det_set] is the MLE estimate of the
        #    poisson parameter lambda for births of this detection set.  This is used for a different
        #     model than birth_count_priors.  Now we model births as occurring as indendent poisson distributions
        #    for every detection set.
        self.birth_lambdas_by_group = birth_lambdas_by_group

        self.posOnly_covariance_blocks = posOnly_covariance_blocks #posOnly_covariance_blocks[(det_name1, det_name2)] = posOnly_cov_block_12

        self.meas_noise_mean = meas_noise_mean
        self.posAndSize_inv_covariance_blocks = posAndSize_inv_covariance_blocks


        self.clutter_posAndSize_inv_covariance_blocks = clutter_posAndSize_inv_covariance_blocks
        self.clutter_posOnly_covariance_blocks = clutter_posOnly_covariance_blocks
        self.clutter_meas_noise_mean_posAndSize = clutter_meas_noise_mean_posAndSize

        self.R_default = R_default
        self.H = H

        self.USE_PYTHON_GAUSSIAN = USE_PYTHON_GAUSSIAN
        self.USE_CONSTANT_R = USE_CONSTANT_R

        self.score_intervals = score_intervals

        self.p_birth_likelihood = p_birth_likelihood 
        self.p_clutter_likelihood = p_clutter_likelihood

        self.CHECK_K_NEAREST_TARGETS = CHECK_K_NEAREST_TARGETS
        self.K_NEAREST_TARGETS = K_NEAREST_TARGETS

        self.scale_prior_by_meas_orderings = scale_prior_by_meas_orderings

        self.fraction_of_group_type_that_is_clutter=fraction_of_group_type_that_is_clutter

        self.SPEC = SPEC

        #training_counts model means we count the number of frames we observe i births (or clutters)
        #and divide by the total number of frames to get the probability of i births.
        #poisson means we fit (MLE) this data to a poisson distribution
        assert(SPEC['birth_clutter_model'] in ['training_counts', 'poisson'])
        self.birth_clutter_model = SPEC['birth_clutter_model']
        if SPEC['birth_clutter_model'] == 'poisson':
            #Calculate maximum likelihood estimates of Poisson parameter for clutter and birth counts
            self.clutter_lambda = 0 #The expected number of clutter objects in a frame, also MLE of lambda for Poisson distribution
            for clutter_count, probability in self.clutter_grpCountByFrame_priors.iteritems():
                self.clutter_lambda += clutter_count*probability
            self.birth_lambda = 0 #The expected number of birth objects in a frame, also MLE of lambda for Poisson distribution
            for birth_count, probability in self.birth_count_priors.iteritems():
                self.birth_lambda += birth_count*probability

        print "posOnly_covariance_blocks"
        print posOnly_covariance_blocks
        #sleep(5)

        #print "target_groupEmission_priors: ", self.target_groupEmission_priors
        #print "clutter_grpCountByFrame_priors: ", self.clutter_grpCountByFrame_priors
        #print "clutter_group_priors: ", self.clutter_group_priors
        #print "birth_count_priors: ", self.birth_count_priors
        #print "posOnly_covariance_blocks: ", self.posOnly_covariance_blocks
        #print "meas_noise_mean: ", self.meas_noise_mean
        #print "posAndSize_inv_covariance_blocks: ", self.posAndSize_inv_covariance_blocks

    def get_all_possible_measurement_groups(self):
        '''
        Outputs: 
        - all_measurement_groups: list of ImmutableSet objects with length ((2^#measurement sources)-1).  All possible groups
            of measurements are enumerated in the list.
        '''
        all_measurement_groups = []
        for group_size in range(1, len(self.det_names) + 1):
            cur_det_combos = combinations(self.det_names, group_size)
            for det_combo in cur_det_combos:
                all_measurement_groups.append(ImmutableSet(det_combo))
        assert(len(all_measurement_groups) == 2**len(self.det_names) - 1)
        return all_measurement_groups

    def birth_groupCount_prior(self, group_count):
        if self.birth_clutter_model == "training_counts":
            if group_count in self.birth_count_priors:
                return self.birth_count_priors[group_count]\
                    *self.SPEC['coord_ascent_params']['birth_model_prior_const'][0]**group_count
            else:
                return PRIOR_EPSILON
        else:
            assert(self.birth_clutter_model == 'poisson')
            return poisson.pmf(mu = self.birth_lambda, k = group_count)
    def birth_group_prior(self, det_group):
        #The prior probability that a birth object will be observed by this specific detection group (det_group).
        #Birth objects are always observed and we assume target group emission priors are the same for birth
        #objects and general valid objects, so this is the probability that a target emits this det_group divided
        #by the probability that a target emits any detection group.
        if det_group in self.target_groupEmission_priors:
            returnVal = self.target_groupEmission_priors[det_group]/(1.0 - self.target_groupEmission_priors[ImmutableSet([])]) 
            assert (returnVal>0), (self.target_groupEmission_priors, det_group, self.target_groupEmission_priors[det_group], self.target_groupEmission_priors[ImmutableSet([])])
            return returnVal
        else:
            return PRIOR_EPSILON


    def clutter_groupCount_prior(self, group_count):
        if self.birth_clutter_model == "training_counts":        
            if group_count in self.clutter_grpCountByFrame_priors:
                return self.clutter_grpCountByFrame_priors[group_count]\
                    *self.SPEC['coord_ascent_params']['clutter_model_prior_const'][0]**group_count
            else:
                return PRIOR_EPSILON
        else:
            assert(self.birth_clutter_model == 'poisson')
            return poisson.pmf(mu = self.clutter_lambda, k = group_count)

    def clutter_group_prior(self, det_group):
        if det_group in self.clutter_group_priors:
            return self.clutter_group_priors[det_group]
        else:
            return PRIOR_EPSILON

    def find_clutter_priors_by_det():
        #Compute marginal_det_priors where marginal_det_priors[det_name][n] is the prior probability

        #Marginalize over self.clutter_group_priors to find marginal priors for each detection source

        #dictionary, key=det_name value=priors for det_name
        marginal_priors = {}

def l2_dist(a,b):
    x_dist = a[0] - b[0]
    y_dist = a[1] - b[1]
    distance = math.sqrt(x_dist**2 + y_dist**2)
    return distance

def boxoverlap(a,b,criterion="union"):
    """
        boxoverlap computes intersection over union for bbox a and b in KITTI format.
        If the criterion is 'union', overlap = (a inter b) / a union b).
        If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
        Inputs:
        - a: numpy array, [x_center, y_center, width, height] for detection a
        - b: numpy array, [x_center, y_center, width, height] for detection b
    """
    a_x1 = a[0]-a[2]/2
    a_x2 = a[0]+a[2]/2
    a_y1 = a[1]-a[3]/2
    a_y2 = a[1]+a[3]/2

    b_x1 = b[0]-b[2]/2
    b_x2 = b[0]+b[2]/2
    b_y1 = b[1]-b[3]/2
    b_y2 = b[1]+b[3]/2

    x1 = max(a_x1, b_x1)
    y1 = max(a_y1, b_y1)
    x2 = min(a_x2, b_x2)
    y2 = min(a_y2, b_y2)
    
    w = x2-x1
    h = y2-y1

    if w<=0. or h<=0.:
        return 0.
    inter = w*h
    aarea = (a_x2-a_x1) * (a_y2-a_y1)
    barea = (b_x2-b_x1) * (b_y2-b_y1)
    # intersection over union overlap
    if criterion.lower()=="union":
        o = inter / float(aarea+barea-inter)
    elif criterion.lower()=="a":
        o = float(inter) / float(aarea)
    else:
        raise TypeError("Unkown type for criterion")
    return o


def group_detections(meas_groups, det_name, detection_locations, det_widths, det_heights, params):
    """
    Take a list of detections and try to associate them with detection groups from other measurement sources
    Inputs:
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - det_name: name of the detection source we are currently associating with current detection groups
    - detections: a list of detections from a specific measurement source, sequence, and frame
    - seq_idx: the sequence index
    - frame_idx: the frame index (in the specified sequence)

    Outputs:
    None, but meas_groups will be modified, with the new detections added (passed by reference)
    """
    hm = Munkres()
    max_cost = 1e9

    # use hungarian method to associate, using boxoverlap 0..1 as cost
    # build cost matrix
    cost_matrix = []
    this_ids = [[],[]]

    assert(len(detection_locations) == len(det_widths) and len(det_widths) == len(det_heights))
    #combine into 4d detections
    detections = []
    for det_idx, det_loc in enumerate(detection_locations):
        detections.append(np.array([det_loc[0], det_loc[1], det_widths[det_idx], det_heights[det_idx]]))

    for cur_detection in detections:
        cost_row = []
        for cur_detection_group in meas_groups:
        #     min_cost = max_cost
        #     for grpd_det_name, grouped_detection in cur_detection_group.iteritems():
        #         # overlap == 1 is cost ==0
        #         c = 1-boxoverlap(cur_detection, grouped_detection)
        #         if c < min_cost:
        #             min_cost = c
        #     # gating for boxoverlap
        #     #if min_cost<=params.SPEC['coord_ascent_params'][det_name]:
        #     if min_cost<=params.SPEC['coord_ascent_params']['det_grouping_min_overlap_%s' % det_name][0]:
        #         cost_row.append(min_cost)
        #     else:
        #         cost_row.append(max_cost)
        # cost_matrix.append(cost_row)
            min_group_cost = max_cost
            max_group_cost = 0
            for grpd_det_name, grouped_detection in cur_detection_group.iteritems():
                # overlap == 1 is cost ==0
                c = 1-boxoverlap(cur_detection, grouped_detection)
                if c < min_group_cost:
                    min_group_cost = c
                if c > max_group_cost:
                    max_group_cost = c

            # gating for boxoverlap
            # if min_group_cost<=.5:
            #     cost_row.append(min_group_cost)
            if max_group_cost<=.5:
                cost_row.append(min_group_cost)
            else:
                cost_row.append(max_cost)
        cost_matrix.append(cost_row)        
    
    if len(detections) is 0:
        cost_matrix=[[]]
    # associate
    #switch to linear_assignment for speed, but need to deal with more rows than columns
    #lin_assign = linear_assignment.LinearAssignment(cost_matrix)
    #solution = lin_assign.solution
    #association_list = zip([i for i in range(len(solution))], solution)
    association_list = hm.compute(cost_matrix)

    associated_detection_indices = []
    check_det_count = 0
    for row,col in association_list:
        # apply gating on boxoverlap
        c = cost_matrix[row][col]
        if c < max_cost:
            associated_detection = detections[row]
            associated_detection_indices.append(row)
            associated_detection_group = meas_groups[col]

            #double check
            check_det_count += 1
            min_cost = max_cost
            for grpd_det_name, grouped_detection in associated_detection_group.iteritems():
                # overlap == 1 is cost ==0
                check_c = 1-boxoverlap(associated_detection, grouped_detection)
                if check_c < min_cost:
                    min_cost = check_c
            assert(min_cost == c), (min_cost, c)
            #done double check                

            associated_detection_group[det_name] = associated_detection                


    for det_idx in range(len(detections)):
        if not(det_idx in associated_detection_indices):
            meas_groups.append({det_name: detections[det_idx]})
            check_det_count += 1
    assert(check_det_count == len(detections))

 

def sample_and_reweight(particle, measurement_locations, widths, heights, det_names, \
    cur_time, measurement_scores, params):
    """
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle
    - measurement_locations: a list where measurement_locations[i] is a list of all measurement locations from the current
        time instance from the ith measurement source (i.e. different object detection algorithms
        or different sensors)
    - det_names: a list of names of measurement sources, where det_names[i] corresponds to measurement_locations[i]
    - measurement_scores: a list where measurement_scores[i] is a list containing scores for every measurement in
        measurement_list[i]
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - measurement_associations: A list where measurement_associations[i] is a list of association values
        for each measurements in measurement_locations[i].  Association values correspond to:
        measurement_associations[i][j] = -1 -> measurement is clutter
        measurement_associations[i][j] = particle.targets.living_count -> measurement is a new target
        measurement_associations[i][j] in range [0, particle.targets.living_count-1] -> measurement is of
            particle.targets.living_targets[measurement_associations[i][j]]

    - imprt_re_weight: After processing this measurement the particle's
        importance weight will be:
        new_importance_weight = old_importance_weight * imprt_re_weight
    - targets_to_kill: a list containing the indices of targets that should be killed, beginning
        with the smallest index in increasing order, e.g. [0, 4, 6, 33]

    - meas_grp_img_feats: list of 1d image feature arrays (of length 128) for each measurement

    - meas_assoc_gt_obj_ids: 
        - when SPEC['proposal_distr'] == 'ground_truth_assoc':
            list containing the track_id of the ground truth object each measurement is 
            associated with, or -1 if the measurement is clutter
        - when SPEC['proposal_distr'] != 'ground_truth_assoc': None

    """

    #get death probabilities for each target in a numpy array
    num_targs = particle.targets.living_count
    p_target_deaths = []
    for target in particle.targets.living_targets:
        p_target_deaths.append(target.death_prob)
        assert(p_target_deaths[len(p_target_deaths) - 1] >= 0 and p_target_deaths[len(p_target_deaths) - 1] <= 1)

    # a list containing the number of measurements detected by each source
    # used in prior calculation to count the number of ordered vectors given
    # an unordered association set
    meas_counts_by_source = [] 
    for meas_list in measurement_locations:
        meas_counts_by_source.append(len(meas_list))

    ASSOCIATE_MEASUREMENTS_TO_GT_BEFORE_GROUPING = True
    if params.SPEC['proposal_distr'] == 'ground_truth_assoc' and ASSOCIATE_MEASUREMENTS_TO_GT_BEFORE_GROUPING: #evaluate best possible ground truth associations
        frame_idx = int(round(cur_time/DEFAULT_TIME_STEP))
        seq_idx=params.SPEC['seq_idx']
        meas_groups_helper = defaultdict(list) #key: gt index, val:list of pairs (meas_src_name, detection)

        for meas_source_index, cur_src_measurements in enumerate(measurement_locations):
            meas_source_name = params.SPEC['det_names'][meas_source_index]
            det_gt_association_pairs = group_detectionSource_to_groundTruth(gt_objects=params.SPEC['gt_obects'][seq_idx][frame_idx],\
                detection_locations=cur_src_measurements, det_widths=widths[meas_source_index], det_heights=heights[meas_source_index])
            for det_idx, gt_idx in enumerate(det_gt_association_pairs):
                meas_groups_helper[gt_idx].append((meas_source_name, np.array([measurement_locations[meas_source_index][det_idx][0], measurement_locations[meas_source_index][det_idx][1], widths[meas_source_index][det_idx], heights[meas_source_index][det_idx]])))

        meas_groups = []
        for gt_idx, meas_list in meas_groups_helper.iteritems():
            cur_det_grp = {}
            for meas_source_name, measurement in meas_list:
                cur_det_grp[meas_source_name] = measurement
            meas_groups.append(cur_det_grp)
    else:
        meas_groups = []
        for det_idx, det_name in enumerate(det_names):
            group_detections(meas_groups, det_name, measurement_locations[det_idx], widths[det_idx], heights[det_idx], params)

    frame_idx = int(round(cur_time/DEFAULT_TIME_STEP))
    if params.SPEC['condition_emission_prior_img_feat']:
        #list of length len(meas_groups) containing image features for each measurement group
        meas_grp_img_feats = []
        for grp_idx, meas_grp in enumerate(meas_groups):
            for det_name in params.SPEC['det_names']:
                if det_name in meas_grp:
                    det = meas_grp[det_name]
                    debug_info = {'seq_idx': params.SPEC['seq_idx'], 'det_name': det_name}
                    meas_grp_img_feats.append(get_det_features(params.SPEC['seq_det_feature_arrays'][(det_name, params.SPEC['seq_idx'])], det, frame_idx, debug_info))
                    break
    else:
        meas_grp_img_feats = None

    meas_assoc_gt_obj_ids = None
    (targets_to_kill, meas_grp_associations, meas_grp_means, meas_grp_covs, proposal_probability, meas_assoc_gt_obj_ids, sampled_meas_targ_assoc_idx) =  \
        sample_grouped_meas_assoc_and_death(particle, meas_groups, particle.targets.living_count, \
        p_target_deaths, cur_time, measurement_scores, params, meas_counts_by_source=meas_counts_by_source,\
        measurement_locations=measurement_locations, widths=widths, heights=heights)



    living_target_indices = []
    unassociated_target_indices = []
    for i in range(particle.targets.living_count):
        if(not i in targets_to_kill):
            living_target_indices.append(i)
        if(not i in meas_grp_associations):
            unassociated_target_indices.append(i)


    assert(params.SPEC['use_log_probs'] in ['True', 'False', 'Compare'])

    if params.SPEC['use_log_probs'] in ['False', 'Compare']:
        likelihood = get_likelihood(particle, meas_groups, particle.targets.living_count,
                                       meas_grp_associations, params, log=False)
        assoc_prior = get_assoc_prior(particle, meas_groups, meas_grp_associations, params, meas_counts_by_source, log=False, cur_time=cur_time)
        death_prior = calc_death_prior(living_target_indices, p_target_deaths, unassociated_target_indices, log=False)
        exact_probability = likelihood * assoc_prior * death_prior

        if params.SPEC['normalize_log_importance_weights']:
            imprt_re_weight = ln_zero_approx(exact_probability/proposal_probability)
        else:
            imprt_re_weight = exact_probability/proposal_probability


    if params.SPEC['use_log_probs'] in ['True', 'Compare']:        
        log_likelihood = get_likelihood(particle, meas_groups, particle.targets.living_count,
                                       meas_grp_associations, params, log=True)
        log_assoc_prior = get_assoc_prior(particle, meas_groups, meas_grp_associations, params, meas_counts_by_source, log=True, cur_time=cur_time)
        if params.SPEC['proposal_distr'] != 'ground_truth_assoc':
            log_death_prior = calc_death_prior(living_target_indices, p_target_deaths, unassociated_target_indices, log=True)
        else:
            log_death_prior = 0.0
        log_exact_probability = log_likelihood + log_assoc_prior + log_death_prior

        if params.SPEC['normalize_log_importance_weights']:
            imprt_re_weight = log_exact_probability - ln_zero_approx(proposal_probability)
        else:
            imprt_re_weight = math.exp(log_exact_probability - ln_zero_approx(proposal_probability))    


    if params.SPEC['use_log_probs'] == 'Compare':
        imprt_re_weightA = exact_probability/proposal_probability
        imprt_re_weightB = math.exp(log_exact_probability - ln_zero_approx(proposal_probability))    
        assert(np.abs(imprt_re_weightA -imprt_re_weightB) < .000001), (imprt_re_weightA, imprt_re_weightB, exact_probability, log_exact_probability, proposal_probability)

    assert(num_targs == particle.targets.living_count)
    #double check targets_to_kill is sorted
    assert(all([targets_to_kill[i] <= targets_to_kill[i+1] for i in xrange(len(targets_to_kill)-1)]))

    if not params.SPEC['normalize_log_importance_weights']:
        assert(imprt_re_weight != 0.0), (exact_probability, proposal_probability, death_prior)

#    particle.likelihood_DOUBLE_CHECK_ME = exact_probability

#    print "imprt_re_weight:", imprt_re_weight
    return (meas_grp_associations, meas_grp_means, meas_grp_covs, meas_grp_img_feats, targets_to_kill, imprt_re_weight, log_exact_probability, proposal_probability, meas_assoc_gt_obj_ids, sampled_meas_targ_assoc_idx)

def sample_grouped_meas_assoc_and_death(particle, meas_groups, total_target_count, 
    p_target_deaths, cur_time, measurement_scores, params, meas_counts_by_source=None, 
    measurement_locations=None, widths=None, heights=None):
    """
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - measurement_scores: type list, measurement_scores[i] is a list containing scores for every measurement in
        measurement_list[i]
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using
    - measurement_locations: a list where measurement_locations[i] is a list of all measurement locations from the current
        time instance from the ith measurement source (i.e. different object detection algorithms
        or different sensors)
    - measurement_widths: a list where measurement_widths[i] is a list of all measurement widths from the current
        time instance from the ith measurement source (i.e. different object detection algorithms
        or different sensors)
    - measurement_heights: a list where measurement_heights[i] is a list of all measurement locations from the current
        time instance from the ith measurement source (i.e. different object detection algorithms
        or different sensors)

    Output:
    - targets_to_kill: a list of targets that have been sampled to die (not killed yet)
    - meas_grp_associations: type list, meas_grp_associations[i] the association for the ith
        measurement group
    - meas_grp_means: list, each element is the combined measurment mean (np array)
    - meas_grp_covs: list, each element is the combined measurment covariance (np array)
    - proposal_probability: proposal probability of the sampled deaths and associations
    - meas_counts_by_source: just used by associate_meas_optimal, probably not
    - meas_assoc_gt_obj_ids: 
        - when SPEC['proposal_distr'] == 'ground_truth_assoc':
            list containing the track_id of the ground truth object each measurement is 
            associated with, or -1 if the measurement is clutter
        - when SPEC['proposal_distr'] != 'ground_truth_assoc': None

    """
#    assert(len(measurement_locations) == len(measurement_scores))
#    measurement_associations = []
#    proposal_probability = 1.0
#    for meas_source_index in range(len(measurement_locations)):
#        (cur_associations, cur_proposal_prob) = associate_measurements_sequentially\
#            (particle, meas_source_index, measurement_locations[meas_source_index], \
#             total_target_count, p_target_deaths, measurement_scores[meas_source_index],\
#             params)
#        measurement_associations.append(cur_associations)
#        proposal_probability *= cur_proposal_prob
#
#    assert(len(measurement_associations) == len(measurement_locations))
#
#    FIXME measurement_associations, proposal_probability
############################################################################################################
    #New implementation
    meas_assoc_gt_obj_ids = None
    assert(params.SPEC['proposal_distr'] in ['sequential', 'min_cost', 'min_cost_corrected', 'optimal', 'traditional_SIR_gumbel', 'ground_truth_assoc'])
    if params.SPEC['proposal_distr'] == 'traditional_SIR_gumbel':
        (meas_grp_associations, meas_grp_means, meas_grp_covs, proposal_probability, targets_to_kill) = \
            associate_meas_gumbel_exact(particle, meas_groups, total_target_count, p_target_deaths, params,\
            meas_counts_by_source, cur_time=cur_time)
#            associate_meas_gumbel_exact(particle, meas_groups, total_target_count, p_target_deaths, params, cur_time=cur_time)        

        unassociated_target_death_probs = []
        for i in range(total_target_count):
            if i in meas_grp_associations:
                target_unassociated = False
            else:
                target_unassociated = True            
            if target_unassociated:
                unassociated_target_death_probs.append(p_target_deaths[i])
            else:
                unassociated_target_death_probs.append(0.0)

        return (targets_to_kill, meas_grp_associations, meas_grp_means, meas_grp_covs, proposal_probability, unassociated_target_death_probs)

    else:
        if params.SPEC['proposal_distr'] == 'sequential':
            (meas_grp_associations, meas_grp_means, meas_grp_covs, proposal_probability) = \
            associate_measurements_sequentially(particle, meas_groups, total_target_count, \
            p_target_deaths, params)
            sampled_meas_targ_assoc_idx = -1

        elif params.SPEC['proposal_distr'] == 'ground_truth_assoc':
            (meas_grp_associations, meas_grp_means, meas_grp_covs, proposal_probability, targets_to_kill, meas_assoc_gt_obj_ids) = \
            get_gt_assoc(particle, meas_groups, cur_time, total_target_count, params, measurement_locations=measurement_locations, widths=widths, heights=heights)
        elif params.SPEC['proposal_distr'] == 'min_cost':
            (meas_grp_associations, meas_grp_means, meas_grp_covs, proposal_probability) = \
            associate_meas_min_cost(particle, meas_groups, total_target_count, \
            p_target_deaths, params, cur_time=cur_time, meas_counts_by_source=meas_counts_by_source)

        elif params.SPEC['proposal_distr'] == 'min_cost_corrected':
            (meas_grp_associations, meas_grp_means, meas_grp_covs, proposal_probability, sampled_meas_targ_assoc_idx) = \
            associate_meas_min_cost_corrected(particle, meas_groups, total_target_count, \
            p_target_deaths, params)        

        else: 
            assert(params.SPEC['proposal_distr'] == 'optimal')
            (meas_grp_associations, meas_grp_means, meas_grp_covs, proposal_probability) = \
            associate_meas_optimal(particle, meas_groups, total_target_count, \
            p_target_deaths, params, meas_counts_by_source, cur_time=cur_time)


    ############################################################################################################

    ############################################################################################################
        if params.SPEC['proposal_distr'] != 'ground_truth_assoc':
            #sample target deaths from unassociated targets
            unassociated_targets = []
            unassociated_target_death_probs = []

            for i in range(total_target_count):
                if i in meas_grp_associations:
                    target_unassociated = False
                else:
                    target_unassociated = True            
                if target_unassociated:
                    unassociated_targets.append(i)
                    unassociated_target_death_probs.append(p_target_deaths[i])
                else:
                    unassociated_target_death_probs.append(0.0)

#            (targets_to_kill, death_probability) =  \
#                sample_target_deaths(particle, unassociated_targets, cur_time)
            ####QUICK TEST KILLING ALL UNASSOCIATED TARGETS########
            (targets_to_kill, death_probability) = (unassociated_targets, 1.0)

            #probability of sampling all associations
            proposal_probability *= death_probability
            assert(proposal_probability != 0.0)

        #debug
        for i in range(total_target_count):
            assert(meas_grp_associations.count(i) == 0 or \
                   meas_grp_associations.count(i) == 1), (meas_grp_associations,  measurement_list, total_target_count, p_target_deaths)
        #done debug

        return (targets_to_kill, meas_grp_associations, meas_grp_means, meas_grp_covs, proposal_probability, meas_assoc_gt_obj_ids, sampled_meas_targ_assoc_idx)



def min_cost_measGrp_target_assoc(meas_grp_means4D, target_pos4D, params, max_assoc_cost):
    """
    Take a list of detections and try to associate them with detection groups from other measurement sources
    Inputs:
    - meas_grp_means4D: list of numpy arrays of combined measurment group x,y,width,height
    - target_pos4D: list of numpy arrays of target positions x,y,width,height

    Outputs:
    - measurement_assoc: list of length=len(meas_grp_means4D).  measurement_assoc[i] = j means
        that the ith measurement group is associated with the jth target.  -1 means the ith
        measurement is not associated with any living target.
    """

    hm = Munkres()
    max_cost = 1e9

    # use hungarian method to associate, using boxoverlap 0..1 as cost
    # build cost matrix
    cost_matrix = []
    this_ids = [[],[]]

    for cur_detection in meas_grp_means4D:
        cost_row = []
        for cur_target in target_pos4D:
            if params.SPEC['targ_meas_assoc_metric'] == 'box_overlap':
                c = 1-boxoverlap(cur_detection, cur_target)
            else:
                assert(params.SPEC['targ_meas_assoc_metric'] == 'distance')
                c = l2_dist(cur_detection, cur_target)
            # gating for boxoverlap
            if c<=max_assoc_cost:
                cost_row.append(c)
            else:
                cost_row.append(max_cost)
        cost_matrix.append(cost_row)
    
    if len(meas_grp_means4D) is 0:
        cost_matrix=[[]]
    # associate
    #lin_assign = linear_assignment.LinearAssignment(cost_matrix)
    #solution = lin_assign.solution
    #association_list = zip([i for i in range(len(solution))], solution)
    association_list = hm.compute(cost_matrix)
    
    measurement_assoc = [-1 for i in range(len(meas_grp_means4D))]
    for row,col in association_list:
        # apply gating on boxoverlap
        c = cost_matrix[row][col]
        if c < max_cost:
            associated_measGrp = meas_grp_means4D[row]
            associated_target = target_pos4D[col]
            measurement_assoc[row] = col

            if params.SPEC['targ_meas_assoc_metric'] == 'box_overlap':
                assert(c == 1-boxoverlap(associated_measGrp, associated_target))
            else:
                assert(np.abs(c - l2_dist(associated_measGrp, associated_target)) < .000001), (c, l2_dist(cur_detection, cur_target))


    return measurement_assoc

def associate_meas_optimal(particle, meas_groups, total_target_count, p_target_deaths, params, meas_counts_by_source, cur_time):
    '''
    Sample measurement associations from the optimal proposal distribution p(c_k | e_{1-k-1}, c_{1:k-1}, y_{1:k}).
    Generally computationally intractable.
    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    '''
    #list of all possible measurement associations.  each entry is a list of length (#measurment groups)
    #where each entry is -1 (clutter association), total_target_target(birth association), or 
    #[0, total_target_count-1] (association with the indicated target)
    all_possible_measurement_associations = []

    #list of proposal probabilities for each association in all_possible_measurement_associations
    #proposal_probabilities[i] will be the proposal probability for association all_possible_measurement_associations[i]
    proposal_probabilities = []

    #target association count; we can association between and all targets with a measurement
    for t_a_c in range(total_target_count + 1): 
        #birth count; we can have between 0 and measurement_count - t_a_c births
        for b_c in range(len(meas_groups)-t_a_c+1): 
            #clutter count; the remaining measurements must all be clutter
            c_c = len(meas_groups) - t_a_c - b_c
            #create a list [0,1,2,...,#measurements - 1]
            measurement_indices = range(len(meas_groups))
            #enumerate all possible groups of measurements that can be associated
            #with targets.  There will be (#measurements choose t_a_c) of these
            for t_a_meas_indices in combinations(measurement_indices, t_a_c):
                #create a list of measurement indices that are NOT associated with a target
                remaining_meas_indices = [idx for idx in measurement_indices if not idx in t_a_meas_indices]
                assert(len(remaining_meas_indices)+len(t_a_meas_indices) == len(measurement_indices))
                #enumerate all possible groups of measurements that can be associated
                #with births.  There will be ((#measurements - t_a_c) choose b_c) of these
                for b_meas_indices in combinations(remaining_meas_indices, b_c):
                    #create a list of remaining measurement indices that must be associated with clutter
                    c_meas_indices = [idx for idx in remaining_meas_indices if not idx in b_meas_indices]
                    assert(len(c_meas_indices)+len(b_meas_indices) == len(remaining_meas_indices))
                    #now enumerate every permutation of target indices that can be associated
                    #with the combination of measurement indices in t_a_meas_indices
                    for t_a_target_indices in permutations(range(total_target_count), t_a_c):
                        #now create the current measurement association
                        cur_meas_association = [-99 for i in range(len(measurement_indices))]
                        #iterate over MEASUREMENTs that are associated with SOME target
                        #idx: the current measurement's index in the tuple of only those measurements
                        #that are associated with targets.
                        #t_a_meas_idx: the current measurement's index in the list of all measurements
                        for (idx, t_a_meas_idx) in enumerate(t_a_meas_indices):
                            cur_associated_target_idx = t_a_target_indices[idx]
                            cur_meas_association[t_a_meas_idx] = cur_associated_target_idx
                        #set birth measurement associations
                        for b_meas_idx in b_meas_indices:
                            cur_meas_association[b_meas_idx] = total_target_count
                        #set clutter measurement associations           
                        for c_meas_idx in c_meas_indices:
                            cur_meas_association[c_meas_idx] = -1
                        #check we assigned each measurement an association
                        for association in cur_meas_association:
                            assert(association != -99)
                            assert(association >= -1 and association <= total_target_count)
                        assert(not cur_meas_association in all_possible_measurement_associations)
                        all_possible_measurement_associations.append(cur_meas_association)    

                        cur_likelihood = get_likelihood(particle, meas_groups, particle.targets.living_count,
                                                       cur_meas_association, params, log=False)
                        cur_assoc_prior = get_assoc_prior(particle, meas_groups, cur_meas_association, params, meas_counts_by_source, log=False, cur_time=cur_time)
                        cur_proposal_probability = cur_likelihood * cur_assoc_prior 
                        proposal_probabilities.append(cur_proposal_probability)
 
    #normalize proposal probabilities and sample association
    proposal_distribution = np.asarray(proposal_probabilities)
    partition_val = float(np.sum(proposal_distribution))
    assert(np.sum(proposal_distribution) != 0.0)
    proposal_distribution /= partition_val
    sampled_assoc_idx = np.random.choice(len(proposal_distribution),
                                            p=proposal_distribution)

############ THIS DOESN"T REALLY BELONG HERE, BUT FOLLOWING RETURN VALUES FOR OTHER PROPOSAL DISTRIBUTIONS ############
    #list of detection group centers, meas_grp_means[i] is a 2-d numpy array
    #of the position of meas_groups[i]
    meas_grp_covs = []   
    meas_grp_means2D = []
    meas_grp_means4D = []
    for (index, detection_group) in enumerate(meas_groups):
        (combined_meas_mean, combined_covariance) = combine_arbitrary_number_measurements_4d(params.posAndSize_inv_covariance_blocks, 
                            params.meas_noise_mean, detection_group)
        combined_meas_pos = combined_meas_mean[0:2]
        meas_grp_means2D.append(combined_meas_pos)
        meas_grp_means4D.append(combined_meas_mean)
        meas_grp_covs.append(combined_covariance)
############ END THIS DOESN"T REALLY BELONG HERE, BUT FOLLOWING RETURN VALUES FOR OTHER PROPOSAL DISTRIBUTIONS ############

#UNCOMMENT ME WHEN DONE DEBUGGING GUMBEL TO WORK WITH OTHER STUFF
    return(all_possible_measurement_associations[sampled_assoc_idx], meas_grp_means4D,
        meas_grp_covs, proposal_distribution[sampled_assoc_idx])


######    ######### DEBUGGING GUMBEL ###############
######    proposal_distr_dict = {}
######    for idx, assoc in enumerate(all_possible_measurement_associations):
######        proposal_distr_dict[tuple(assoc)] = proposal_distribution[idx]
######
######    return(all_possible_measurement_associations[sampled_assoc_idx], meas_grp_means4D,
######        meas_grp_covs, proposal_distribution[sampled_assoc_idx], partition_val, proposal_distribution, proposal_distr_dict) 
######    ######### DONE DEBUGGING GUMBEL ###############

def solve_gumbel_perturbed_assignment(log_probs, ubc_count):
    '''
    We would like to sample from the optimal proposal distribution p(x_k|y_1:k, x_1:k-1),
    the normalization constant is intractable to compute.  We can compute p(y_k, x_k|y_1:k-1, x_1:k-1),
    which is proporitional to p(x_k|y_1:k, x_1:k-1).  Max(p(y_k, x_k|y_1:k-1, x_1:k-1)) can be
    found by solving a linear program given a matrix log probabilities.  We approximate adding gumbel 
    noise to every possible assignment(which would require a combinatorial number of Gumbels), 
    by adding a Gumbel to each entry of the log_probs matrix.  Instead of taking the maximum of 
    each of these Gumbels (which would make the problem nonconvex) we further approximate by summing them.

    Inputs:
    - log_probs: numpy matrix with dimensions (#measurements + 2, #targets + 2)
    - ubc_count: integer, unobserved target+birth+clutter count is constrained to equal ubc_count
    Outpus:
    - assigment: numpy matrix with same dimensions as log_probs, this is the assignment that results
        the largest log probability after perturbing log_probs with Gumbel noise

    - max_log_prob: this is the value of trace((log_probs+Gumbel_Noise)*assignment.T)

    '''
    #sample a Gumbel matrix
    G = numpy.random.gumbel(loc=0.0, scale=1.0, size=(log_probs.shape[0], log_probs.shape[1]))

    M = log_probs.shape[0] - 2 #number of measurements
    T = log_probs.shape[1] - 2 #number of targets
    assert((ubc_count - np.abs(M-T)) % 2 == 0)
    #the number of 1's in the assignment matrix (measurement target assignments + births +
    #clutters + unobserved targets)
    number_of_assignments = min(M, T) - (ubc_count - np.abs(M-T))/2 + ubc_count

    #solve convex optimization problem
    A = cvx.Variable(log_probs.shape[0], log_probs.shape[1])
    objective = cvx.Maximize(cvx.trace((log_probs)*(A.T)) + cvx.trace(G*(A.T))/number_of_assignments*5) #SUPER HACKYISH
    constraints = [A>=0]                   
    for i in range(log_probs.shape[0]-2):
        constraints.append(cvx.sum_entries(A[i, :]) == 1)
    for j in range(log_probs.shape[1]-2):
        constraints.append(cvx.sum_entries(A[:, j]) == 1)
    constraints.append(A[(log_probs.shape[0]-2,log_probs.shape[1]-2)] == 0)
    constraints.append(A[(log_probs.shape[0]-1,log_probs.shape[1]-2)] == 0)
    constraints.append(A[(log_probs.shape[0]-2,log_probs.shape[1]-1)] == 0)
    constraints.append(A[(log_probs.shape[0]-1,log_probs.shape[1]-1)] == 0)

    constraints.append(cvx.sum_entries(A[log_probs.shape[0]-2, :])
                     + cvx.sum_entries(A[log_probs.shape[0]-1, :])
                     + cvx.sum_entries(A[:, log_probs.shape[1]-2])
                     + cvx.sum_entries(A[:, log_probs.shape[1]-1]) == ubc_count)

    prob = cvx.Problem(objective, constraints)
    prob.solve()
    assignment = A.value
    max_log_prob = prob.value
    assert(np.isclose(np.sum(assignment), number_of_assignments, rtol=1e-04, atol=1e-04))
    return(assignment, max_log_prob)


def associate_meas_gumbel(particle, meas_groups, total_target_count, p_target_deaths, params):
    '''
    Sample measurement associations from close to the optimal proposal distribution p(c_k | e_{1-k-1}, c_{1:k-1}, y_{1:k})
    using an approximation to the Gumbel max trick

    ONLY WORKS WITH 1 measurement source
    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    '''
    log_probs = construct_log_probs_matrix(particle, meas_groups, total_target_count, p_target_deaths, params)

    #solve a perturbed assignment problem where the number of unobserved targets, clutter measurements
    #and birth measurements must sum to ubc_count for every possible value that ubc_count can take
    assignment = None
    max_log_prob = None
    for ubc_count in range(np.abs(len(meas_groups) - total_target_count), len(meas_groups) + total_target_count + 1, 2):
        (cur_assignment, cur_max_log_prob) = solve_gumbel_perturbed_assignment(log_probs, ubc_count)
        if max_log_prob == None or cur_max_log_prob > max_log_prob:
            assignment = cur_assignment
            max_log_prob = cur_max_log_prob

    unnormalized_log_proposal_probability = np.trace(np.dot(log_probs, assignment.T))
    unnormalized_proposal_probability = np.exp(unnormalized_log_proposal_probability)

    (meas_associations, dead_target_indices) = convert_assignment_matrix(assignment)

    #Now approximate the partition function
    partition_estimate = 0.0
    for i in range(params.SPEC['num_gumbel_partition_samples']):
        #solve a perturbed assignment problem where the number of unobserved targets, clutter measurements
        #and birth measurements must sum to ubc_count for every possible value that ubc_count can take
        assignment = None
        max_log_prob = None
        for ubc_count in range(np.abs(len(meas_groups) - total_target_count), len(meas_groups) + total_target_count + 1, 2):
            (cur_assignment, cur_max_log_prob) = solve_gumbel_perturbed_assignment(log_probs, ubc_count)
            if max_log_prob == None or cur_max_log_prob > max_log_prob:
                assignment = cur_assignment
                max_log_prob = cur_max_log_prob

#        partition_estimate += max_log_prob - np.euler_gamma*np.sum(assignment)
        partition_estimate += max_log_prob - np.euler_gamma #think about this line some more!!
    partition_estimate = partition_estimate/params.SPEC['num_gumbel_partition_samples'] #now we have log(Z) estimated
    partition_estimate = np.exp(partition_estimate)

    proposal_probability = unnormalized_proposal_probability/partition_estimate
############ THIS DOESN"T REALLY BELONG HERE, BUT FOLLOWING RETURN VALUES FOR OTHER PROPOSAL DISTRIBUTIONS ############
    #list of detection group centers, meas_grp_means[i] is a 2-d numpy array
    #of the position of meas_groups[i]
    meas_grp_covs = []   
    meas_grp_means2D = []
    meas_grp_means4D = []
    for (index, detection_group) in enumerate(meas_groups):
        (combined_meas_mean, combined_covariance) = combine_arbitrary_number_measurements_4d(params.posAndSize_inv_covariance_blocks, 
                            params.meas_noise_mean, detection_group)
        combined_meas_pos = combined_meas_mean[0:2]
        meas_grp_means2D.append(combined_meas_pos)
        meas_grp_means4D.append(combined_meas_mean)
        meas_grp_covs.append(combined_covariance)
############ END THIS DOESN"T REALLY BELONG HERE, BUT FOLLOWING RETURN VALUES FOR OTHER PROPOSAL DISTRIBUTIONS ############


    return(meas_associations, meas_grp_means4D, meas_grp_covs, proposal_probability, dead_target_indices)

def construct_log_probs_matrix(particle, meas_groups, total_target_count, p_target_deaths, params):
    '''
    ONLY WORKS WITH 1 measurement source
    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle         
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Outputs:
    - log_probs: numpy matrix with dimensions (#measurements+2)x(#targets+2) of log probabilities.
        np.trace(np.dot(log_probs,A.T) will be the log probability of an assignment A, given our
        Inputs.  (Where an assignment defines measurement associations to targets, birth or clutter
        and unassociated target life/death)
    '''
    #construct a (#measurements+2)x(#targets+2) matrix of log probabilities
    log_probs = -1*np.ones((len(meas_groups) + 2, total_target_count+2))

    p_target_does_not_emit = params.target_groupEmission_priors[ImmutableSet([])]
    #ONLY WORKS WITH 1 measurement source    
    p_target_emits = 1.0 - p_target_does_not_emit

    #calculate log probs for measurement-target association entries in the log-prob matrix
    for m_idx in range(len(meas_groups)):
        for t_idx in range(total_target_count):
            likelihood = memoized_assoc_likelihood(particle, meas_groups[m_idx], t_idx, params)
            assert(likelihood >= 0.0), likelihood
            if likelihood > 0.0:
                cur_prob = ln_zero_approx(likelihood)
            else:
                cur_prob = -999 #(np.exp(-999) == 0) evaluates to True
            cur_prob += ln_zero_approx(p_target_emits) 
            log_probs[m_idx][t_idx] = cur_prob

    #calculate log probs for target doesn't emit and lives/dies entries in the log-prob matrix
    lives_row_idx = len(meas_groups)
    dies_row_idx = len(meas_groups) + 1
    for t_idx in range(total_target_count):
        #would probably be better to kill offscreen targets before association
        if(particle.targets.living_targets[t_idx].offscreen == True):
            cur_death_prob = .999999999999 #sloppy should define an epsilon or something
        else:
            cur_death_prob = particle.targets.living_targets[t_idx].death_prob
        log_probs[lives_row_idx][t_idx] = ln_zero_approx(p_target_does_not_emit) + ln_zero_approx(1.0 - cur_death_prob)
        log_probs[dies_row_idx][t_idx] = ln_zero_approx(p_target_does_not_emit) + ln_zero_approx(cur_death_prob)

    #add birth/clutter measurement association entries to the log-prob matrix
    clutter_col = total_target_count
    birth_col = total_target_count + 1


    assert(params.SPEC['birth_clutter_likelihood'] == 'aprox1')
    for m_idx in range(len(meas_groups)):
        log_probs[m_idx][clutter_col] = ln_zero_approx(params.clutter_lambda) + \
            ln_zero_approx(birth_clutter_likelihood(meas_groups[m_idx], params, 'clutter')*params.p_clutter_likelihood)
        log_probs[m_idx][birth_col] = ln_zero_approx(params.birth_lambda) + \
            ln_zero_approx(birth_clutter_likelihood(meas_groups[m_idx], params, 'birth')*params.p_birth_likelihood)
   
    return log_probs


def convert_assignment_matrix(assignment_matrix):
    '''
    use along with construct_log_probs_matrix()
    Inputs:
    - assignment_matrix: numpy arrays with dimensions (measurement_count+2, target_count+2).
        representing a state of measurement associations and whether each unobserved target
        is alive or dead.

    Outputs:
    - meas_associations: list of integers, the measurement associations represented by assignment_matrix.
        where meas_associations[j] is an integer representing the association of the jth measurement with:
            clutter = -1
            target association = [0,target_count-1]
            birth = target_count
    - dead_target_indices: list of integers, of length #dead targets.  if target_idx = i in [0,target_count-1]
        is in the list dead_target_indices, target i died.
    
    '''
    meas_associations = []
    measurement_count = assignment_matrix.shape[0] - 2
    target_count = assignment_matrix.shape[1] - 2
    dies_row_idx = measurement_count + 1

    #read off assignments
    for m_idx in range(measurement_count):
        for assign_idx in range(target_count+2):
            if (np.isclose(assignment_matrix[m_idx,assign_idx], 1, rtol=5e-02, atol=5e-02)):
                if assign_idx < target_count: #target association
                    meas_associations.append(assign_idx)
                elif assign_idx == target_count: #clutter
                    meas_associations.append(-1)
                else: #birth
                    meas_associations.append(target_count)
    assert(len(meas_associations) == measurement_count), (assignment_matrix, meas_associations)

    #read off target deaths
    dead_target_indices = []
    for target_idx in range(target_count):
        if (np.isclose(assignment_matrix[dies_row_idx,target_idx], 1, rtol=1e-04, atol=1e-04)):
            dead_target_indices.append(target_idx)    

    return(meas_associations, dead_target_indices)

def construct_log_probs_matrix2(particle, meas_groups, total_target_count, p_target_deaths, params):
    '''
    M = #measurements
    T = #targets

    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle         
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Outputs:
    - log_probs: numpy matrix with dimensions (M+2T)x(T+2M) of log probabilities.
        np.trace(np.dot(log_probs,A.T) will be the log probability of an assignment A, given our
        Inputs.  (Where an assignment defines measurement associations to targets, and is marginalized
    '''
    #construct a (M+2T)x(T+2M) matrix of log probabilities
    M = len(meas_groups)
    T = total_target_count
    log_probs = -1*np.ones((M + 2*T, T + 2*M))
    log_probs *= 99999999999 #setting all entries to very negative value

    p_target_does_not_emit = params.target_groupEmission_priors[ImmutableSet([])]

    print 'log_probs1:'
    print log_probs

    #calculate log probs for target association entries in the log-prob matrix

    for t_idx in range(T):
        for m_idx in range(M):
        #calculate log probs for measurement-target association entries in the log-prob matrix
            likelihood = memoized_assoc_likelihood(particle, meas_groups[m_idx], t_idx, params)
            assert(likelihood >= 0.0), likelihood
            if likelihood > 0.0:
                cur_prob = ln_zero_approx(likelihood)
            else:
                cur_prob = -999 #(np.exp(-999) == 0) evaluates to True
            cur_prob += ln_zero_approx(params.target_groupEmission_priors[get_immutable_set_meas_names(meas_groups[m_idx])]) 
            log_probs[m_idx][t_idx] = cur_prob

        #calculate log probs for target doesn't emit and lives/dies entries in the log-prob matrix
        lives_row_idx = M + 2*t_idx
        dies_row_idx = M + 1 + 2*t_idx
        #would probably be better to kill offscreen targets before association
        if(particle.targets.living_targets[t_idx].offscreen == True):
            cur_death_prob = .999999999999 #sloppy should define an epsilon or something
        else:
            cur_death_prob = particle.targets.living_targets[t_idx].death_prob
        if cur_death_prob == 1.0:
            log_probs[lives_row_idx][t_idx] = -999
        else:
            log_probs[lives_row_idx][t_idx] = ln_zero_approx(p_target_does_not_emit) + ln_zero_approx(1.0 - cur_death_prob)
            log_probs[dies_row_idx][t_idx] = ln_zero_approx(p_target_does_not_emit) + ln_zero_approx(cur_death_prob)

    print 'log_probs2:'
    print log_probs

    #add birth/clutter measurement association entries to the log-prob matrix
    for m_idx in range(M):
        print 'log_probs3:'
        print log_probs

        clutter_col = T + 2*m_idx
        birth_col = T + 1 + 2*m_idx
        assert(params.SPEC['birth_clutter_likelihood'] == 'aprox1')
        log_probs[m_idx][clutter_col] = ln_zero_approx(params.clutter_lambdas_by_group[get_immutable_set_meas_names(meas_groups[m_idx])]) + \
            ln_zero_approx(birth_clutter_likelihood(meas_groups[m_idx], params, 'clutter')*params.p_clutter_likelihood)
        log_probs[m_idx][birth_col] = ln_zero_approx(params.birth_lambdas_by_group[get_immutable_set_meas_names(meas_groups[m_idx])]) + \
            ln_zero_approx(birth_clutter_likelihood(meas_groups[m_idx], params, 'birth')*params.p_birth_likelihood)
    print 'log_probs4:'
    print log_probs

    #set bottom right quadrant to 0's
    for row_idx in range(M, M+2*T):
        for col_idx in range(T, T+2*M):
            log_probs[row_idx][col_idx] = 0.0
    print 'log_probs5:'
    print log_probs

    return log_probs

def convert_assignment_matrix2(assignment_matrix, M, T):
    '''  
    use along with construct_log_probs_matrix2()
    Inputs:
    - assignment_matrix: numpy arrays with dimensions (measurement_count+2, target_count+2).
        representing a state of measurement associations and whether each unobserved target
        is alive or dead.
    - M: #measurements (int)
    - T: #targets (int)

    Outputs:
    - meas_associations: list of integers, the measurement associations represented by assignment_matrix.
        where meas_associations[j] is an integer representing the association of the jth measurement with:
            clutter = -1
            target association = [0,target_count-1]
            birth = target_count
    - dead_target_indices: list of integers, of length #dead targets.  if target_idx = i in [0,target_count-1]
        is in the list dead_target_indices, target i died.
    
    '''
    meas_associations = []

    #read off assignments
    for m_idx in range(M):
        cur_meas_assoc = None
        clutter_col = T + 2*m_idx
        birth_col = T + 1 + 2*m_idx
        for t_idx in range(T):
            if (np.isclose(assignment_matrix[m_idx,t_idx], 1, rtol=1e-02, atol=1e-02)):
                cur_meas_assoc = t_idx
        #clutter
        if cur_meas_assoc == None and np.isclose(assignment_matrix[m_idx,clutter_col], 1, rtol=1e-02, atol=1e-02):
            cur_meas_assoc = -1
        #birth
        if cur_meas_assoc == None:
            assert(np.isclose(assignment_matrix[m_idx,birth_col], 1, rtol=1e-02, atol=1e-02)), (assignment_matrix, M, T, m_idx, birth_col, assignment_matrix[m_idx,birth_col])
            cur_meas_assoc = T
        meas_associations.append(cur_meas_assoc)

    assert(len(meas_associations) == M)

    #read off target deaths
    dead_target_indices = []
    for t_idx in range(T):
        #target is unassociated
        if not t_idx in meas_associations:
            lives_row_idx = M + 2*t_idx
            dies_row_idx = M + 1 + 2*t_idx
            if (np.isclose(assignment_matrix[dies_row_idx,t_idx], 1, rtol=1e-02, atol=1e-02)):
                dead_target_indices.append(t_idx)
            else:
                assert(np.isclose(assignment_matrix[lives_row_idx,t_idx], 1, rtol=1e-02, atol=1e-02))

    return(meas_associations, dead_target_indices)

def solve_gumbel_perturbed_assignment2(log_probs, M, T):
    '''
    use along with construct_log_probs_matrix2()


    BELOW WASN"T CHECKED FROM COPYING FROM solve_gumbel_perturbed_assignment  
    We would like to sample from the optimal proposal distribution p(x_k|y_1:k, x_1:k-1),
    the normalization constant is intractable to compute.  We can compute p(y_k, x_k|y_1:k-1, x_1:k-1),
    which is proporitional to p(x_k|y_1:k, x_1:k-1).  Max(p(y_k, x_k|y_1:k-1, x_1:k-1)) can be
    found by solving a linear program given a matrix log probabilities.  We approximate adding gumbel 
    noise to every possible assignment(which would require a combinatorial number of Gumbels), 
    by adding a Gumbel to each entry of the log_probs matrix.  Instead of taking the maximum of 
    each of these Gumbels (which would make the problem nonconvex) we further approximate by summing them.

    Inputs:
    - log_probs: numpy matrix with dimensions (#measurements + 2, #targets + 2)
    - M: #measurements (int)
    - T: #targets (int)

    Outpus:
    - assigment: numpy matrix with same dimensions as log_probs, this is the assignment that results
        the largest log probability after perturbing log_probs with Gumbel noise

    - max_log_prob: this is the value of trace((log_probs+Gumbel_Noise)*assignment.T)

    '''
    #sample a Gumbel matrix
    G = numpy.random.gumbel(loc=0.0, scale=1.0, size=(log_probs.shape[0], log_probs.shape[1]))

    number_of_assignments = M + T

    #solve convex optimization problem
    A = cvx.Variable(log_probs.shape[0], log_probs.shape[1])
    objective = cvx.Maximize(cvx.trace((log_probs)*(A.T)) + cvx.trace(G*(A.T))/number_of_assignments) 
    constraints = [A>=0]                   
    for row_idx in range(M):
        constraints.append(cvx.sum_entries(A[row_idx, :]) == 1)
    for col_idx in range(T):
        constraints.append(cvx.sum_entries(A[:, col_idx]) == 1)

    for row_idx in range(M, M+2*T, 2):
        constraints.append(cvx.sum_entries(A[row_idx, :])
                         + cvx.sum_entries(A[row_idx+1, :]) == 1)
    for col_idx in range(T, T+2*M, 2):
        constraints.append(cvx.sum_entries(A[:, col_idx])
                         + cvx.sum_entries(A[:, col_idx+1]) == 1)

#    for idx, cur_solver in enumerate([cvx.CVXOPT, cvx.ECOS_BB, cvx.SCS, cvx.ECOS]):
    for idx, cur_solver in enumerate([cvx.SCS, cvx.ECOS]):
        prob = cvx.Problem(objective, constraints)
        print idx
        prob.solve(solver=cur_solver)
        assignment = A.value
        max_log_prob = prob.value
        print "assignment:", assignment
        print "max_log_prob:", assignment

    assert(np.isclose(np.sum(assignment), number_of_assignments, rtol=1e-02, atol=1e-02))
    return(assignment, max_log_prob)









def construct_log_probs_matrix3(particle, meas_groups, total_target_count, p_target_deaths, params, kill_all_unassociated_targets=True,\
    always_create_target=True):
    '''
    M = #measurements
    T = #targets

    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle         
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Outputs:
    - log_probs: numpy matrix with dimensions (2*M+2*T)x(2*M+2*T) of log probabilities.
        np.trace(np.dot(log_probs,A.T) will be the log probability of an assignment A, given our
        Inputs.  (Where an assignment defines measurement associations to targets, and is marginalized

    - clutter_probs_conditioned_unassoc: (list of floats) length M, the probability that each measurement
        is clutter conditioned on it being unassociated with any target
    '''

    log_probs4, conditional_birth_probs4, conditional_death_probs4, clutter_probs_conditioned_unassoc = construct_log_probs_matrix4(particle, meas_groups, total_target_count, p_target_deaths, params)
    
    #construct a (2*M+2*T)x(2*M+2*T) matrix of log probabilities
    M = len(meas_groups)
    T = total_target_count
    log_probs = np.ones((2*M + 2*T, 2*T + 2*M))
    log_probs *= -1*INFEASIBLE_COST #setting all entries to very negative value

    p_target_does_not_emit = params.target_groupEmission_priors[ImmutableSet([])]

#    print 'log_probs1:'
#    print log_probs

    #calculate log probs for target association entries in the log-prob matrix




    for t_idx in range(T):
        #always birth measurements and propogate clutter/birth probabilities
        if params.SPEC['optimistically_birth_targets']:# and particle.targets.living_targets[t_idx].fully_living == False:
            target_is_clutter_prob = particle.targets.living_targets[t_idx].clutter_prob_for_optimistic_birth
            fully_living_prob = 1 - target_is_clutter_prob
        else:
            target_is_clutter_prob = 0.0
            fully_living_prob = 1.0


        for m_idx in range(M):
        #calculate log probs for measurement-target association entries in the log-prob matrix
            likelihood = memoized_assoc_likelihood(particle, meas_groups[m_idx], t_idx, params)
            assert(likelihood >= 0.0), likelihood
            if likelihood > 0.0:
                cur_prob = ln_zero_approx(likelihood)
            else:
                cur_prob = -999 #(np.exp(-999) == 0) evaluates to True
            cur_prob += ln_zero_approx(params.target_groupEmission_priors[get_immutable_set_meas_names(meas_groups[m_idx])]) 
            log_probs[m_idx][t_idx] = cur_prob + ln_zero_approx(fully_living_prob)
            assert(not math.isnan(log_probs[m_idx][t_idx]))
        #calculate log probs for target doesn't emit and lives/dies entries in the log-prob matrix
        lives_row_idx = M + 2*t_idx
        dies_row_idx = M + 1 + 2*t_idx
        #would probably be better to kill offscreen targets before association
        if kill_all_unassociated_targets:
            cur_death_prob = .99999999999999
            # cur_death_prob = 1.0
        else:
            if(particle.targets.living_targets[t_idx].offscreen == True):
                cur_death_prob = .999999999999 #sloppy should define an epsilon or something
            else:
                cur_death_prob = particle.targets.living_targets[t_idx].death_prob
            if(cur_death_prob == 1.0):
                cur_death_prob = .99999999999 #still getting an error with domain error, trying this

            if(cur_death_prob == 0):
                cur_death_prob = 10**-100

        assert(p_target_does_not_emit > 0 and cur_death_prob > 0 and cur_death_prob <= 1.0), (p_target_does_not_emit, cur_death_prob)
        log_probs[lives_row_idx][t_idx] = ln_zero_approx(p_target_does_not_emit) + ln_zero_approx(1.0 - cur_death_prob) + ln_zero_approx(fully_living_prob)
        # log_probs[dies_row_idx][t_idx] = ln_zero_approx(p_target_does_not_emit) + ln_zero_approx(cur_death_prob)
        log_probs[dies_row_idx][t_idx] = ln_zero_approx(p_target_does_not_emit*cur_death_prob*fully_living_prob + target_is_clutter_prob)
        assert(not math.isnan(log_probs[lives_row_idx][t_idx]))
        assert(not math.isnan(log_probs[dies_row_idx][t_idx]))

#    print 'log_probs2:'
#    print log_probs

    clutter_probs_conditioned_unassoc = []

    #add birth/clutter measurement association entries to the log-prob matrix
    for m_idx in range(M):
#        print 'log_probs3:'
#        print log_probs

        clutter_col = T + 2*m_idx
        birth_col = T + 1 + 2*m_idx
        assert(params.SPEC['birth_clutter_likelihood'] == 'aprox1')

        if get_immutable_set_meas_names(meas_groups[m_idx]) in params.birth_lambdas_by_group:
            birth_lambda = params.birth_lambdas_by_group[get_immutable_set_meas_names(meas_groups[m_idx])]
        if not get_immutable_set_meas_names(meas_groups[m_idx]) in params.birth_lambdas_by_group or birth_lambda == 0:
            #use a small value if we never saw one of these groups in our training data            
            birth_lambda = min(params.birth_lambdas_by_group.itervalues())/100

        if get_immutable_set_meas_names(meas_groups[m_idx]) in params.clutter_lambdas_by_group:
            clutter_lambda = params.clutter_lambdas_by_group[get_immutable_set_meas_names(meas_groups[m_idx])]
        if not get_immutable_set_meas_names(meas_groups[m_idx]) in params.clutter_lambdas_by_group or clutter_lambda == 0:
            clutter_lambda = min(params.clutter_lambdas_by_group.itervalues())/100

        if always_create_target:
            # log_probs[m_idx][clutter_col] = -1 * INFEASIBLE_COST
            log_probs[m_idx][clutter_col] = -np.inf

        else:
            log_probs[m_idx][clutter_col] = ln_zero_approx(clutter_lambda) + \
                ln_zero_approx(birth_clutter_likelihood(meas_groups[m_idx], params, 'clutter')*params.p_clutter_likelihood)
        assert(not math.isnan(log_probs[m_idx][clutter_col]))    

        clutter_probs_conditioned_unassoc.append(clutter_lambda*birth_clutter_likelihood(meas_groups[m_idx], params, 'clutter')*params.p_clutter_likelihood/ \
            (clutter_lambda * birth_clutter_likelihood(meas_groups[m_idx], params, 'clutter')*params.p_clutter_likelihood + \
            birth_lambda * birth_clutter_likelihood(meas_groups[m_idx], params, 'birth')*params.p_birth_likelihood))

        log_probs[m_idx][birth_col] = ln_zero_approx(birth_lambda) + \
            ln_zero_approx(birth_clutter_likelihood(meas_groups[m_idx], params, 'birth')*params.p_birth_likelihood)
        assert(not math.isnan(log_probs[m_idx][birth_col]))            
        #HACKING BELOW!!!
#        log_probs[m_idx][birth_col] = ln_zero_approx(params.clutter_lambdas_by_group[get_immutable_set_meas_names(meas_groups[m_idx])]) + \
#            ln_zero_approx(birth_clutter_likelihood(meas_groups[m_idx], params, 'birth')*params.p_birth_likelihood)
#    print 'log_probs4:'
#    print log_probs

    assert(len(clutter_probs_conditioned_unassoc) == M)
    #set bottom right quadrant to 0's
    for row_idx in range(M, 2*M+2*T):
        for col_idx in range(T, 2*T+2*M):
            log_probs[row_idx][col_idx] = 0.0
    # print 'construct_log_probs_matrix3 called'
    # print "log_probs3 before smoothing:"
    # print log_probs

    if params.SPEC['smooth_assoc_probs'] == True:
        smoothed_assoc_probs = np.exp(log_probs[:M, :T])
        smoothed_assoc_probs = 2/(1 + np.exp(-params.SPEC['smoothing_factor']*smoothed_assoc_probs)) - 1 #smoothing_factor should be larger than roughly too, bigger is more smoothing
        for m_idx in range(M):
            for t_idx in range(T):
                log_probs[m_idx, t_idx] = ln_zero_approx(smoothed_assoc_probs[m_idx, t_idx])


    # print 'construct_log_probs_matrix3 called'
    # print "log_probs3 after smoothing:"
    # print log_probs
    # print "for comparison, log_probs4:"
    # print log_probs4
    # print "conditional_birth_probs4:", conditional_birth_probs4
    # print "conditional_death_probs4:", conditional_death_probs4

    return log_probs, clutter_probs_conditioned_unassoc



def construct_log_probs_matrix4(particle, meas_groups, total_target_count, p_target_deaths, params, kill_all_unassociated_targets=True,\
    always_create_target=True):
    '''
    M = #measurements
    T = #targets


    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle         
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Outputs:
    - log_probs: numpy matrix with dimensions (M+T)x(M+T) with probabilities (note we return the natural log of these values!):
        [a_11    ...     a_1T   um_1 0   ...  0]
        [.               .      0   .          ]
        [.               .      .      .       ]
        [.               .      .         .    ]
        [a_M1    ...     a_MT   0    ...   um_M]
        [ut_1    ...     ut_T   1    ...      1]
        [.               .      .             .]
        [.               .      .             .]
        [ut_1    ...     ut_T   1    ...      1]    
        - upper left quadrant is a MxT submatrix and composed of a_ij = the association probability of
          measurement i to target j
        - upper right quadrant is an MxM submatrix.  Row i is composed of M repetitions of 
          um_i = the probability that measurement i is unassociated with a target (marginalized over whether the
          measurement is clutter or a birth)
        - lower left quadrant is a TxT submatrix.  It is a diagonal matrix with elements ut_i = the
          probability that target i doesn't emit a measuremnt (marginalized over
          whether it lives or dies)
        - lower right quadrant is an TxM submatrix of all 1's

    - conditional_birth_probs: numpy array of length (#measurement groups).  conditional_birth_probs[i] is the probability
        that measurement group i is a birth, given that it is not associated with any target. (1 - conditional_birth_probs[i]) 
        is the probability that measurement group i is clutter, given that it is not associated with any target.
    - conditional_death_probs: numpy array of length (#targets).  conditional_death_probs[i] is the probability
        that target i dies, given that it is not associated with a measurement. (1 - conditional_death_probs[i]) 
        is the probability that target i lives, given that it is not associated with a measurement.



    '''
    M = len(meas_groups)
    T = total_target_count
    log_probs = np.ones((M + T, M + T))
    log_probs *= -1*INFEASIBLE_COST #setting all entries to very negative value

    p_target_does_not_emit = params.target_groupEmission_priors[ImmutableSet([])]
    
    conditional_birth_probs = np.ones(M)
    conditional_death_probs = np.ones(T)
    clutter_probs_conditioned_unassoc = []
#    print 'log_probs1:'
#    print log_probs

    #calculate log probs for target association entries in the log-prob matrix

    for t_idx in range(T):
        if params.SPEC['optimistically_birth_targets']:# and particle.targets.living_targets[t_idx].fully_living == False:
            target_is_clutter_prob = particle.targets.living_targets[t_idx].clutter_prob_for_optimistic_birth
            fully_living_prob = 1 - target_is_clutter_prob
        else:
            target_is_clutter_prob = 0.0
            fully_living_prob = 1.0

        for m_idx in range(M):
        #calculate log probs for measurement-target association entries in the log-prob matrix
            likelihood = memoized_assoc_likelihood(particle, meas_groups[m_idx], t_idx, params)
            assert(likelihood >= 0.0), likelihood
            if likelihood > 0.0:
                cur_prob = ln_zero_approx(likelihood)
            else:
                cur_prob = -999 #(np.exp(-999) == 0) evaluates to True
            cur_prob += ln_zero_approx(params.target_groupEmission_priors[get_immutable_set_meas_names(meas_groups[m_idx])]) 
            log_probs[m_idx][t_idx] = cur_prob * fully_living_prob
            assert(not math.isnan(log_probs[m_idx][t_idx]))
        #calculate log probs for target doesn't emit and lives/dies entries in the log-prob matrix
        if kill_all_unassociated_targets:
            # cur_death_prob = .99999999999999
            cur_death_prob = 1.0
        else:
            if(particle.targets.living_targets[t_idx].offscreen == True):
                cur_death_prob = .999999999999 #sloppy should define an epsilon or something
            else:
                cur_death_prob = particle.targets.living_targets[t_idx].death_prob
            if(cur_death_prob == 1.0):
                cur_death_prob = .99999999999 #still getting an error with domain error, trying this

            if(cur_death_prob == 0):
                cur_death_prob = 10**-100

        assert(p_target_does_not_emit > 0 and cur_death_prob > 0 and cur_death_prob <= 1.0), (p_target_does_not_emit, cur_death_prob)
        log_probs[M:,t_idx] = ln_zero_approx(p_target_does_not_emit*fully_living_prob + target_is_clutter_prob)
        # log_probs[lives_row_idx][t_idx] = ln_zero_approx(p_target_does_not_emit) + ln_zero_approx(1.0 - cur_death_prob)
        # log_probs[dies_row_idx][t_idx] = ln_zero_approx(p_target_does_not_emit) + ln_zero_approx(cur_death_prob)
        conditional_death_probs[t_idx] = (fully_living_prob*p_target_does_not_emit*cur_death_prob + target_is_clutter_prob)/\
                                        (fully_living_prob*p_target_does_not_emit + target_is_clutter_prob)

        print "conditional_death_probs[t_idx]:", conditional_death_probs[t_idx]


    if params.SPEC['smooth_assoc_probs'] == True:
        smoothed_assoc_probs = np.exp(log_probs[:M, :T])
        smoothed_assoc_probs = 2/(1 + np.exp(-params.SPEC['smoothing_factor']*smoothed_assoc_probs)) - 1 #smoothing_factor should be larger than roughly too, bigger is more smoothing
        log_probs[:M, :T] = np.log(smoothed_assoc_probs)


#    print 'log_probs2:'
#    print log_probs

    #add birth/clutter measurement association entries to the log-prob matrix
    for m_idx in range(M):
#        print 'log_probs3:'
#        print log_probs

        assert(params.SPEC['birth_clutter_likelihood'] == 'aprox1')

        if get_immutable_set_meas_names(meas_groups[m_idx]) in params.birth_lambdas_by_group:
            birth_lambda = params.birth_lambdas_by_group[get_immutable_set_meas_names(meas_groups[m_idx])]
        if not get_immutable_set_meas_names(meas_groups[m_idx]) in params.birth_lambdas_by_group or birth_lambda == 0:
            #use a small value if we never saw one of these groups in our training data            
            birth_lambda = min(params.birth_lambdas_by_group.itervalues())/100

        if get_immutable_set_meas_names(meas_groups[m_idx]) in params.clutter_lambdas_by_group:
            clutter_lambda = params.clutter_lambdas_by_group[get_immutable_set_meas_names(meas_groups[m_idx])]
        if not get_immutable_set_meas_names(meas_groups[m_idx]) in params.clutter_lambdas_by_group or clutter_lambda == 0:
            clutter_lambda = min(params.clutter_lambdas_by_group.itervalues())/100

        if always_create_target:
            # log_clutter_prob = -1 * INFEASIBLE_COST
            log_clutter_prob = -np.inf
        else:
            log_clutter_prob = ln_zero_approx(clutter_lambda) + \
                ln_zero_approx(birth_clutter_likelihood(meas_groups[m_idx], params, 'clutter')*params.p_clutter_likelihood)

        orig_log_clutter_prob = ln_zero_approx(clutter_lambda) + \
            ln_zero_approx(birth_clutter_likelihood(meas_groups[m_idx], params, 'clutter')*params.p_clutter_likelihood)


        log_birth_prob = ln_zero_approx(birth_lambda) + \
            ln_zero_approx(birth_clutter_likelihood(meas_groups[m_idx], params, 'birth')*params.p_birth_likelihood)

        log_birth_or_clutter_prob = ln_zero_approx(np.exp(log_clutter_prob) + np.exp(log_birth_prob))
        log_probs[m_idx][T +  m_idx] = log_birth_or_clutter_prob
        assert(not math.isnan(log_probs[m_idx][T +  m_idx]))
        conditional_birth_probs[m_idx] = np.exp(log_birth_prob)/(np.exp(log_birth_prob) + np.exp(log_clutter_prob))
        clutter_probs_conditioned_unassoc.append(np.exp(orig_log_clutter_prob)/(np.exp(log_birth_prob) + np.exp(orig_log_clutter_prob)))
    #set bottom right quadrant to 0's
    for row_idx in range(M, M+T):
        for col_idx in range(T, T+M):
            log_probs[row_idx][col_idx] = 0.0
#    print 'log_probs5:'
#    print log_probs

    # print "!"*80
    # print "log_probs4 matrix:"
    # for row in range(log_probs.shape[0]):
    #     for col in range(log_probs.shape[1]):
    #         print log_probs[row][col],
    #     print 
    # print

    return log_probs, conditional_birth_probs, conditional_death_probs, clutter_probs_conditioned_unassoc

def ln_zero_approx(x):
    #return a small number instead of -infinity for ln(0)
    if x == 0:
        return -sys.maxint
    else:
        return math.log(x)

def convert_assignment_matrix3(assignment_matrix, M, T):
    '''  

    M = #measurements
    T = #targets
    use along with construct_log_probs_matrix3()
    Inputs:
    - assignment_matrix: numpy arrays with dimensions (2*M+2*T)x(2*M+2*T).
        representing a state of measurement associations and whether each unobserved target
        is alive or dead.
    - M: #measurements (int)
    - T: #targets (int)

    Outputs:
    - meas_associations: list of integers, the measurement associations represented by assignment_matrix.
        where meas_associations[j] is an integer representing the association of the jth measurement with:
            clutter = -1
            target association = [0,target_count-1]
            birth = target_count
    - dead_target_indices: list of integers, of length #dead targets.  if target_idx = i in [0,target_count-1]
        is in the list dead_target_indices, target i died.
    
    '''
    meas_associations = []

    #read off assignments
    for m_idx in range(M):
        cur_meas_assoc = None
        clutter_col = T + 2*m_idx
        birth_col = T + 1 + 2*m_idx
        for t_idx in range(T):
            if (np.isclose(assignment_matrix[m_idx,t_idx], 1, rtol=1e-02, atol=1e-02)):
                cur_meas_assoc = t_idx
        #clutter
        if cur_meas_assoc == None and np.isclose(assignment_matrix[m_idx,clutter_col], 1, rtol=1e-02, atol=1e-02):
            cur_meas_assoc = -1
        #birth
        if cur_meas_assoc == None:
            assert(np.isclose(assignment_matrix[m_idx,birth_col], 1, rtol=1e-02, atol=1e-02)), (assignment_matrix, M, T, m_idx, birth_col, assignment_matrix[m_idx,birth_col])
            cur_meas_assoc = T
        meas_associations.append(cur_meas_assoc)

    assert(len(meas_associations) == M)

    #read off target deaths
    dead_target_indices = []
    for t_idx in range(T):
        #target is unassociated
        if not t_idx in meas_associations:
            lives_row_idx = M + 2*t_idx
            dies_row_idx = M + 1 + 2*t_idx
            if (np.isclose(assignment_matrix[dies_row_idx,t_idx], 1, rtol=1e-02, atol=1e-02)):
                dead_target_indices.append(t_idx)
            else:
                assert(np.isclose(assignment_matrix[lives_row_idx,t_idx], 1, rtol=1e-02, atol=1e-02))

    return(meas_associations, dead_target_indices)


#TEST ME against convert_assignment_matrix3!!!
def convert_assignment_pairs_to_associations3(assignment_pairs, M, T):
    '''  
    Same functionality as convert_assignment_matrix3, but takes in a list of pairs corresponding
    to 1's in the assignment matrix.

    M = #measurements
    T = #targets
    use along with construct_log_probs_matrix3()
    Inputs:
    - assignment_pairs: list of pairs where each pair represents an association in the assignment (1's in assignment matrix)
    - M: #measurements (int)
    - T: #targets (int)

    Outputs:
    - meas_associations: list of integers, the measurement associations represented by assignment_matrix.
        where meas_associations[j] is an integer representing the association of the jth measurement with:
            clutter = -1
            target association = [0,target_count-1]
            birth = target_count
    - dead_target_indices: list of integers, of length #dead targets.  if target_idx = i in [0,target_count-1]
        is in the list dead_target_indices, target i died.
    
    '''
    assert(len(assignment_pairs) == 2*(M+T))
    meas_associations = []

    assignment_pairs = sorted(assignment_pairs, key=itemgetter(0)) #sort pairs by row_idx

    for row_idx in range(M): #get all measurement associations
        assert(assignment_pairs[row_idx][0] == row_idx)
        assoc_idx = assignment_pairs[row_idx][1]
        if assoc_idx < T: #measurement-target association
            meas_associations.append(assoc_idx)
        else: #birth/clutter association
            clutter_col = T + 2*row_idx
            birth_col = T + 1 + 2*row_idx
            if assoc_idx == clutter_col: #clutter
                meas_associations.append(-1)
            else: #birth
                assert(assoc_idx == birth_col), (row_idx, assoc_idx, birth_col, M, T)
                meas_associations.append(T)

    #get unassociated target deaths
    dead_target_indices = []    
    for row_idx in range(M, M+2*T): 
        col_idx = assignment_pairs[row_idx][1]
        if col_idx < T: #we have an unassociated target
            dies_row_idx = M + 1 + 2*col_idx
            if row_idx == dies_row_idx: #target dies
                dead_target_indices.append(col_idx)



    return(meas_associations, dead_target_indices)


def convert_assignment_pairs_to_matrix3(assignment_pairs, M, T):
    '''  
    Inputs:
    - assignment_pairs: list of pairs where each pair represents an association in the assignment (1's in assignment matrix)
    - M: #measurements (int)
    - T: #targets (int)

    Outputs:
    - assignment_matrix: numpy array with dimensions (2*M+2*T)x(2*M+2*T).    
    
    '''
    assert(len(assignment_pairs) == (2*M+2*T)), (assignment_pairs, M, T)

    assignment_matrix = np.zeros(((2*M+2*T), (2*M+2*T)))
    for (row_idx, col_idx) in assignment_pairs:
        assignment_matrix[row_idx, col_idx] = 1

    return assignment_matrix


def solve_gumbel_perturbed_assignment3(log_probs, M, T):
    '''
    use along with construct_log_probs_matrix3()


    BELOW DESCRIPTION WASN"T CHECKED FROM COPYING FROM solve_gumbel_perturbed_assignment  
    We would like to sample from the optimal proposal distribution p(x_k|y_1:k, x_1:k-1),
    the normalization constant is intractable to compute.  We can compute p(y_k, x_k|y_1:k-1, x_1:k-1),
    which is proporitional to p(x_k|y_1:k, x_1:k-1).  Max(p(y_k, x_k|y_1:k-1, x_1:k-1)) can be
    found by solving a linear program given a matrix log probabilities.  We approximate adding gumbel 
    noise to every possible assignment(which would require a combinatorial number of Gumbels), 
    by adding a Gumbel to each entry of the log_probs matrix.  Instead of taking the maximum of 
    each of these Gumbels (which would make the problem nonconvex) we further approximate by summing them.

    Inputs:
    - log_probs: numpy matrix with dimensions (#measurements + 2, #targets + 2)
    - M: #measurements (int)
    - T: #targets (int)

    Outpus:
    - assigment: numpy matrix with same dimensions as log_probs, this is the assignment that results
        the largest log probability after perturbing log_probs with Gumbel noise

    - max_log_prob: this is the value of trace((log_probs+Gumbel_Noise)*assignment.T)

    '''
    number_of_assignments = 2*(M + T)

    #sample a Gumbel matrix
    G = numpy.random.gumbel(loc=0.0, scale=1.0, size=(log_probs.shape[0], log_probs.shape[1]))
    G /= number_of_assignments

    #solve convex optimization problem
    A = cvx.Variable(log_probs.shape[0], log_probs.shape[1])
    objective = cvx.Maximize(cvx.trace((log_probs)*(A.T)) + cvx.trace(G*(A.T))/number_of_assignments) 
    constraints = [A>=0]                   
    for row_idx in range(2*(M + T)):
        constraints.append(cvx.sum_entries(A[row_idx, :]) == 1)
    for col_idx in range(2*(M + T)):
        constraints.append(cvx.sum_entries(A[:, col_idx]) == 1)

######    for idx, cur_solver in enumerate([cvx.CVXOPT, cvx.ECOS_BB, cvx.SCS, cvx.ECOS]):
#####    for idx, cur_solver in enumerate([cvx.SCS, cvx.ECOS]):
#####        prob = cvx.Problem(objective, constraints)
#####        print idx
#####        prob.solve(solver=cur_solver)
#####        assignment = A.value
#####        max_log_prob = prob.value
#####        print "assignment:", assignment
#####        print "max_log_prob:", assignment

    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=cvx.SCS)
    assignment = A.value
    max_log_prob = prob.value

    assert(np.isclose(np.sum(assignment), number_of_assignments, rtol=1e-02, atol=1e-02))
    return(assignment, max_log_prob)









def associate_meas_gumbel_exact(particle, meas_groups, total_target_count, p_target_deaths, params, meas_counts_by_source, cur_time):
    '''
    Sample measurement associations from the optimal proposal distribution 
    p(c_k | e_{1-k-1}, c_{1:k-1}, y_{1:k})
    using the Gumbel max trick over all enumerated associations

    ONLY WORKS WITH 1 measurement source
    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    '''
    log_probs = construct_log_probs_matrix(particle, meas_groups, total_target_count, p_target_deaths, params)

    USE_GUMBEL_TRICK = True
    if USE_GUMBEL_TRICK:
        #find the maximum Gumbel perturbed log probability of all assignments
        all_assignments = enumerate_assignments(total_target_count, len(meas_groups))
        all_perturbed_log_probs = [np.trace(np.dot(log_probs,A.T))
            +numpy.random.gumbel(loc=0.0, scale=1.0, size=1) for A in all_assignments]

    #    print 'given', total_target_count, 'targets and', len(meas_groups), 'measurements, we are checking',
    #    print len(all_assignments), 'possible assignments'

        max_log_prob = max(all_perturbed_log_probs)
        assignment = all_assignments[all_perturbed_log_probs.index(max_log_prob)]
        (meas_associations, dead_target_indices) = convert_assignment_matrix(assignment)

        unnormalized_log_proposal_probability = np.trace(np.dot(log_probs, assignment.T))
        unnormalized_proposal_probability = np.exp(unnormalized_log_proposal_probability)

        #Now compute the exact partition function
        all_log_probs = [np.trace(np.dot(log_probs,A.T)) for A in all_assignments]
        all_probs = [np.exp(log_prob) for log_prob in all_log_probs]
        partition_val = np.sum(all_probs)
        proposal_probability = unnormalized_proposal_probability/partition_val
    else: #don't use gumbel for sampling, debugging
        all_assignments = enumerate_assignments(total_target_count, len(meas_groups))
        ######## DEBUGGING ###########
        print 'total_target_count =', total_target_count
        print 'measurement_count =', len(meas_groups)
        print 'number of assignments =', len(all_assignments)
#        for assignment in all_assignments:
#            print assignment
#            print 

        ######## DONE DEBUGGING ###########

        all_unnorm_log_probs = [np.trace(np.dot(log_probs,A.T)) for A in all_assignments]
        all_unnorm_probs = [np.exp(log_prob) for log_prob in all_unnorm_log_probs]
        partition_val = np.sum(all_unnorm_probs)
        all_norm_probs = [prob/partition_val for prob in all_unnorm_probs]


        ######## DEBUGGING ###########
        matrix_proposal_excluding_deaths = {}
        for idx, assignment in enumerate(all_assignments):
            (meas_associations, dead_target_indices) = convert_assignment_matrix(assignment)
            if tuple(meas_associations) in matrix_proposal_excluding_deaths:
                matrix_proposal_excluding_deaths[tuple(meas_associations)] += \
                    all_norm_probs[idx]
            else:
                matrix_proposal_excluding_deaths[tuple(meas_associations)] = \
                    all_norm_probs[idx]

        (a, b, c, d, check_partition_val, check_proposal_distribution, check_proposal_distr_dict) = associate_meas_optimal(particle, meas_groups, \
            total_target_count, p_target_deaths, params, meas_counts_by_source, cur_time=cur_time)
        
        for assoc, check_prob in check_proposal_distr_dict.iteritems():
            assert(assoc in matrix_proposal_excluding_deaths)
            assert(np.isclose(check_prob, matrix_proposal_excluding_deaths[assoc], rtol=1e-04, atol=1e-04)), (assoc, check_prob, matrix_proposal_excluding_deaths[assoc])

        print 'length of matrix proposal distribution excluding deaths =', len(matrix_proposal_excluding_deaths)
        print 'length of optimal distribution =', len(check_proposal_distribution)
        assert(len(matrix_proposal_excluding_deaths) == len(check_proposal_distribution)), (len(matrix_proposal_excluding_deaths), len(check_proposal_distribution))
#        print 'matrix proposal distribution excluding deaths ='
#        print (matrix_proposal_excluding_deaths)
#        print 'optimal distribution ='
#        print (check_proposal_distribution)

        ######## DONE DEBUGGING ###########

        sampled_assign_idx = np.random.choice(len(all_norm_probs), p=all_norm_probs)

        proposal_probability = all_norm_probs[sampled_assign_idx]
        assignment = all_assignments[sampled_assign_idx]
        (meas_associations, dead_target_indices) = convert_assignment_matrix(assignment)




############ THIS DOESN"T REALLY BELONG HERE, BUT FOLLOWING RETURN VALUES FOR OTHER PROPOSAL DISTRIBUTIONS ############
    #list of detection group centers, meas_grp_means[i] is a 2-d numpy array
    #of the position of meas_groups[i]
    meas_grp_covs = []   
    meas_grp_means2D = []
    meas_grp_means4D = []
    for (index, detection_group) in enumerate(meas_groups):
        (combined_meas_mean, combined_covariance) = combine_arbitrary_number_measurements_4d(params.posAndSize_inv_covariance_blocks, 
                            params.meas_noise_mean, detection_group)
        combined_meas_pos = combined_meas_mean[0:2]
        meas_grp_means2D.append(combined_meas_pos)
        meas_grp_means4D.append(combined_meas_mean)
        meas_grp_covs.append(combined_covariance)
############ END THIS DOESN"T REALLY BELONG HERE, BUT FOLLOWING RETURN VALUES FOR OTHER PROPOSAL DISTRIBUTIONS ############


    return(meas_associations, meas_grp_means4D, meas_grp_covs, proposal_probability, dead_target_indices)



def solve_perturbed_max_gumbel(particle, meas_groups, total_target_count, p_target_deaths, params):
    '''
    Solve gumbel perturbed linear program to approximately sample
    from p(x_k| x_1:k-1, y_1:k).


    '''
    USE_LOG_PROBS_3 = False

    if USE_LOG_PROBS_3:
        log_probs = construct_log_probs_matrix3(particle, meas_groups, total_target_count, p_target_deaths, params)

        #solve a perturbed assignment problem 
        (assignment, max_log_prob) = solve_gumbel_perturbed_assignment3(log_probs, len(meas_groups), total_target_count)
        print 'log_probs:',
        print log_probs

        print 'assignment:',
        print assignment

        print 'max_log_prob:', max_log_prob

    #not used here, but should probably combine this function with where used in traditional SIR gumbel
    #    unnormalized_log_proposal_probability = np.trace(np.dot(log_probs, assignment.T))
    #    unnormalized_proposal_probability = np.exp(unnormalized_log_proposal_probability)

        (meas_associations, dead_target_indices) = convert_assignment_matrix3(assignment, len(meas_groups), total_target_count)


        return(meas_associations, dead_target_indices, max_log_prob)
    else:
        log_probs = construct_log_probs_matrix(particle, meas_groups, total_target_count, p_target_deaths, params)

        #solve a perturbed assignment problem where the number of unobserved targets, clutter measurements
        #and birth measurements must sum to ubc_count for every possible value that ubc_count can take
        assignment = None
        max_log_prob = None
        for ubc_count in range(np.abs(len(meas_groups) - total_target_count), len(meas_groups) + total_target_count + 1, 2):
            (cur_assignment, cur_max_log_prob) = solve_gumbel_perturbed_assignment(log_probs, ubc_count)
            if max_log_prob == None or cur_max_log_prob > max_log_prob:
                assignment = cur_assignment
                max_log_prob = cur_max_log_prob

    #not used here, but should probably combine this function with where used in traditional SIR gumbel
    #    unnormalized_log_proposal_probability = np.trace(np.dot(log_probs, assignment.T))
    #    unnormalized_proposal_probability = np.exp(unnormalized_log_proposal_probability)

        (meas_associations, dead_target_indices) = convert_assignment_matrix(assignment)


        return(meas_associations, dead_target_indices, max_log_prob)


def solve_perturbed_max_gumbel_exact(particle, meas_groups, total_target_count, p_target_deaths, params):
    '''
    Solve gumbel perturbed max log(p(x_k, y_k | x_1:k-1, y_1:k-1)) exactly by enumerating all
    possibly x_k and adding an independent gumbel to each to exactly sample
    from p(x_k| x_1:k-1, y_1:k).


    '''
    log_probs = construct_log_probs_matrix(particle, meas_groups, total_target_count, p_target_deaths, params)

    #find the maximum Gumbel perturbed log probability of all assignments
    all_assignments = enumerate_assignments(total_target_count, len(meas_groups))
    all_perturbed_log_probs = [np.trace(np.dot(log_probs,A.T))
        +numpy.random.gumbel(loc=0.0, scale=1.0, size=1) for A in all_assignments]

    print 'given', total_target_count, 'targets and', len(meas_groups), 'measurements, we are checking',
    print len(all_assignments), 'possible assignments'

    max_log_prob = max(all_perturbed_log_probs)
    assignment = all_assignments[all_perturbed_log_probs.index(max_log_prob)] #assignment with the maximum perturbed log probability
#not used here, but should probably combine this function with where used in traditional SIR gumbel
#    unnormalized_log_proposal_probability = np.trace(np.dot(log_probs, assignment.T))
#    unnormalized_proposal_probability = np.exp(unnormalized_log_proposal_probability)

    (meas_associations, dead_target_indices) = convert_assignment_matrix(assignment)


    return(meas_associations, dead_target_indices, max_log_prob)


def get_unobserved_target_indices(target_count, measurement_association):
    '''
    Inputs:
    - target_count: the number of targets    
    - measurement_association: list where measurement_association[j]
        is an integer representing the association of the jth measurement with:
            clutter = -1
            target association = [0,target_count-1]
            birth = target_count

    Outputs:
    - unobserved_targets: list of integer indices of targets that are not associated
        with a measurement in measurement_association
    '''
    unobserved_targets = []
    for i in range(target_count):
        if not i in measurement_association:
            unobserved_targets.append(i)
    return unobserved_targets

def get_assoc_assignments(target_count, measurement_association):
    '''
    get all assignments for a particular association of measurements (enumerate 
    unobserved target life/death options)
    Inputs:
    - target_count: the number of targets    
    - measurement_association: list where measurement_association[j]
        is an integer representing the association of the jth measurement with:
            clutter = -1
            target association = [0,target_count-1]
            birth = target_count

    Outputs:
    - assoc_assignments: list of numpy arrays with dimensions (measurement_count+2, target_count+2).
        each array represents a state of measurement associations and whether each unobserved target
        is alive or dead for the particular measurement associations specified by measurement_association
    '''
    assoc_assignments = []
    base_assignment = np.zeros((len(measurement_association) + 2, target_count + 2))
    for meas_idx, assoc_idx in enumerate(measurement_association):
        assert(assoc_idx >= -1 and assoc_idx <= target_count)
        if assoc_idx >= 0 and assoc_idx < target_count: #target association
            base_assignment[meas_idx][assoc_idx] = 1
        elif assoc_idx == -1: #clutter
            base_assignment[meas_idx][target_count] = 1
        else:
            assert(assoc_idx == target_count) #clutter
            base_assignment[meas_idx][target_count+1] = 1

    unobserved_targets = get_unobserved_target_indices(target_count, measurement_association)
    unobserved_target_count = len(unobserved_targets)
    possible_deaths = []
    for assign_idx in range(2**unobserved_target_count):
        cur_deaths = []
        for unobserved_targ_idx in range(unobserved_target_count):
            cur_deaths.append(assign_idx//(2**unobserved_targ_idx) % 2)
        possible_deaths.append(cur_deaths)

    for cur_deaths in possible_deaths:
        cur_assignment = np.copy(base_assignment)
        for idx, target_idx in enumerate(unobserved_targets):
            if cur_deaths[idx] == 0: #target with index target_idx lives
                cur_assignment[len(measurement_association)][target_idx] = 1
            else: #target with index target_idx dies
                cur_assignment[len(measurement_association)+1][target_idx] = 1
        assoc_assignments.append(cur_assignment)
    
    return assoc_assignments


def enumerate_assignments(target_count, measurement_count):
    '''
    return all the possible valid assignment matrices, that is measurement associations (target,birth, and clutter)
    and unobserved target life/death options
    Inputs:
    - target_count: the number of targets
    - measurement_count: the number of measurements
    
    Outputs:
    - all_assignments: list of numpy arrays with dimensions (measurement_count+2, target_count+2).
        each array represents a state of measurement associations and whether each unobserved target
        is alive or dead
    '''
    all_assignments = []
    all_measurement_associations = enumerate_measurement_associations(target_count, measurement_count)
    for meas_assoc in all_measurement_associations:
        all_assignments.extend(get_assoc_assignments(target_count, meas_assoc))
    return all_assignments

def enumerate_measurement_associations(target_count, measurement_count):
    '''
    Inputs:
    - target_count: the number of targets
    - measurement_count: the number of measurements

    Outputs:
    - all_possible_measurement_associations: list of lists, all_possible_measurement_associations[i]
        is the i'th possible measurement association, where all_possible_measurement_associations[i][j]
        is an integer representing the association of the jth measurement with:
            clutter = -1
            target association = [0,target_count-1]
            birth = target_count
    '''
    all_possible_measurement_associations = []
    #target association count; we can association between and all targets with a measurement
    for t_a_c in range(target_count + 1): 
        #birth count; we can have between 0 and measurement_count - t_a_c births
        for b_c in range(measurement_count-t_a_c+1): 
            #clutter count; the remaining measurements must all be clutter
            c_c = measurement_count - t_a_c - b_c
            #create a list [0,1,2,...,#measurements - 1]
            measurement_indices = range(measurement_count)
            #enumerate all possible groups of measurements that can be associated
            #with targets.  There will be (#measurements choose t_a_c) of these
            for t_a_meas_indices in combinations(measurement_indices, t_a_c):
                #create a list of measurement indices that are NOT associated with a target
                remaining_meas_indices = [idx for idx in measurement_indices if not idx in t_a_meas_indices]
                assert(len(remaining_meas_indices)+len(t_a_meas_indices) == len(measurement_indices))
                #enumerate all possible groups of measurements that can be associated
                #with births.  There will be ((#measurements - t_a_c) choose b_c) of these
                for b_meas_indices in combinations(remaining_meas_indices, b_c):
                    #create a list of remaining measurement indices that must be associated with clutter
                    c_meas_indices = [idx for idx in remaining_meas_indices if not idx in b_meas_indices]
                    assert(len(c_meas_indices)+len(b_meas_indices) == len(remaining_meas_indices))
                    #now enumerate every permutation of target indices that can be associated
                    #with the combination of measurement indices in t_a_meas_indices
                    for t_a_target_indices in permutations(range(target_count), t_a_c):
                        #now create the current measurement association
                        cur_meas_association = [-99 for i in range(len(measurement_indices))]
                        #iterate over MEASUREMENTs that are associated with SOME target
                        #idx: the current measurement's index in the tuple of only those measurements
                        #that are associated with targets.
                        #t_a_meas_idx: the current measurement's index in the list of all measurements
                        for (idx, t_a_meas_idx) in enumerate(t_a_meas_indices):
                            cur_associated_target_idx = t_a_target_indices[idx]
                            cur_meas_association[t_a_meas_idx] = cur_associated_target_idx
                        #set birth measurement associations
                        for b_meas_idx in b_meas_indices:
                            cur_meas_association[b_meas_idx] = target_count
                        #set clutter measurement associations           
                        for c_meas_idx in c_meas_indices:
                            cur_meas_association[c_meas_idx] = -1
                        #check we assigned each measurement an association
                        for association in cur_meas_association:
                            assert(association != -99)
                            assert(association >= -1 and association <= target_count)
                        assert(not cur_meas_association in all_possible_measurement_associations)
                        all_possible_measurement_associations.append(cur_meas_association)    

    return all_possible_measurement_associations



def get_immutable_set_meas_names(meas_group):
    '''
    Inputs:
    - meas_group: a dictionary of detections, key='det_name', value=detObject

    Outputs:
    - meas_names_set: an ImmutableSet of the det_names in this group (all the keys in meas_group)
    '''
    #create set of the names of detection sources present in this detection group
    group_meas_names = []
    for meas_name, meas in meas_group.iteritems():
        group_meas_names.append(meas_name)
    meas_names_set = ImmutableSet(group_meas_names)        
    return meas_names_set        


def unnormalized_marginal_meas_target_assoc(particle, meas_groups, total_target_count, params):

    """
    Sample measurement target associations marginalized over birth association, clutter associations, and
    unassociated target death.
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - meas_grp_means4D
    - meas_grp_covs
    - marginal_meas_target_proposal_distr: numpy array, UNNORMALIZED proposal distribution over 3 measurement-target associations,
    marginalized over birth/clutter/life/death (important for modified SIS)
    - proposal_measurement_target_associations: list of lists of integers, proposal_measurement_target_associations[i]
        is the measurement association proposed with probability marginal_meas_target_proposal_distr[i].  Elements
        of association are integers with values:
            [0, total_target_count-1]: target association
            -1: birth OR clutter association
    """
    marginal_meas_target_proposal_distr = []
    proposal_measurement_target_associations = []
    #list of detection group centers, meas_grp_means[i] is a 2-d numpy array
    #of the position of meas_groups[i]
    meas_grp_covs = []   
    meas_grp_means2D = []
    meas_grp_means4D = []
    for (index, detection_group) in enumerate(meas_groups):
        (combined_meas_mean, combined_covariance) = combine_arbitrary_number_measurements_4d(params.posAndSize_inv_covariance_blocks, 
                            params.meas_noise_mean, detection_group)
        combined_meas_pos = combined_meas_mean[0:2]
        meas_grp_means2D.append(combined_meas_pos)
        meas_grp_means4D.append(combined_meas_mean)
        meas_grp_covs.append(combined_covariance)


    #get list of target bounding boxes  
    target_pos4D = []
    for target_index in range(total_target_count):
        target = particle.targets.living_targets[target_index]            
        target_location = np.squeeze(np.dot(params.H, target.x)) 
        target_pos4D.append(np.array([target_location[0], target_location[1], target.width, target.height]))

    meas_target_association_possibilities = []
    marginal_association_probs = []
#    for idx, max_assoc_cost in enumerate(params.SPEC['target_detection_max_dists']):


    if params.SPEC['targ_meas_assoc_metric'] == 'distance':
        max_costs = params.SPEC['target_detection_max_dists']
    else:
        assert(params.SPEC['targ_meas_assoc_metric'] == 'box_overlap')
        max_costs = params.SPEC['target_detection_max_overlaps']

    for max_assoc_cost in max_costs:
        list_of_measurement_associations = min_cost_measGrp_target_assoc(meas_grp_means4D, target_pos4D, params, max_assoc_cost)

        proposal_probability = 1.0
        observed_target_count = 0
        for m_idx in range(len(meas_groups)):
            t_idx = list_of_measurement_associations[m_idx]
            if t_idx != -1: #target association
                likelihood = memoized_assoc_likelihood(particle, meas_groups[m_idx], t_idx, params)
                assert(likelihood >= 0.0), likelihood
                if likelihood == 0:
                    likelihood = .000000000000000001 #something small, probably should make this nicer
                p_emission = params.target_groupEmission_priors[get_immutable_set_meas_names(meas_groups[m_idx])]
                proposal_probability *= likelihood*p_emission
                observed_target_count+=1

        unobserved_target_count = total_target_count - observed_target_count
        assert(unobserved_target_count >= 0)
        proposal_probability *= (params.target_groupEmission_priors[ImmutableSet([])])**unobserved_target_count


        #get a list of all possible measurement groups, length (2^#measurement sources)-1, each detection source can be in the set
        #or out of the set, but we can't have the empty set
        detection_groups = params.get_all_possible_measurement_groups()
        remaining_meas_count_by_groups = defaultdict(int)
        #count remaining measurements by measurement sources present in the group
        for (index, meas_group) in enumerate(meas_groups):
            if list_of_measurement_associations[index] == -1:
                remaining_meas_count_by_groups[get_immutable_set_meas_names(meas_group)] += 1

        #we multiply the proposal probability by this to marginalize over possible births and clutter associations
        b_c_prob_factor = conditional_birth_clutter_partition(remaining_meas_count_by_groups, params)

        proposal_probability *= b_c_prob_factor #now we have the marginal probability of these measurement associations
        marginal_meas_target_proposal_distr.append(proposal_probability)
        proposal_measurement_target_associations.append(list_of_measurement_associations)


    marginal_meas_target_proposal_distr = np.asarray(marginal_meas_target_proposal_distr)
    assert(np.sum(marginal_meas_target_proposal_distr) != 0.0)


    return (meas_grp_means4D, meas_grp_covs, marginal_meas_target_proposal_distr, proposal_measurement_target_associations)



def conditional_birth_clutter_distribution(remaining_meas_count_by_groups, params):
    '''
    Inputs:
    - remaining_meas_count_by_groups: dictioary, key = ImmutableSet(det_names in group), val=remaining measurement count of this 
        measurement group type

    Outputs 
    - conditional_proposals: dictionary with
        key = ImmutableSet(det_names in group)
        val = dictionary = 
           'proposal_distribution': numpy array, proposal distribution of birth/clutter counts for det_names groups
           'birth_counts': list of integers, same length as proposal_distribution, ith element contains birth count
                           for the ith proposal probability, clutter count = remaining_meas_count - birth_count
    '''

    conditional_proposals = {}
    partition_val = conditional_birth_clutter_partition(remaining_meas_count_by_groups, params)

    for meas_group, remaining_count in remaining_meas_count_by_groups.iteritems():
        if meas_group in params.birth_lambdas_by_group:
            birth_lambda = params.birth_lambdas_by_group[meas_group]
        if not meas_group in params.birth_lambdas_by_group or birth_lambda == 0:
            #use a small value if we never saw one of these groups in our training data            
            birth_lambda = min(params.birth_lambdas_by_group.itervalues())/100000

        if meas_group in params.clutter_lambdas_by_group:
            clutter_lambda = params.clutter_lambdas_by_group[meas_group]
        if not meas_group in params.clutter_lambdas_by_group or clutter_lambda == 0:
            clutter_lambda = min(params.clutter_lambdas_by_group.itervalues())/100000
                
        #calculate proposal for this measurement group type
        cur_group_proposal = []
        cur_group_birth_counts = []
        for b_c in range(remaining_count + 1): #we can have 0 to remaining_count births
            c_c = remaining_count - b_c #the rest of the remaining measurements of this group type are then clutters
            cur_prob = ((birth_lambda*params.p_birth_likelihood)**b_c)*((clutter_lambda*params.p_clutter_likelihood)**c_c)*nCr(remaining_count, b_c) #FIX ME, birth clutter likelihood!!!!!
            cur_group_proposal.append(cur_prob)
            cur_group_birth_counts.append(b_c)

        cur_group_proposal = np.asarray(cur_group_proposal)
        assert(np.sum(cur_group_proposal) != 0.0)
        cur_group_proposal /= float(np.sum(cur_group_proposal))
        cur_group_entry = {'proposal_distribution': cur_group_proposal,
                           'birth_counts': cur_group_birth_counts}
        conditional_proposals[meas_group] = cur_group_entry

    return conditional_proposals

def conditional_birth_clutter_partition(remaining_meas_count_by_groups, params):
    '''
    get the value of the partition function of all birth/clutter associations over all measurement group types,
    conditioned on the total remaining measurements of each group type
    Inputs:
    - remaining_meas_count_by_groups: dictioary, key = ImmutableSet(det_names in group), val=remaining measurement count of this 
        measurement group type
    '''
    partition_val = 1.0
    for meas_group, remaining_count in remaining_meas_count_by_groups.iteritems():
        if meas_group in params.birth_lambdas_by_group:
            birth_lambda = params.birth_lambdas_by_group[meas_group]
        if not meas_group in params.birth_lambdas_by_group or birth_lambda == 0:
            #use a small value if we never saw one of these groups in our training data            
            birth_lambda = min(params.birth_lambdas_by_group.itervalues())/100000

        if meas_group in params.clutter_lambdas_by_group:
            clutter_lambda = params.clutter_lambdas_by_group[meas_group]
        if not meas_group in params.clutter_lambdas_by_group or clutter_lambda == 0:
            clutter_lambda = min(params.clutter_lambdas_by_group.itervalues())/100000

        cur_sum = 0
        for b_c in range(remaining_count + 1): #we can have 0 to remaining_count births
            c_c = remaining_count - b_c #the rest of the remaining measurements of this group type are then clutters
            cur_sum += ((birth_lambda*params.p_birth_likelihood)**b_c)*((clutter_lambda*params.p_clutter_likelihood)**c_c)*nCr(remaining_count, b_c) #FIX ME, birth clutter likelihood!!!!!
        partition_val *= cur_sum
    return partition_val

def associate_meas_min_cost_corrected(particle, meas_groups, total_target_count, p_target_deaths, params):

    """
    First sample measurement associations from a small set of min cost matchings, with different max costs,
    marginalized over birth/clutter/death.  Then sample birth/death/clutter conditioned on measurement
    target associations
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - meas_targ_assoc: list of associations for each measurement group
    - proposal_probability: proposal probability of the sampled deaths and associations
        
    """
    # 1. sample measurment target associations marginalized over birth/clutter/unassociated death
    (meas_grp_means4D, meas_grp_covs, marginal_meas_target_proposal_distr, proposal_measurement_target_associations) = \
    unnormalized_marginal_meas_target_assoc(particle, meas_groups, total_target_count, params)

    marginal_meas_target_proposal_distr /= float(np.sum(marginal_meas_target_proposal_distr))

    sampled_meas_targ_assoc_idx = np.random.choice(len(marginal_meas_target_proposal_distr),
                                                        p=marginal_meas_target_proposal_distr)

    proposal_probability = marginal_meas_target_proposal_distr[sampled_meas_targ_assoc_idx]
    meas_targ_assoc = proposal_measurement_target_associations[sampled_meas_targ_assoc_idx]


    #get a list of all possible measurement groups, length (2^#measurement sources)-1, each detection source can be in the set
    #or out of the set, but we can't have the empty set
    detection_groups = params.get_all_possible_measurement_groups()
    remaining_meas_count_by_groups = defaultdict(int)
    unassociated_meas_indices_by_groups = defaultdict(list)
    #count remaining measurements by measurement sources present in the group
    for (index, meas_group) in enumerate(meas_groups):
        if meas_targ_assoc[index] == -1:
            remaining_meas_count_by_groups[get_immutable_set_meas_names(meas_group)] += 1
            unassociated_meas_indices_by_groups[get_immutable_set_meas_names(meas_group)].append(index)
    total_birth_count = 0
    total_clutter_count = 0
    conditional_proposals = conditional_birth_clutter_distribution(remaining_meas_count_by_groups, params)
    for meas_group, proposal_info in conditional_proposals.iteritems():
        # 2. sample # of births and clutter conditioned on 1. or each measurement group type
        sampled_birth_count_idx = np.random.choice(len(proposal_info['proposal_distribution']),
                                                    p=proposal_info['proposal_distribution'])
        sampled_birth_count = proposal_info['birth_counts'][sampled_birth_count_idx]
        sampled_clutter_count = remaining_meas_count_by_groups[meas_group] - sampled_birth_count
        total_birth_count += sampled_birth_count
        total_clutter_count += sampled_clutter_count
        birth_count_proposal_prob = proposal_info['proposal_distribution'][sampled_birth_count_idx]
        proposal_probability *= birth_count_proposal_prob

        # 3. uniformly sample which unassociated measurements are birth/clutter according to the counts from 2.
        unassociated_measurements = unassociated_meas_indices_by_groups[meas_group]
        proposal_probability *= nCr(len(unassociated_measurements), sampled_birth_count)
        for b_c_idx in range(sampled_birth_count):
            sampled_birth_idx = np.random.choice(len(unassociated_measurements))
            meas_targ_assoc[unassociated_measurements[sampled_birth_idx]] = total_target_count #set to birth val
            del unassociated_measurements[sampled_birth_idx]
 
    assert(meas_targ_assoc.count(total_target_count) == total_birth_count)
    assert(meas_targ_assoc.count(-1) == total_clutter_count)

    return(meas_targ_assoc, meas_grp_means4D, meas_grp_covs, proposal_probability, sampled_meas_targ_assoc_idx)


def get_det_features(seq_det_feature_array, det, frame_idx, debug_info):
    '''
    Get image features for a single object detection
    
    Inputs:
    - seq_det_feature_array: array, output by deepsort with dimensions
        (# of detections in the corresponding MOT sequence) x 135 containing
        features for detections from a specific object detection algorithm
        on a specific training sequence.
        Each row corresponds to a single detection in the particular training sequence, 
        the first 10 dimensions contain information about detection:
        [1 indexed frame_idx, junk, bb_left, bb_top, bb_width, bb_height, detection confidence score]
        The next 128 elements are extracted image features
    
    - det: type numpy array, the object detection whose features we will return
          [x_position, y_position, width, height]
    
    - frame_idx: int, the 0 indexed frame det occurs in
    Outputs:
    - features: 1d array of length 128, containing the
        features stored in seq_det_feature_array corresponding to det
    '''
    det_width = det[2]
    det_height = det[3]
    det_x1 = det[0] - det_width/2
    det_y1 = det[1] - det_height/2

    for det_idx in range(seq_det_feature_array.shape[0]):
        #MOT stores frame indices indexed from 1 so we subtract 1 below
        if seq_det_feature_array[det_idx][0] - 1 == frame_idx and \
              np.abs(seq_det_feature_array[det_idx][2] - det_x1) < .001 and\
              np.abs(seq_det_feature_array[det_idx][3] - det_y1) < .001 and\
              np.abs(seq_det_feature_array[det_idx][4] - det_width) < .001 and\
              np.abs(seq_det_feature_array[det_idx][5] - det_height) < .001:
            features = seq_det_feature_array[det_idx, 7:]
            assert(features.shape == (128,)), features.shape
#            print "detection found"
            return features
        
    print "couldn't find detection: seq_idx=", debug_info['seq_idx'], 'det_name=', debug_info['det_name'], 
    print 'frame_idx=', frame_idx, seq_det_feature_array[0][0] - 1, frame_idx==seq_det_feature_array[0][0] - 1
    print "det.x1", det.x1, seq_det_feature_array[0][2], det.x1==seq_det_feature_array[0][2]
    print "det.y1", det.y1, seq_det_feature_array[0][3], det.y1==seq_det_feature_array[0][3]
    print "det.width ", det.width, seq_det_feature_array[0][4], det.width==seq_det_feature_array[0][4]
    print "det.height", det.height, seq_det_feature_array[0][5], det.height==seq_det_feature_array[0][5]
    
    #f = open('./debug.quick', 'r')
    #[frame_idx, frame_idx_f, x1, x1_f,  y1, y1_f,  width, width_f,  height, height_f] = pickle.load(f)
    #f.close()
    f = open('./debug.quick', 'w')
    pickle.dump([frame_idx, seq_det_feature_array[0][0] - 1, det.x1, seq_det_feature_array[0][2],  det.y1, seq_det_feature_array[0][3],  det.width, seq_det_feature_array[0][4],  det.height, seq_det_feature_array[0][5]], f)
    f.close()                
    
    np.set_printoptions(precision=2, suppress=True)
    print seq_det_feature_array[0:10, 0:10]
    
    assert(False), 'detection not found'



def obj_l2_dist(a,b):
    x_dist = a.x - b.x
    y_dist = a.y - b.y
    distance = math.sqrt(x_dist**2 + y_dist**2)
    return distance

def obj_boxoverlap(a,b,criterion="union"):
    """
        boxoverlap computes intersection over union for bbox a and b in KITTI format.
        Criterion only implemented for 'union', overlap = (a inter b) / a union b).
        Inputs:
        - a: type gtObject or detObject
        - b: type gtObject or detObject
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
    if criterion.lower()=="union":
        o = inter / float(aarea+barea-inter)
    else:
        raise TypeError("Unkown type for criterion")
    return o

def obj_dist(a, b, distance_metric):
    if distance_metric == 'euclidean':
        return obj_l2_dist(a,b)
    elif distance_metric == 'bb_overlap':
        return 1-obj_boxoverlap(a,b,criterion="union")
    else:
        raise TypeError("Unkown type for distance_metric")



def get_k_NN(obj, all_neighbor_objs, k, distance_metric):
    '''
    return indices of and distances to the k nearest neighbors of obj in all_neighbor_objs
    
    Inputs:
    - obj: type BoundingBoxLocation
    - all_neighbor_objs: type list of BoundingBoxLocations
    - k: int, find nearest k neighbors
    - distance_metric: type string, either 'euclidean' or 'bb_overlap'

    Outputs:
    - neigbor_indices: length k of neigbor indices
        --nearest_k_neighbors[i] is the index in all_neighbor_objs of the ith nearest neighbor to obj
    '''
    #list of tuples with
    # -neighbor_dists[i][0]: index of a neighbor obj
    # -neighbor_dists[i][1]: distance between obj and the other obj
    neighbor_dists = []

    #find nearest neighbor objects to obj
    for neighbor_idx, neighbor_obj in enumerate(all_neighbor_objs):
        neighbor_dists.append((neighbor_idx, obj_dist(obj, neighbor_obj, distance_metric)))
    sorted_neighbor_dists = sorted(neighbor_dists, key=lambda x: x[1]) #sort by distance, the second element in each tuple
    nearest_k_neighbors = sorted_neighbor_dists[0:k]
    neigbor_indices = []
    for nn_idx, nn_dist in nearest_k_neighbors:
        neigbor_indices.append(nn_idx)
    #debugging
    assert(len(neigbor_indices) == min(k, len(all_neighbor_objs))), (len(neigbor_indices), k, len(all_neighbor_objs))
    for nn in neigbor_indices:
        assert(nn!=None)
    #end debugging
    return neigbor_indices

class BoundingBoxLocation:
    '''
    Position and size of a bounding box
    '''
    def __init__(self, x, y, width, height):
        self.x = x #x position of bounding box center
        self.y = y #y position of bounding box center
        self.width = width
        self.height = height

        self.x1 = x - width/2
        self.x2 = x + width/2
        self.y1 = y - width/2
        self.y2 = y + width/2


def get_target_emission_prior_conditionImgFt(target_img_feats, target_BB, target_assoc,\
    meas_grp_img_feats, meas_grp_BBs, params):
    '''

    Inputs:
    - target_img_feats: 2d array with dimension (1,128), image features for the target whose prior emission/association probability conditioned
        on image features we are calculating
    - target_BB: type BoundingBoxLocation for for the target whose prior emission/association probability conditioned
        on image features we are calculating
    - target_assoc: int, index in (meas_grp_img_feats) of the measurement group this target is associated with, or -1
        if unassociated
    - meas_grp_img_feats: list of length len(meas_groups) containing image features for each measurement group
    - meas_grp_BBs: list of length len(meas_groups) containing bounding boxes for each measurement group's combined bounding box
        order same as meas_grp_img_feats    

    Outputs:
    - emission_prob: the prior probabiltiy, conditioned on image features, that this target emits a measurement
        with image features meas_grp_img_feats[target_assoc] or doesn't emit a measurement (if target_assoc = -1)
    '''
    #k_NN: int, the number of nearest neighbor detections to each target we compute association priors for
    k_NN=params.SPEC['emission_prior_k_NN']
    #distance_metric: type string, either 'euclidean' or 'bb_overlap'
    distance_metric=params.SPEC['emission_prior_distance_metric']
    #tf_emission_priors: dictionary containing tensor flow model and placeholders    
    tf_emission_priors=params.SPEC['tf_emission_priors']

    #nn_meas is a list containing the nearest neighbor measurements to target i
    #nn_meas[i] is the integer index in meas_grp_img_feats/meas_grp_BBs of the ith nearest measurement 
    nn_meas = get_k_NN(obj=target_BB, all_neighbor_objs=meas_grp_BBs, k=k_NN, distance_metric=distance_metric)
    #if the target is associated with a measurement that is not one of its k_NN nearest neighbors, replace
    #its most distant measurement nearest neighbor with the associated measurement
    if target_assoc != -1 and not target_assoc in nn_meas:
        nn_meas[-1] = target_assoc

    cos_distances = []
    for meas_idx in nn_meas:
        cos_distances.append(spatial.distance.cosine(target_img_feats, meas_grp_img_feats[meas_idx]))

    #in case the target has less than k_NN nearest neighbors, pad cos_distances with 1's
    if len(cos_distances) < k_NN:
        num_missing_NN = k_NN - len(cos_distances)
        for i in range(num_missing_NN):
            cos_distances.append(1)

    cos_distances = np.array([cos_distances]) #make 2d array

    emission_priors = tf_emission_priors['priors'].eval(feed_dict={tf_emission_priors['x']: cos_distances,\
                                                                   tf_emission_priors['keep_prob']: 1.0})
    assert(emission_priors.size == k_NN+1), (emission_priors.size, k_NN+1)
    if target_assoc == -1:
        emission_prob = emission_priors[0, k_NN]
    else:
        emission_prob = emission_priors[0, nn_meas.index(target_assoc)]
    return emission_prob

def get_all_targets_emission_prior_conditionImgFt(particle, meas_groups, \
                    measurement_assoc,frame_idx, params):
    '''
    get the prior probability of all targets emitting/not emitting and associating with measurements as 
    specified in measurement_assoc, conditioned on bounding box image features
    Inputs:
    - particle: type Particle, get association/emission priors for every target in this Particle
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detOdetection (should be numpy array of [x,y,width,height])
    - measurement_assoc: list of length=#measurement groups.  measurement_assoc[i] = j means
        that the ith measurement group is associated with the jth target.  -1 means the ith
        measurement is not associated with any living target.    
    - frame_idx: int, the 0 indexed current frame 
    - params: type Parameters
    Outputs:
    - total_emission_prior: product of individual target emissions priors (modelling as independent)
    '''

    #seq_det_feature_arrays: dictionary with
    #    key: ('det_name', seq_idx)
    #    value: array output by deepsort with dimensions
    #        (# of detections in the corresponding MOT sequence) x 135 containing
    #        features for detections from the object detection algorithm specified
    #        by 'det_name' on a specific training sequence.
    #        Each row corresponds to a single detection in the particular training sequence, 
    #        the first 10 dimensions contain information about detection:
    #        [1 indexed frame_idx, junk, bb_left, bb_top, bb_width, bb_height, detection confidence score]
    #        The next 128 elements are extracted image features    
    seq_det_feature_arrays=params.SPEC['seq_det_feature_arrays']
    #seq_idx: int, index of the current sequence
    seq_idx=params.SPEC['seq_idx']
    meas_grp_means4D = []
    for (index, detection_group) in enumerate(meas_groups):
        (combined_meas_mean, combined_covariance) = combine_arbitrary_number_measurements_4d(params.posAndSize_inv_covariance_blocks, 
                            params.meas_noise_mean, detection_group)
        meas_grp_means4D.append(combined_meas_mean)



    #list of length len(meas_groups) containing image features for each measurement group
    meas_grp_img_feats = []
    #list of length len(meas_groups) containing bounding boxes for each measurement group's combined bounding box
    meas_grp_BBs = []
    for grp_idx, meas_grp in enumerate(meas_groups):
        meas_grp_BBs.append(BoundingBoxLocation(x=meas_grp_means4D[grp_idx][0],
                                                y=meas_grp_means4D[grp_idx][1],
                                                width=meas_grp_means4D[grp_idx][2],
                                                height=meas_grp_means4D[grp_idx][3]))
        for det_name in params.SPEC['det_names']:
            if det_name in meas_grp:
                det = meas_grp[det_name]
                debug_info = {'seq_idx': params.SPEC['seq_idx'], 'det_name': det_name}
                meas_grp_img_feats.append(get_det_features(seq_det_feature_arrays[(det_name, seq_idx)], det, frame_idx, debug_info))
                break

    #list of length (#living targets) containing image features for each living target
    #ordered in the standard order of particle.targets.living_targets
    all_target_img_feats = []
    #list of length (#living targets) containing bounding boxes for each living target
    #ordered in the standard order of particle.targets.living_targets
    target_BBs = []    
    for target_index in range(particle.targets.living_count):
        target = particle.targets.living_targets[target_index]            
        all_target_img_feats.append(target.img_features)
        target_BBs.append(BoundingBoxLocation(x=target.x[0][0],
                                              y=target.x[2][0],
                                              width=target.width,
                                              height=target.height))


    target_NN_dets = []
    total_emission_prior = 1.0
    for t_idx in range(particle.targets.living_count):
        if not t_idx in measurement_assoc:
            target_assoc = -1 #target isn't associated with a measurement
        else:
            target_assoc = measurement_assoc.index(t_idx) #index of measurement the target is associated with

        cur_target_emission_prior = get_target_emission_prior_conditionImgFt(target_img_feats=all_target_img_feats[t_idx], \
            target_BB=target_BBs[t_idx], target_assoc=target_assoc,\
            meas_grp_img_feats=meas_grp_img_feats, meas_grp_BBs=meas_grp_BBs, params=params)
        total_emission_prior *= cur_target_emission_prior

    return total_emission_prior



def min_cost_BB_assoc(bb_lA, bb_lB, distance_metric='bb_overlap', max_assoc_cost=.5):
    """
    Take two lists of bounding boxes and associate so the sum of associated distances is minimized.

    Inputs:
    - bb_lA: list of BoundingBoxLocations
    - bb_lB: list of BoundingBoxLocations
    - distance_metric: string 'bb_overlap' or 'euclidean'
    - max_assoc_cost: association costs (distances) greater than this value are not allowed

    Outputs:
    - measurement_assoc: list of length=len(bb_lA).  measurement_assoc[i] = j means
        that the ith bounding box from bb_lA is associated with the jth bounding box 
        from bb_lB.  -1 means the ith bounding box form bb_lA is not associated with 
        any bounding box from bb_lB.
    """

    hm = Munkres()
    max_cost = 1e9

    # use hungarian method to associate, using boxoverlap 0..1 as cost
    # build cost matrix
    cost_matrix = []
    this_ids = [[],[]]

    for bb_A in bb_lA:
        cost_row = []
        for bb_B in bb_lB:
            c = obj_dist(bb_A, bb_B, distance_metric)
            # gating for boxoverlap
            if c<=max_assoc_cost:
                cost_row.append(c)
            else:
                cost_row.append(max_cost)
        cost_matrix.append(cost_row)
    
    if len(bb_lA) is 0:
        cost_matrix=[[]]
    # associate
    #lin_assign = linear_assignment.LinearAssignment(cost_matrix)
    #solution = lin_assign.solution
    #association_list = zip([i for i in range(len(solution))], solution)
    association_list = hm.compute(cost_matrix)
    
    measurement_assoc = [-1 for i in range(len(bb_lA))]
    for row,col in association_list:
        # apply gating on boxoverlap
        c = cost_matrix[row][col]
        if c < max_cost:
            bb_A = bb_lA[row]
            bb_B = bb_lB[col]
            measurement_assoc[row] = col
            if np.abs(c - obj_dist(bb_A, bb_B, distance_metric)) > .000001:
                print "bb_lA:"
                for bb_A in bb_lA:
                    print bb_A.x, bb_A.y, bb_A.width, bb_A.height
                print "bb_lB:"
                for bb_B in bb_lB:
                    print bb_B.x, bb_B.y, bb_B.width, bb_B.height
                print "association_list:", association_list
            assert(np.abs(c - obj_dist(bb_A, bb_B, distance_metric)) < .000001), (c, obj_dist(bb_A, bb_B, distance_metric), cost_matrix, bb_lA, bb_lB)

    return measurement_assoc


def group_detectionSource_to_groundTruth(gt_objects, detection_locations, det_widths, det_heights):
    """
    Take a list of detections and try to associate them with ground truth bounding boxes

    Inputs:
    

    Outputs:
    - det_gt_association_pairs: (list of pairs of ints) each pair represents an association between 
        a detection and a ground truth object.  The pair specifies the indices (in input lists) of
        the (detection, ground truth object).
    """
    hm = Munkres()
    max_cost = 1e9

    # use hungarian method to associate, using boxoverlap 0..1 as cost
    # build cost matrix
    cost_matrix = []
    this_ids = [[],[]]

    assert(len(detection_locations) == len(det_widths) and len(det_widths) == len(det_heights))
    #combine into 4d detections
    detections = []
    for det_idx, det_loc in enumerate(detection_locations):
        detections.append(np.array([det_loc[0], det_loc[1], det_widths[det_idx], det_heights[det_idx]]))

    ground_truth = []
    for gt_object in gt_objects:
        ground_truth.append(np.array([gt_object.x, gt_object.y, gt_object.width, gt_object.height]))

    for cur_detection in detections:
        cost_row = []
        for cur_gt in ground_truth:
            c = 1-boxoverlap(cur_detection, cur_gt)
            # gating for boxoverlap
            if c<=.5:
                cost_row.append(c)
            else:
                cost_row.append(max_cost)

        cost_matrix.append(cost_row)
    
    if len(detections) is 0:
        cost_matrix=[[]]
    # associate
    #switch to linear_assignment for speed, but need to deal with more rows than columns
    #lin_assign = linear_assignment.LinearAssignment(cost_matrix)
    #solution = lin_assign.solution
    #association_list = zip([i for i in range(len(solution))], solution)
    association_list = hm.compute(cost_matrix)

    det_gt_association_pairs = []

    for det_idx,gt_idx in association_list:
        # apply gating on boxoverlap
        c = cost_matrix[det_idx][gt_idx]
        if c < max_cost:
            det_gt_association_pairs.append((det_idx, gt_idx))
    return det_gt_association_pairs

def get_gt_assoc(particle, meas_groups, cur_time, total_target_count, params, measurement_locations=None, widths=None, heights=None):
    """
    Check upper bound on our tracking performance by calculating ground truth measurement associations.
    Steps:
        1. combine meas_groups to form a single bounding box for each group and and associate
           with ground truth in gt_objects[seq_idx][frame_idx]
        2. associate the state of all targets (particle.targets.living_targets) at frame_idx-1 
            ground truth in gt_objects[seq_idx][frame_idx-1]
        3. associate measurements with targets based on results from 1 and 2

    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - cur_time: float, frame_idx*DEFAULT_TIME_STEP where frame_idx is 0 indexed
    - total_target_count: the number of living targets on the previous time instace, used
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - gt_associations: list of ground truth associations for each measurement group:
        -1: clutter
        [0, total_target_count-1]: target association
        total_target_count: birth
    - 
    - meas_assoc_gt_obj_ids: list containing the track_id of the ground truth object each measurement is 
        associated with, or -1 if the measurement is clutter

        
    """
    frame_idx = int(round(cur_time/DEFAULT_TIME_STEP))
    seq_idx=params.SPEC['seq_idx']

    #meas_grp_means4D: list of numpy arrays of combined measurment group x,y,width,height
    meas_grp_covs = []   
    meas_grp_means2D = []
    meas_grp_means4D = []
    for (index, detection_group) in enumerate(meas_groups):
        (combined_meas_mean, combined_covariance) = combine_arbitrary_number_measurements_4d(params.posAndSize_inv_covariance_blocks, 
                            params.meas_noise_mean, detection_group)
        combined_meas_pos = combined_meas_mean[0:2]
        meas_grp_means2D.append(combined_meas_pos)
        meas_grp_means4D.append(combined_meas_mean)
        meas_grp_covs.append(combined_covariance)

    #list of type BoundingBoxLocation for meas_groups
    meas_group_BBs = []
    #we assume iteration order over python dictionaries is consistent.
    #this should be fine since we aren't changing any values, but is somewhat dangerous.
    for det_group in meas_grp_means4D:
        print "det_group shape:", det_group.shape
        meas_group_BBs.append(BoundingBoxLocation(det_group[0], \
                                                  det_group[1], \
                                                  det_group[2], \
                                                  det_group[3]))

    #list of type boundingboxlocation for particle.targets.living_targets at frame_idx-1
    prv_target_BBs = []
    #list of type boundingboxlocation for particle.targets.living_targets at frame_idx
    cur_target_BBs = []
    #list of integers of the track_ids of the gt_objects each target at frame_idx was last associated with
    cur_target_last_gt_assocs = []
    for cur_target in particle.targets.living_targets:
        x = cur_target.all_states[-2][0][0][0]
        y = cur_target.all_states[-2][0][2][0]
        width = cur_target.all_states[-2][1]
        height = cur_target.all_states[-2][2]
        prv_target_BBs.append(BoundingBoxLocation(x, y, width, height))

        x = cur_target.all_states[-1][0][0][0]
        y = cur_target.all_states[-1][0][2][0]
        width = cur_target.all_states[-1][1]
        height = cur_target.all_states[-1][2]
        cur_target_BBs.append(BoundingBoxLocation(x, y, width, height))
        cur_target_last_gt_assocs.append(cur_target.last_gt_assoc)
    gt_objects = params.SPEC['gt_obects']
    #list of type BoundingBoxLocation for gt_objects[seq_idx][frame_idx]
    cur_gt_BBs = []
    #list of track_ids for gt_objects[seq_idx][frame_idx]
    cur_gt_IDs = []
    for gt_obj in gt_objects[seq_idx][frame_idx]:
        cur_gt_BBs.append(BoundingBoxLocation(gt_obj.x, \
                                              gt_obj.y, \
                                              gt_obj.width, \
                                              gt_obj.height))
        cur_gt_IDs.append(gt_obj.track_id)

    #list of type BoundingBoxLocation for gt_objects[seq_idx][frame_idx-1]
    prv_gt_BBs = []
    #list of track_ids for gt_objects[seq_idx][frame_idx-1]
    prv_gt_IDs = []
    for gt_obj in gt_objects[seq_idx][frame_idx-1]:
        prv_gt_BBs.append(BoundingBoxLocation(gt_obj.x, \
                                              gt_obj.y, \
                                              gt_obj.width, \
                                              gt_obj.height))
        prv_gt_IDs.append(gt_obj.track_id)

    #meas_gt_assoc_list[i] = j means meas_group i is associated with cur_gt j
    #meas_gt_assoc_list[i] = -1 means meas_group i clutter.
    meas_gt_assoc_list = min_cost_BB_assoc(meas_group_BBs, cur_gt_BBs, distance_metric='bb_overlap', max_assoc_cost=.5)
    #gt_target_assoc_list[i] = j means prv_gt i is associated with target j at frame_idx-1
    #gt_target_assoc_list[i] = -1 means prv_gt i is unassociated with any target at frame_idx-1.
    gt_target_assoc_list = min_cost_BB_assoc(prv_gt_BBs, prv_target_BBs, distance_metric='bb_overlap', max_assoc_cost=.5)

    MinCostAssoc_gtDeaths = False
    if MinCostAssoc_gtDeaths:
######### QUICK CHECK Min-cost associations and ground truth deaths #############
        #gt_target_assoc_list[i] = j means meas_group i is associated with target j at frame_idx
        #gt_target_assoc_list[i] = -1 means meas_group i is unassociated with any target at frame_idx.
        meas_target_assoc_list = min_cost_BB_assoc(meas_group_BBs, cur_target_BBs, distance_metric='bb_overlap', max_assoc_cost=.5)
        gt_associations = []
        #a list containing the track_id of the ground truth object each measurement is 
        #associated with, or -1 if the measurement is clutter
        meas_assoc_gt_obj_ids = []
        for meas_idx, target_idx in enumerate(meas_target_assoc_list):
            cur_gt_idx = meas_gt_assoc_list[meas_idx]
            if cur_gt_idx == -1: #this is a clutter measurement
                meas_assoc_gt_obj_ids.append(-1)
            else:
                gt_id = cur_gt_IDs[cur_gt_idx]
                meas_assoc_gt_obj_ids.append(gt_id)
    
            if target_idx == -1: #measurement unassociated with a target, clutter or birth
                gt_id = cur_gt_IDs[cur_gt_idx]
                if cur_gt_idx == -1: #this is a clutter measurement
                    gt_associations.append(-1)
                else:#this is a birth measurement
                    gt_associations.append(total_target_count)
    
            else:
                gt_associations.append(target_idx)
######### END QUICK CHECK Min-cost associations and ground truth deaths #############

    else:
        #associate measurements:
        # -clutter: if the measurement is not associated with any ground truth obect at frame_idx
        # -target: if the measurement is associated with a ground truth object at frame_idx and
        #   that same ground truth object is associated with a target at frame_idx-1
        # -birth: if the measurement is associated with a ground truth object at frame_idx and
        #   that ground truth obect is not associated with any targets at frame_idx-1
        gt_associations = []
        #a list containing the track_id of the ground truth object each measurement is 
        #associated with, or -1 if the measurement is clutter
        meas_assoc_gt_obj_ids = []
        for meas_idx, cur_gt_idx in enumerate(meas_gt_assoc_list):
            if cur_gt_idx == -1: #this is a clutter measurement
                gt_associations.append(-1)
                meas_assoc_gt_obj_ids.append(-1)
            else:
                gt_id = cur_gt_IDs[cur_gt_idx]
                meas_assoc_gt_obj_ids.append(gt_id)
                if gt_id in prv_gt_IDs:
                    prv_gt_idx = prv_gt_IDs.index(gt_id)
                    target_idx = gt_target_assoc_list[prv_gt_idx]
                    if target_idx == -1: #this is a birth measurement
                        gt_associations.append(total_target_count)
                    else:
                        gt_associations.append(target_idx)
                else: #this is a birth measurement
                    gt_associations.append(total_target_count)

    #Find ground truth target deaths as follows:
    # 1. Remove ground truth objects that have been associated with a measurement at frame_idx from cur_gt_BBs
    # 2. Remove associated targets from target_BBs
    # 3. Associate remaining targets with ground truth objects at frame_idx
    # 4a. Ground truth target deaths are targets that are still
    #     unassociated after 3.
    # 4b. Alternatively, ground truth target deaths are unassociated targets after 3. AND
    #     targets that are associated with a ground truth object other than the ground
    #     truth object they were last associated with
    USE_4B = True

    #indices of targets that have died
    gt_target_death_indices = []
    # 1. Find ground truth bounding boxes that have not been associated with a measurement at frame_idx 
    associated_gt_indices = [] #indices in cur_gt_BBs of unassociated gt objects
    for meas_idx, cur_gt_idx in enumerate(meas_gt_assoc_list):
        associated_gt_indices.append(cur_gt_idx)
    unassociated_gt_BBs = [] #bounding boxes of unassociated gt objects 
    unassociated_gt_track_ids = [] #track_ids of unassociated gt objects 
    for cur_gt_idx, cur_gt_bb in enumerate(cur_gt_BBs):
        if not cur_gt_idx in associated_gt_indices:
            unassociated_gt_BBs.append(cur_gt_bb)
            unassociated_gt_track_ids.append(cur_gt_IDs[cur_gt_idx])
    # 2. Find targets that have not been associated with a measurement at frame_idx
    unassociated_target_BBs = []
    unassociated_target_indices = []
    unassociated_target_last_gt_assoc_ids = [] #track_ids of gt object targets were last associated with
    for target_idx, target_BB in enumerate(cur_target_BBs):
        if not target_idx in gt_associations:
            unassociated_target_BBs.append(target_BB)
            unassociated_target_indices.append(target_idx)
            unassociated_target_last_gt_assoc_ids.append(cur_target_last_gt_assocs[target_idx])
#####    Quick test of killing all unassociated targets immediately    
#####            gt_target_death_indices.append(target_idx)
#####    proposal_probability = 1.0
#####    assert(len(meas_assoc_gt_obj_ids) == len(gt_associations))
#####    return(gt_associations, meas_grp_means4D, meas_grp_covs, proposal_probability, gt_target_death_indices, meas_assoc_gt_obj_ids)

    # 3. Associate remaining targets with remaining ground truth objects at frame_idx
    #unassoc_target_gt_assoc_list[i] = j means unassociated target i is associated with unassociated cur_gt j
    #unassoc_target_gt_assoc_list[i] = -1 means unassociated target i is unassociated and is dead
    unassoc_target_gt_assoc_list = min_cost_BB_assoc(unassociated_target_BBs, unassociated_gt_BBs, distance_metric='bb_overlap', max_assoc_cost=.5)
    for target_idx, target_assoc in enumerate(unassoc_target_gt_assoc_list):
        if target_assoc == -1:
            gt_target_death_indices.append(unassociated_target_indices[target_idx])
        elif USE_4B: #apply 4b
            last_gt_assoc_id = unassociated_target_last_gt_assoc_ids[target_idx]
            cur_gt_assoc_id = unassociated_gt_track_ids[target_assoc]
            if last_gt_assoc_id != cur_gt_assoc_id:
                gt_target_death_indices.append(unassociated_target_indices[target_idx])

    proposal_probability = 1.0

    assert(len(meas_assoc_gt_obj_ids) == len(gt_associations))
    return(gt_associations, meas_grp_means4D, meas_grp_covs, proposal_probability, gt_target_death_indices, meas_assoc_gt_obj_ids)


def associate_meas_min_cost(particle, meas_groups, total_target_count, p_target_deaths, params, cur_time, meas_counts_by_source):

    """
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using
    - cur_time: float, frame_idx*DEFAULT_TIME_STEP where frame_idx is 0 indexed

    Output:
    - list_of_measurement_associations: list of associations for each measurement group
    - proposal_probability: proposal probability of the sampled deaths and associations
        
    """

    #sample measurement associations
    birth_count = 0
    clutter_count = 0

    #list of detection group centers, meas_grp_means[i] is a 2-d numpy array
    #of the position of meas_groups[i]
    meas_grp_covs = []   
    meas_grp_means2D = []
    meas_grp_means4D = []
    for (index, detection_group) in enumerate(meas_groups):
        (combined_meas_mean, combined_covariance) = combine_arbitrary_number_measurements_4d(params.posAndSize_inv_covariance_blocks, 
                            params.meas_noise_mean, detection_group)
        combined_meas_pos = combined_meas_mean[0:2]
        meas_grp_means2D.append(combined_meas_pos)
        meas_grp_means4D.append(combined_meas_mean)
        meas_grp_covs.append(combined_covariance)


    sampled_assoc_idx = np.random.choice(2, p=[params.SPEC['prob_sample_assoc_uniform'], 1-params.SPEC['prob_sample_assoc_uniform']])
    if(sampled_assoc_idx == 0): #sample associations uniformly at random, so proposal distribution has full support
        possible_measurement_associations = [i for i in range(-1, total_target_count+1)]
        final_measurement_associations = []
        joint_proposal_probability = params.SPEC['prob_sample_assoc_uniform']
        for meas_idx in range(len(meas_groups)):
            cur_meas_assoc = np.random.choice(possible_measurement_associations)
            final_measurement_associations.append(cur_meas_assoc)
            joint_proposal_probability *= 1.0/len(possible_measurement_associations)
            if cur_meas_assoc!=-1 and cur_meas_assoc!=total_target_count: #targets are associated at most once
                possible_measurement_associations.remove(cur_meas_assoc)


        return(final_measurement_associations, meas_grp_means4D, meas_grp_covs, joint_proposal_probability)

    #otherwise, sample from min_cost measurement-target associations
    #get list of target bounding boxes  
    target_pos4D = []
    for target_index in range(total_target_count):
        target = particle.targets.living_targets[target_index]            
        target_location = np.squeeze(np.dot(params.H, target.x)) 
        target_pos4D.append(np.array([target_location[0], target_location[1], target.width, target.height]))


    complete_association_possibilities = []

    conditional_proposal_probabilities = []
    marginal_proposal_distribution = []

#    for idx, max_assoc_cost in enumerate(params.SPEC['target_detection_max_dists']):
    if params.SPEC['targ_meas_assoc_metric'] == 'distance':
        max_costs = params.SPEC['target_detection_max_dists']
    else:
        assert(params.SPEC['targ_meas_assoc_metric'] == 'box_overlap')
        max_costs = params.SPEC['target_detection_max_overlaps']

    print '#'*80, "measurement associations to sample from"
    for max_assoc_cost in max_costs:
        list_of_measurement_associations = min_cost_measGrp_target_assoc(meas_grp_means4D, target_pos4D, params, max_assoc_cost)
        print list_of_measurement_associations
        #the proposal probability of birth/clutter associations conditioned on list_of_measurement_associations
        #target-measurement associations
        conditional_proposal_probability = 1.0

        remaining_meas_count = list_of_measurement_associations.count(-1)
        for (index, detection_group) in enumerate(meas_groups):
            if list_of_measurement_associations[index] == -1:
                #create set of the names of detection sources present in this detection group
                group_det_names = []
                for det_name, det in detection_group.iteritems():
                    group_det_names.append(det_name)
                det_names_set = ImmutableSet(group_det_names)


                #create proposal distribution for the current measurement
                #compute target association proposal probabilities
                proposal_distribution_list = []


                cur_birth_prior = PRIOR_EPSILON
                if params.birth_clutter_model == 'training_counts':
                    for bc, prior in params.birth_count_priors.iteritems():
                        additional_births = max(0.0, min(bc - birth_count, remaining_meas_count))
                        if additional_births <= remaining_meas_count:
                            cur_birth_prior += prior*additional_births/remaining_meas_count 
                else:
                    assert(params.birth_clutter_model == 'poisson')
                    for additional_births in range(1,remaining_meas_count+1):
                        prior = params.birth_groupCount_prior(additional_births + birth_count)
                        cur_birth_prior += prior*additional_births/remaining_meas_count  

                cur_birth_prior *= params.birth_group_prior(det_names_set)
                cur_birth_prior *= params.SPEC['coord_ascent_params']['birth_proposal_prior_const'][0]
                assert(cur_birth_prior*params.p_birth_likelihood**len(detection_group) > 0), (cur_birth_prior,params.p_birth_likelihood,len(detection_group))



                cur_clutter_prior = PRIOR_EPSILON
                if params.birth_clutter_model == 'training_counts':                
                    for cc, prior in params.clutter_grpCountByFrame_priors.iteritems():
                        additional_clutter = max(0.0, min(cc - clutter_count, remaining_meas_count))
                        if additional_clutter <= remaining_meas_count:            
                            cur_clutter_prior += prior*additional_clutter/remaining_meas_count 
                else:
                    assert(params.birth_clutter_model == 'poisson')
                    for additional_clutter in range(1,remaining_meas_count+1):
                        prior = params.clutter_groupCount_prior(additional_clutter + clutter_count)
                        cur_clutter_prior += prior*additional_clutter/remaining_meas_count  

                cur_clutter_prior *= params.clutter_group_prior(det_names_set)
                cur_clutter_prior *= params.SPEC['coord_ascent_params']['clutter_proposal_prior_const'][0]
                assert(cur_clutter_prior*params.p_clutter_likelihood**len(detection_group) > 0), (cur_clutter_prior, params.p_clutter_likelihood, len(detection_group))


        #        cur_birth_prior = cur_clutter_prior
                if params.SPEC['birth_clutter_likelihood'] == 'const1':
                    proposal_distribution_list.append(cur_birth_prior*params.p_birth_likelihood**len(detection_group)) #Quick test, make nicer!!
                    proposal_distribution_list.append(cur_clutter_prior*params.p_clutter_likelihood**len(detection_group)) #Quick test, make nicer!!

                elif params.SPEC['birth_clutter_likelihood'] == 'const2':
                    proposal_distribution_list.append(cur_birth_prior*params.p_birth_likelihood) #Quick test, make nicer!!
                    proposal_distribution_list.append(cur_clutter_prior*params.p_clutter_likelihood) #Quick test, make nicer!!

                elif params.SPEC['birth_clutter_likelihood'] == 'aprox1':

                    birth_likelihood = birth_clutter_likelihood(detection_group, params, 'birth')
                    proposal_distribution_list.append(cur_birth_prior*birth_likelihood*params.p_birth_likelihood) 

                    clutter_likelihood = birth_clutter_likelihood(detection_group, params, 'clutter')
                    proposal_distribution_list.append(cur_clutter_prior*clutter_likelihood*params.p_clutter_likelihood) #Quick test, make nicer!!
                else:
                    print "Invalid params.SPEC['birth_clutter_likelihood']"
                    sys.exit(1);

                #normalize the proposal distribution
                proposal_distribution = np.asarray(proposal_distribution_list)
                assert(np.sum(proposal_distribution) != 0.0), (index, remaining_meas_count, len(proposal_distribution), proposal_distribution, birth_count, clutter_count, len(measurement_list), total_target_count)
                proposal_distribution /= float(np.sum(proposal_distribution))
                proposal_length = 2
                assert(len(proposal_distribution) == proposal_length), (proposal_length, len(proposal_distribution))

    #            if particle.max_importance_weight:
    #                print "proposal_distribution:", proposal_distribution

                sampled_assoc_idx = np.random.choice(len(proposal_distribution),
                                                        p=proposal_distribution)


                if(sampled_assoc_idx == 0): #birth association
                    birth_count += 1
                    list_of_measurement_associations[index] = total_target_count
                else: #clutter association
                    assert(sampled_assoc_idx == 1)
                    assert(list_of_measurement_associations[index] == -1) #already -1 from min_cost_measGrp_target_assoc
                    clutter_count += 1

                conditional_proposal_probability *= proposal_distribution[sampled_assoc_idx]
                remaining_meas_count -= 1

        complete_association_possibilities.append(list_of_measurement_associations)
        conditional_proposal_probabilities.append(conditional_proposal_probability)

        #calculate the exact association probability, (marginalized over target death)
        if params.SPEC['use_log_probs'] == 'False':
            likelihood = get_likelihood(particle, meas_groups, particle.targets.living_count,
                                           measurement_associations=list_of_measurement_associations, params=params, log=False)
            assoc_prior = get_assoc_prior(particle, meas_groups, meas_grp_associations=list_of_measurement_associations, \
                                          params=params, meas_counts_by_source=meas_counts_by_source, log=False, cur_time=cur_time)
            exact_probability = likelihood * assoc_prior    
            marginal_proposal_distribution.append(exact_probability)
    
        else:        
            log_likelihood = get_likelihood(particle, meas_groups, particle.targets.living_count,
                                           measurement_associations=list_of_measurement_associations, params=params, log=True)
            log_assoc_prior = get_assoc_prior(particle, meas_groups, meas_grp_associations=list_of_measurement_associations, \
                                              params=params, meas_counts_by_source=meas_counts_by_source, log=True, cur_time=cur_time)
            log_exact_probability = log_likelihood + log_assoc_prior
            marginal_proposal_distribution.append(log_exact_probability)


    marginal_proposal_distribution = np.asarray(marginal_proposal_distribution)
    if params.SPEC['use_log_probs'] == 'True': #exponentiate elements, dividing each by the largest before exponentiation to avoid 0's
        max_log_prob = np.max(marginal_proposal_distribution)
        marginal_proposal_distribution = marginal_proposal_distribution - max_log_prob
        marginal_proposal_distribution = np.exp(marginal_proposal_distribution)    

#   assert(np.sum(marginal_proposal_distribution) != 0.0), marginal_proposal_distribution
    if np.sum(marginal_proposal_distribution) == 0.0: #everything unlikely, sample uniformly
        marginal_proposal_distribution = np.ones(marginal_proposal_distribution.size)
    marginal_proposal_distribution /= float(np.sum(marginal_proposal_distribution))
    sampled_idx = np.random.choice(len(marginal_proposal_distribution),
                                                p=marginal_proposal_distribution)


    final_measurement_associations = complete_association_possibilities[sampled_idx]
    joint_proposal_probability = marginal_proposal_distribution[sampled_idx]*conditional_proposal_probabilities[sampled_idx]*\
                                 (1-params.SPEC['prob_sample_assoc_uniform'])

#    print 'returnHI'
    assert(remaining_meas_count == 0)
    return(final_measurement_associations, meas_grp_means4D, meas_grp_covs, joint_proposal_probability)



def associate_measurements_sequentially(particle, meas_groups, total_target_count, p_target_deaths, params):

    """
    Try sampling associations with each measurement sequentially
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - meas_groups: a list of detection groups, where each detection group is a dictionary of detections 
        in the group, key='det_name', value=detection (should be numpy array of [x,y,width,height])
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - list_of_measurement_associations: list of associations for each measurement group
    - proposal_probability: proposal probability of the sampled deaths and associations
        
    """
    list_of_measurement_associations = []
    proposal_probability = 1.0

    #sample measurement associations
    birth_count = 0
    clutter_count = 0
    remaining_meas_count = len(meas_groups)

    #list of detection group centers, meas_grp_means[i] is a 2-d numpy array
    #of the position of meas_groups[i]
    meas_grp_covs = []   
    meas_grp_means2D = []
    meas_grp_means4D = []
    for (index, detection_group) in enumerate(meas_groups):
        (combined_meas_mean, combined_covariance) = combine_arbitrary_number_measurements_4d(params.posAndSize_inv_covariance_blocks, 
                            params.meas_noise_mean, detection_group)
        combined_meas_pos = combined_meas_mean[0:2]
        meas_grp_means2D.append(combined_meas_pos)
        meas_grp_means4D.append(combined_meas_mean)
        meas_grp_covs.append(combined_covariance)


    def get_k_nearest_targets(measurement, k):
        """
        Inputs:
        - measurement: the measurement
        - k: integer, number of nearest targets to return

        Output:
        - k_nearest_target_indices: list of indices of the k nearest (L2 distance between 
            bounding box centers) targets in the living target list
        """

        k_nearest_target_indices = []
        k_nearest_target_dists = []
        for target_index in range(total_target_count):
            target = particle.targets.living_targets[target_index]            
            target_location = np.squeeze(np.dot(params.H, target.x))
            distance = (measurement[0] - target_location[0])**2 + (measurement[1] - target_location[1])**2
            if len(k_nearest_target_indices) < k: #add target
                k_nearest_target_indices.append(target_index)
                k_nearest_target_dists.append(distance)
            elif distance < max(k_nearest_target_dists):
                target_idx_to_replace = k_nearest_target_dists.index(max(k_nearest_target_dists))
                k_nearest_target_indices[target_idx_to_replace] = target_index
                k_nearest_target_dists[target_idx_to_replace] = distance

        return k_nearest_target_indices


    for (index, detection_group) in enumerate(meas_groups):
        #create proposal distribution for the current measurement
        #compute target association proposal probabilities
        proposal_distribution_list = []

        #create set of the names of detection sources present in this detection group
        group_det_names = []
        for det_name, det in detection_group.iteritems():
            group_det_names.append(det_name)
        det_names_set = ImmutableSet(group_det_names)

        if params.CHECK_K_NEAREST_TARGETS:
            targets_to_check = get_k_nearest_targets(meas_grp_means2D[index], params.K_NEAREST_TARGETS)
        else:
            targets_to_check = [i for i in range(total_target_count)]

#        for target_index in range(total_target_count):
        for target_index in targets_to_check:
            cur_target_likelihood = memoized_assoc_likelihood(particle, detection_group, target_index, params)
            targ_likelihoods_summed_over_meas = 0.0

            debug_idx = 0
            for meas_index2, detection_group2 in enumerate(meas_groups):
                targ_likelihoods_summed_over_meas += memoized_assoc_likelihood(particle, detection_group2, target_index, params)
                debug_idx += 1

            if((targ_likelihoods_summed_over_meas != 0.0) and (not target_index in list_of_measurement_associations)\
                and p_target_deaths[target_index] < 1.0):
                cur_target_prior = params.target_groupEmission_priors[det_names_set]*cur_target_likelihood \
                                  /targ_likelihoods_summed_over_meas
            else:
                cur_target_prior = 0.0

#            print "debug_idx", debug_idx
#            print "len(meas_groups)", len(meas_groups)
#            print "cur_target_prior", cur_target_prior
#            print "targ_likelihoods_summed_over_meas", targ_likelihoods_summed_over_meas
#            print "target_index in list_of_measurement_associations", (target_index in list_of_measurement_associations)
#            print " p_target_deaths[target_index] < 1.0", ( p_target_deaths[target_index] < 1.0)
#            print "p_target_deaths:", p_target_deaths
#            sleep(5)
            proposal_distribution_list.append(cur_target_likelihood*cur_target_prior)


        cur_birth_prior = PRIOR_EPSILON
        if params.birth_clutter_model == 'training_counts':
            for bc, prior in params.birth_count_priors.iteritems():
                additional_births = max(0.0, min(bc - birth_count, remaining_meas_count))
                if additional_births <= remaining_meas_count:
                    cur_birth_prior += prior*additional_births/remaining_meas_count 
        else:
            assert(params.birth_clutter_model == 'poisson')
            for additional_births in range(1,remaining_meas_count+1):
                prior = params.birth_groupCount_prior(additional_births + birth_count)
                cur_birth_prior += prior*additional_births/remaining_meas_count  

        cur_birth_prior *= params.birth_group_prior(det_names_set)
        assert(cur_birth_prior*params.p_birth_likelihood**len(detection_group) > 0), (cur_birth_prior,params.p_birth_likelihood,len(detection_group))



        cur_clutter_prior = PRIOR_EPSILON
        if params.birth_clutter_model == 'training_counts':                
            for cc, prior in params.clutter_grpCountByFrame_priors.iteritems():
                additional_clutter = max(0.0, min(cc - clutter_count, remaining_meas_count))
                if additional_clutter <= remaining_meas_count:            
                    cur_clutter_prior += prior*additional_clutter/remaining_meas_count 
        else:
            assert(params.birth_clutter_model == 'poisson')
            for additional_clutter in range(1,remaining_meas_count+1):
                prior = params.clutter_groupCount_prior(additional_clutter + clutter_count)
                cur_clutter_prior += prior*additional_clutter/remaining_meas_count  

        cur_clutter_prior *= params.clutter_group_prior(det_names_set)
        assert(cur_clutter_prior*params.p_clutter_likelihood**len(detection_group) > 0), (cur_clutter_prior, params.p_clutter_likelihood, len(detection_group))




#        cur_birth_prior = cur_clutter_prior
        if params.SPEC['birth_clutter_likelihood'] == 'const1':
            proposal_distribution_list.append(cur_birth_prior*params.p_birth_likelihood**len(detection_group)) #Quick test, make nicer!!
            proposal_distribution_list.append(cur_clutter_prior*params.p_clutter_likelihood**len(detection_group)) #Quick test, make nicer!!

        elif params.SPEC['birth_clutter_likelihood'] == 'const2':
            proposal_distribution_list.append(cur_birth_prior*params.p_birth_likelihood) #Quick test, make nicer!!
            proposal_distribution_list.append(cur_clutter_prior*params.p_clutter_likelihood) #Quick test, make nicer!!

        elif params.SPEC['birth_clutter_likelihood'] == 'aprox1':

            birth_likelihood = birth_clutter_likelihood(detection_group, params, 'birth')
            proposal_distribution_list.append(cur_birth_prior*birth_likelihood*params.p_birth_likelihood) 

            clutter_likelihood = birth_clutter_likelihood(detection_group, params, 'clutter')
            proposal_distribution_list.append(cur_clutter_prior*clutter_likelihood*params.p_clutter_likelihood) #Quick test, make nicer!!
        else:
            print "Invalid params.SPEC['birth_clutter_likelihood']"
            sys.exit(1);

        #normalize the proposal distribution
        proposal_distribution = np.asarray(proposal_distribution_list)
        assert(np.sum(proposal_distribution) != 0.0), (index, remaining_meas_count, len(proposal_distribution), proposal_distribution, birth_count, clutter_count, len(measurement_list), total_target_count)
        proposal_distribution /= float(np.sum(proposal_distribution))
        if params.CHECK_K_NEAREST_TARGETS:
            proposal_length = min(params.K_NEAREST_TARGETS+2, total_target_count+2)
            assert(len(proposal_distribution) == proposal_length), (proposal_length, len(proposal_distribution))

        else:
            assert(len(proposal_distribution) == total_target_count+2), len(proposal_distribution)

#        if particle.max_importance_weight:
#            print "proposal_distribution:", proposal_distribution

        sampled_assoc_idx = np.random.choice(len(proposal_distribution),
                                                p=proposal_distribution)

        if params.CHECK_K_NEAREST_TARGETS:
            possible_target_assoc_count = min(params.K_NEAREST_TARGETS, total_target_count)
            if(sampled_assoc_idx <= possible_target_assoc_count): #target or birth association
                if(sampled_assoc_idx == possible_target_assoc_count): #birth
                    birth_count += 1
                    list_of_measurement_associations.append(total_target_count)
                else: #target
                    list_of_measurement_associations.append(targets_to_check[sampled_assoc_idx])

            else: #clutter association
                assert(sampled_assoc_idx == possible_target_assoc_count+1)
                list_of_measurement_associations.append(-1)
                clutter_count += 1

        else: #we considered association with all targets
            if(sampled_assoc_idx <= total_target_count): #target or birth association
                list_of_measurement_associations.append(sampled_assoc_idx)
                if(sampled_assoc_idx == total_target_count):
                    birth_count += 1
            else: #clutter association
                assert(sampled_assoc_idx == total_target_count+1)
                list_of_measurement_associations.append(-1)
                clutter_count += 1

        proposal_probability *= proposal_distribution[sampled_assoc_idx]

        remaining_meas_count -= 1


    assert(remaining_meas_count == 0)
    return(list_of_measurement_associations, meas_grp_means4D, meas_grp_covs, proposal_probability)


def sample_target_deaths(particle, unassociated_targets, cur_time):
    """
    Sample target deaths, given they have not been associated with a measurement, using probabilities
    learned from data.
    Also kill all targets that are offscreen.

    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - unassociated_targets: a list of target indices that have not been associated with a measurement

    Output:
    - targets_to_kill: a list of targets (integer indices in particle.targets.living_targets)
        that have been sampled to die (not killed yet)
    - probability_of_deaths: the probability of the sampled deaths
    """
    targets_to_kill = []
    probability_of_deaths = 1.0

    for target_idx in range(len(particle.targets.living_targets)):
        #kill offscreen targets with probability 1.0
        if(particle.targets.living_targets[target_idx].offscreen == True):
            targets_to_kill.append(target_idx)
            assert(particle.targets.living_targets[target_idx].death_prob == 1.0)
        elif(target_idx in unassociated_targets):
            cur_death_prob = particle.targets.living_targets[target_idx].death_prob
            if(random.random() < cur_death_prob):
                targets_to_kill.append(target_idx)
                probability_of_deaths *= cur_death_prob
            else:
                probability_of_deaths *= (1 - cur_death_prob)
    return (targets_to_kill, probability_of_deaths)

def calc_death_prior(living_target_indices, p_target_deaths, unassociated_target_indices, log):
    """
    - log: Bool, True return log probability, False return actual probability

    """

    if log: #return log of prior
        log_death_prior = 0.0
        for (cur_target_index, cur_target_death_prob) in enumerate(p_target_deaths):
            if not(cur_target_index in living_target_indices):
                log_death_prior += ln_zero_approx(cur_target_death_prob)
                assert((cur_target_death_prob) != 0.0), cur_target_death_prob        
            elif cur_target_index in unassociated_target_indices:
                log_death_prior += ln_zero_approx((1.0 - cur_target_death_prob))
                assert((1.0 - cur_target_death_prob) != 0.0), cur_target_death_prob

        return log_death_prior



    else: #return actual prior
        death_prior = 1.0
        for (cur_target_index, cur_target_death_prob) in enumerate(p_target_deaths):
            if not(cur_target_index in living_target_indices):
                death_prior *= cur_target_death_prob
                assert((cur_target_death_prob) != 0.0), (cur_target_death_prob, p_target_deaths)
            elif cur_target_index in unassociated_target_indices:
                death_prior *= (1.0 - cur_target_death_prob)
                assert((1.0 - cur_target_death_prob) != 0.0), cur_target_death_prob

        return death_prior

def nCr(n,r):
    return math.factorial(n) / math.factorial(r) / math.factorial(n-r)


def count_association_orderings(meas_counts_by_source, birth_count_by_group, clutter_count_by_group):
    num_orderings = -1
    for cur_source_meas_count in meas_counts_by_source:
        if num_orderings == -1:
            num_orderings = math.factorial(cur_source_meas_count)
        else:
            num_orderings *= math.factorial(cur_source_meas_count)
    for (grp, cur_grp_birth_count) in birth_count_by_group.iteritems():
        orderings_fact = math.factorial(cur_grp_birth_count)
        assert(num_orderings % orderings_fact == 0), (num_orderings, orderings_fact, birth_count_by_group, clutter_count_by_group, meas_counts_by_source)
        num_orderings //= orderings_fact #use integer division
    for (grp, cur_grp_clutter_count) in clutter_count_by_group.iteritems():
        orderings_fact = math.factorial(cur_grp_clutter_count)
        assert(num_orderings % orderings_fact == 0), (num_orderings, orderings_fact, birth_count_by_group, clutter_count_by_group, meas_counts_by_source)
        num_orderings //= orderings_fact #use integer division
    return num_orderings


def count_meas_orderings(M, T, b, c):
    """
    We define target observation priors in terms of whether each target was observed and it
    is irrelevant which measurement the target is associated with.  Likewise, birth count priors
    and clutter count priors are defined in terms of total counts, not which specific measurements
    are associated with clutter and births.  This function counts the number of possible 
    measurement-association assignments given we have already chosen which targets are observed, 
    how many births occur, and how many clutter measurements are present.  

    If the prior probability of observing T specific targets, b births, and c clutter observations 
    given M measurements is divided by the count_meas_orderings(M, T, b, c), the prior probability 
    is split between all possible associations.  This ensures that our prior is a proper probability
    distribution that sums to one over the entire state space.  

    Calculates the ordered vector of associations by equally prior probability of unordered set of
    associations between all orderings.  This is the most straightforward idea, but it seems problematic.
    As the number of targets increases, the number of possible measurment target associations blows
    up and prior must be spilt between all.  It may make more sense to simply calculate the prior
    of an unordered measurement set and then calculate likelihood based on the unordered set of observations.

####### 
#######   However, we our calculating the prior:

#######   p(c_k, #y_k | e_1:k-1, c_1:k-1, y_1:k-1, #y_1:k-1)
#######   
#######   Note we are given all past measurements, associations, and the state of all living targets at the
#######   last time instance.  


    [
    *OLD EXPLANATION BELOW*:
    We view the the ordering of measurements on any time instance as arbitrary.  This
    function counts the number of possible measurement orderings given we have already
    chosen which targets are observed, how many births occur, and how many clutter 
    measurements are present.
    ]
    
    Inputs:
    - M: the number of measurements
    - T: the number of observed targets
    - b: the number of birth associations
    - c: the number of clutter associations

    This must be true: M = T+b+c

    Output:
    - combinations: the number of measurement orderings as a float. The value is:
        combinations = nCr(M, T)*math.factorial(T)*nCr(M-T, b)

    """
    assert(M == T + b + c)
    combinations = nCr(M, T)*math.factorial(T)*nCr(M-T, b)
    return float(combinations)


def combine_arbitrary_number_measurements_4d(blocked_cov_inv, meas_noise_mean, detection_group, det_groups_as_detObjs=False):
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

    -detection_group: dictionary of detections to combine, key='det_name', value=detObject

    """
    meas_count = len(detection_group) #number of associated measurements

    #dictionary containing all measurements in appropriately formatted numpy arrays
    reformatted_zs = {}
    for det_name, det in detection_group.iteritems():
        if det_groups_as_detObjs:
            cur_z = np.array([det.x - meas_noise_mean[det_name][0], 
                              det.y - meas_noise_mean[det_name][1],
                              det.width - meas_noise_mean[det_name][2],
                              det.height - meas_noise_mean[det_name][3]])

        else:
            cur_z = np.array([det[0] - meas_noise_mean[det_name][0], 
                              det[1] - meas_noise_mean[det_name][1],
                              det[2] - meas_noise_mean[det_name][2],
                              det[3] - meas_noise_mean[det_name][3]])
        reformatted_zs[det_name] = cur_z
    A = 0
    b = 0
    for det_name1, det in reformatted_zs.iteritems():
        for det_name2, ignore_me_det in detection_group.iteritems():
            A += blocked_cov_inv[(det_name1, det_name2)]
            b += np.dot(det, blocked_cov_inv[(det_name1, det_name2)])
    combined_meas_mean = np.dot(inv(A), b.transpose())
    combined_covariance = inv(A)
    assert(combined_meas_mean.shape == (4,)), (meas_count, detection_group)
    return (combined_meas_mean.flatten(), combined_covariance)




def get_assoc_prior(particle, meas_groups, meas_grp_associations, params, meas_counts_by_source, log, cur_time):
    """
    Inputs:
    - particle: type Particle
    - log: Bool, True return log probability, False return actual probability
    - living_target_count: number of living counts, measurement associations that correspond to association
        with a target will be in the range [0, living_target_count)
    - meas_counts_by_source: a list containing the number of measurements detected by each source
    """

    living_target_count = particle.targets.living_count
    #count the number of unique target associations
    unique_assoc = set(meas_grp_associations)
    if(living_target_count in unique_assoc):
        unique_assoc.remove(living_target_count)
    if((-1) in unique_assoc):
        unique_assoc.remove((-1))

    #the number of targets we observed on this time instance
    observed_target_count = len(unique_assoc)
    #the number of targets we don't observe on this time instance
    #but are still alive on this time instance
    unobserved_target_count = living_target_count - observed_target_count


    if log: #return log prior
        birth_count_by_group = defaultdict(int) #key = measurement source group, value = count of births detected by this group of sources
        clutter_count_by_group = defaultdict(int) #key = measurement source group, value = count of clutter detected by this group of sources

        if params.SPEC['condition_emission_prior_img_feat']==True:
            frame_idx = int(round(cur_time/DEFAULT_TIME_STEP))
            log_prior = ln_zero_approx(get_all_targets_emission_prior_conditionImgFt(particle=particle, meas_groups=meas_groups, \
                    measurement_assoc=meas_grp_associations,frame_idx=frame_idx, params=params))

        else:
            assert(params.SPEC['condition_emission_prior_img_feat']==False)            
            log_prior = unobserved_target_count*ln_zero_approx(params.target_groupEmission_priors[ImmutableSet([])])
        
        for meas_grp_idx, meas_grp_assoc in enumerate(meas_grp_associations):
            #get the names of detection sources in this group
            group_det_names = []
            for det_name, det in meas_groups[meas_grp_idx].iteritems():
                group_det_names.append(det_name)
            det_names_set = ImmutableSet(group_det_names)          
      
            if params.SPEC['condition_emission_prior_img_feat']==False and \
               meas_grp_assoc>=0 and meas_grp_assoc < living_target_count: #target association      
                log_prior += ln_zero_approx(params.target_groupEmission_priors[det_names_set])
            elif meas_grp_assoc == -1: #clutter
                log_prior += ln_zero_approx(params.clutter_group_prior(det_names_set))
                clutter_count_by_group[det_names_set] += 1
            else: #birth
                assert(meas_grp_assoc == living_target_count or params.SPEC['condition_emission_prior_img_feat']==True), (meas_grp_assoc, living_target_count)
                log_prior += ln_zero_approx(params.birth_group_prior(det_names_set))
                birth_count_by_group[det_names_set] += 1

        birth_count = meas_grp_associations.count(living_target_count)
        log_prior += ln_zero_approx(params.birth_groupCount_prior(birth_count))

        clutter_count = meas_grp_associations.count(-1)
        log_prior += ln_zero_approx(params.clutter_groupCount_prior(clutter_count))

        if params.SPEC['scale_prior_by_meas_orderings'] == 'count_multi_src_orderings':
            number_association_orderings = count_association_orderings(meas_counts_by_source, birth_count_by_group, clutter_count_by_group)        
            log_prior -= ln_zero_approx(number_association_orderings)

        return log_prior

    else: #return actual prior
        birth_count_by_group = defaultdict(int) #key = measurement source group, value = count of births detected by this group of sources
        clutter_count_by_group = defaultdict(int) #key = measurement source group, value = count of clutter detected by this group of sources

        if params.SPEC['condition_emission_prior_img_feat']==True:
            frame_idx = int(round(cur_time/DEFAULT_TIME_STEP))
            prior = get_all_targets_emission_prior_conditionImgFt(particle=particle, meas_groups=meas_groups, \
                    measurement_assoc=meas_grp_associations,frame_idx=frame_idx, params=params)
        else:
            assert(params.SPEC['condition_emission_prior_img_feat']==False)
            prior = params.target_groupEmission_priors[ImmutableSet([])]**unobserved_target_count
        
        for meas_grp_idx, meas_grp_assoc in enumerate(meas_grp_associations):
            #get the names of detection sources in this group
            group_det_names = []
            for det_name, det in meas_groups[meas_grp_idx].iteritems():
                group_det_names.append(det_name)
            det_names_set = ImmutableSet(group_det_names)          
      
            if params.SPEC['condition_emission_prior_img_feat']==False and \
               meas_grp_assoc>=0 and meas_grp_assoc < living_target_count: #target association
                prior *= params.target_groupEmission_priors[det_names_set]
            elif meas_grp_assoc == -1:#clutter
                prior *= params.clutter_group_prior(det_names_set)
                clutter_count_by_group[det_names_set] += 1
            else: #birth
                assert(meas_grp_assoc == living_target_count or params.SPEC['condition_emission_prior_img_feat']==True), (meas_grp_assoc, living_target_count)
                prior *= params.birth_group_prior(det_names_set)
                birth_count_by_group[det_names_set] += 1                

        birth_count = meas_grp_associations.count(living_target_count)
        prior *= params.birth_groupCount_prior(birth_count)

        clutter_count = meas_grp_associations.count(-1)
        prior *= params.clutter_groupCount_prior(clutter_count)

        if params.SPEC['scale_prior_by_meas_orderings'] == 'count_multi_src_orderings':
            number_association_orderings = count_association_orderings(meas_counts_by_source, birth_count_by_group, clutter_count_by_group)        
            prior /= number_association_orderings

        return prior

def get_assoc_prior_prev(living_target_indices, total_target_count, number_measurements, 
             measurement_associations, measurement_scores, params,\
             meas_source_index):
    """
    Calculate the prior probability of the observed number of measurements and their assigned associations
    given all past measurements, their associations, and living targets (particularly important, we are 
    given the number of targets currently alive). That is, calculate:
    p(c_k, #y_k | e_1:k-1, c_1:k-1, y_1:k-1, #y_1:k-1)




    Input: 
    - living_target_indices: a list of indices of targets from last time instance that are still alive
    - total_target_count: the number of living targets on the previous time instace
    - number_measurements: the number of measurements on this time instance
    - measurement_associations: a list of association values for each measurement. Each association has the value
        of a living target index (index from last time instance), target birth (total_target_count), 
        or clutter (-1)
    -p_target_emission: the probability that a target will emit a measurement on a 
        time instance (the same for all targets and time instances)
    -birth_count_prior: a probability distribution, specified as a list, such that
        birth_count_prior[i] = the probability of i births during any time instance
    -clutter_count_prior: a probability distribution, specified as a list, such that
        clutter_count_prior[i] = the probability of i clutter measurements during 
        any time instance
    """

    assert(len(measurement_associations) == number_measurements), (number_measurements, len(measurement_associations), measurement_associations)
    #number of targets from the last time instance that are still alive
    living_target_count = len(living_target_indices)
    #numnber of targets from the last time instance that died
    dead_target_count = total_target_count - living_target_count

    #count the number of unique target associations
    unique_assoc = set(measurement_associations)
    if(total_target_count in unique_assoc):
        unique_assoc.remove(total_target_count)
    if((-1) in unique_assoc):
        unique_assoc.remove((-1))

    #the number of targets we observed on this time instance
    observed_target_count = len(unique_assoc)

    #the number of target measurements by measurement score
    meas_counts_by_score = [0 for i in range(len(params.score_intervals[meas_source_index]))]
    for i in range(len(measurement_associations)):
        if measurement_associations[i] != -1 and measurement_associations[i] != total_target_count:
            index = params.get_score_index(measurement_scores[i], meas_source_index)
            meas_counts_by_score[index] += 1

    #the number of targets we don't observe on this time instance
    #but are still alive on this time instance
    unobserved_target_count = living_target_count - observed_target_count
    #the number of new targets born on this time instance
    birth_count = measurement_associations.count(total_target_count)
    birth_counts_by_score = [0 for i in range(len(params.score_intervals[meas_source_index]))]
    for i in range(len(measurement_associations)):
        if measurement_associations[i] == total_target_count:
            index = params.get_score_index(measurement_scores[i], meas_source_index)
            birth_counts_by_score[index] += 1
    #the number of clutter measurements on this time instance
    clutter_count = measurement_associations.count(-1)
    clutter_counts_by_score = [0 for i in range(len(params.score_intervals[meas_source_index]))]
    for i in range(len(measurement_associations)):
        if measurement_associations[i] == -1:
            index = params.get_score_index(measurement_scores[i], meas_source_index)
            clutter_counts_by_score[index] += 1

    assert(observed_target_count + birth_count + clutter_count == number_measurements),\
        (number_measurements, observed_target_count, birth_count, clutter_count, \
        total_target_count, measurement_associations)


    params.check_counts(clutter_counts_by_score, birth_counts_by_score, meas_source_index)

    #the prior probability of this number of measurements with these associations
    if params.scale_prior_by_meas_orderings == 'original':
        p_target_does_not_emit = 1.0 - sum(params.target_emission_probs[meas_source_index])
        assoc_prior = (p_target_does_not_emit)**(unobserved_target_count) \
                      /count_meas_orderings(number_measurements, observed_target_count, \
                                            birth_count, clutter_count)
        for i in range(len(params.score_intervals[meas_source_index])):
            assoc_prior *= params.target_emission_probs[meas_source_index][i]**(meas_counts_by_score[i]) \
                              *params.birth_probabilities[meas_source_index][i][birth_counts_by_score[i]] \
                              *params.clutter_probabilities[meas_source_index][i][clutter_counts_by_score[i]]
    elif params.scale_prior_by_meas_orderings == 'corrected_with_score_intervals':
        p_target_does_not_emit = 1.0 - sum(params.target_emission_probs[meas_source_index])
        assoc_prior = (p_target_does_not_emit)**(unobserved_target_count)
        for i in range(len(params.score_intervals[meas_source_index])):
            #The number of measurements in the current score interval associatd with a target
            cur_score_T = meas_counts_by_score[i]
            #The number of measurements in the current score interval associatd with a birth
            cur_score_B = birth_counts_by_score[i]
            #The number of measurements in the current score interval associatd with clutter
            cur_score_C = clutter_counts_by_score[i]
            #The total number of measurements in the current score interval
            cur_score_M = cur_score_T + cur_score_B + cur_score_C

            assoc_prior *= params.target_emission_probs[meas_source_index][i]**(cur_score_T) \
                              *params.birth_probabilities[meas_source_index][i][cur_score_B] \
                              *params.clutter_probabilities[meas_source_index][i][cur_score_C] \
                              /count_meas_orderings(cur_score_M, cur_score_T, \
                                            cur_score_B, cur_score_C)
    elif params.scale_prior_by_meas_orderings == 'ignore_meas_orderings':
        p_target_does_not_emit = 1.0 - sum(params.target_emission_probs[meas_source_index])
        assoc_prior = (p_target_does_not_emit)**(unobserved_target_count)
        for i in range(len(params.score_intervals[meas_source_index])):
            assoc_prior *= params.target_emission_probs[meas_source_index][i]**(meas_counts_by_score[i]) \
                              *params.birth_probabilities[meas_source_index][i][birth_counts_by_score[i]] \
                              *params.clutter_probabilities[meas_source_index][i][clutter_counts_by_score[i]]
    else:
        raise ValueError('Invalid params.scale_prior_by_meas_orderings value: %s' % params.scale_prior_by_meas_orderings)

    #####TESTING
    meas_orderings = count_meas_orderings(number_measurements, observed_target_count, \
                                        birth_count, clutter_count)

    for i in range(len(params.score_intervals[meas_source_index])):
        assert(params.target_emission_probs[meas_source_index][i]**(meas_counts_by_score[i])!=0), params.target_emission_probs[meas_source_index][i]**(meas_counts_by_score[i])
        assert(params.birth_probabilities[meas_source_index][i][birth_counts_by_score[i]] != 0), (birth_counts_by_score[i], i, params.birth_probabilities[meas_source_index][i])
        assert(params.clutter_probabilities[meas_source_index][i][clutter_counts_by_score[i]] != 0), (clutter_counts_by_score[i], i, params.clutter_probabilities[meas_source_index][i])

    #####DONE TESTING

    return assoc_prior

def birth_clutter_likelihood(detection_group, params, likelihood_type):
    """
    Inputs:
    - detection_group: dictionary of associated detections with key='det_name' and
        value=detection where detection is a numpy array of [x,y,width,height]
    - params: type Parameters, we will use posOnly_covariance_blocks  where 
        posOnly_covariance_blocks[(det_name1, det_name2)] = posOnly_cov_block_12
    - likelihood_type: string, 'clutter' or 'birth'.  Use covariance of measurements with ground
        truth objects for 'birth' and clutter objects for 'clutter'

    Outputs:
    - likelihood: float, the likelihood that this group of detections
        was produced by a clutter or birth object.
    """

    #if we only have 1 detection in the group the return value is 1.0 and we can skip the
    #work.  Also, seemed to be getting a bug calculating below:
    # File "/atlas/u/jkuck/rbpf_fireworks/rbpf_sampling_manyMeasSrcs.py", line 1269, in birth_clutter_likelihood
    #   likelihood *= math.exp(-.5*(A - B))
    # OverflowError: math range error

    if len(detection_group) == 1:
        return 1.0

    assert(likelihood_type in ['clutter', 'birth'])
    #number of dimensions in measurement space
    d = params.posOnly_covariance_blocks[params.posOnly_covariance_blocks.keys()[0]].shape[0]
    #number of measurements in the group
    n = len(detection_group)
    likelihood = (2*math.pi)**(-.5*(n-1)*d)
#    if len(detection_group) > 1:
#        print "likelihood1 =", likelihood

    #calculate the product over all detections of the determinant of the inverse
    #of the detection's measurement noise covariance matrix
    prod_of_determinants = 1.0
    for (det_name, det) in detection_group.iteritems():
        if likelihood_type == 'birth':
            cur_cov = params.posOnly_covariance_blocks[(det_name, det_name)]
        else:
            cur_cov = params.clutter_posOnly_covariance_blocks[(det_name, det_name)]
        cur_cov_inv_det = numpy.linalg.det(inv(cur_cov))
        prod_of_determinants *= cur_cov_inv_det

    #calculate the determinant of the sum over all detections' inverse measurement
    #noise covariance matrices
    determinant_of_sum = 0.0
    for (det_name, det) in detection_group.iteritems():
        if likelihood_type == 'birth':
            cur_cov = params.posOnly_covariance_blocks[(det_name, det_name)]
        else:
            cur_cov = params.clutter_posOnly_covariance_blocks[(det_name, det_name)]
        determinant_of_sum += inv(cur_cov)
    determinant_of_sum = numpy.linalg.det(determinant_of_sum)

    likelihood *= math.sqrt(prod_of_determinants/determinant_of_sum)
#    if len(detection_group) > 1:
#        print "likelihood2 =", likelihood


    #calculate terms in the likelihood's exponent
    A = 0.0
    sum_cInv_pos = 0.0
    sum_cInv = 0.0    
    for (det_name, det) in detection_group.iteritems():
        if likelihood_type == 'birth':
            cur_cov = params.posOnly_covariance_blocks[(det_name, det_name)]
        else:
            cur_cov = params.clutter_posOnly_covariance_blocks[(det_name, det_name)]
        det_pos = np.array([[det[0]],
                            [det[1]]])
        cInv_pos = np.dot(inv(cur_cov), det_pos)
        A += np.dot(np.dot(cInv_pos.T, cur_cov), cInv_pos)
        sum_cInv_pos += cInv_pos
        sum_cInv += inv(cur_cov)        

    B = np.dot(np.dot(sum_cInv_pos.T, inv(sum_cInv)), sum_cInv_pos)
#    if len(detection_group) > 1:
#        print "A=", A
#        print "B=", B

    likelihood *= math.exp(-.5*(A - B))

#    print "likelihood3:", likelihood
    return likelihood

def get_likelihood(particle, meas_groups, total_target_count,
                   measurement_associations, params, log):
    """
    - log: Bool, True return log probability, False return actual probability


    """
    if not log:
        likelihood = 1.0
        assert(len(measurement_associations) == len(meas_groups))
        for meas_index, meas_association in enumerate(measurement_associations):
            if(meas_association == total_target_count): #birth
                if params.SPEC['birth_clutter_likelihood'] == 'const1':
                    likelihood *= params.p_birth_likelihood**len(meas_groups[meas_index])
                elif params.SPEC['birth_clutter_likelihood'] == 'const2':
                    likelihood *= params.p_birth_likelihood
                elif params.SPEC['birth_clutter_likelihood'] == 'aprox1':
                    likelihood *= birth_clutter_likelihood(meas_groups[meas_index], params, 'birth')*params.p_birth_likelihood
                else:
                    print "Invalid params.SPEC['birth_clutter_likelihood']"
                    sys.exit(1);       
#                assert(likelihood != 0.0), (likelihood, params.SPEC['birth_clutter_likelihood'], 'birth')
         
            elif(meas_association == -1): #clutter
                if params.SPEC['birth_clutter_likelihood'] == 'const1':
                    likelihood *= params.p_clutter_likelihood**len(meas_groups[meas_index])
                elif params.SPEC['birth_clutter_likelihood'] == 'const2':
                    likelihood *= params.p_clutter_likelihood
                elif params.SPEC['birth_clutter_likelihood'] == 'aprox1':
                    likelihood *= birth_clutter_likelihood(meas_groups[meas_index], params, 'clutter')*params.p_clutter_likelihood
                else:
                    print "Invalid params.SPEC['birth_clutter_likelihood']"
                    sys.exit(1);               
#                assert(likelihood != 0.0), (likelihood, params.SPEC['birth_clutter_likelihood'], 'clutter')

            else:
                assert(meas_association >= 0 and meas_association < total_target_count), (meas_association, total_target_count)
                target_likelihood = memoized_assoc_likelihood(particle, meas_groups[meas_index], meas_association, params)
                likelihood *= target_likelihood
#                assert(likelihood != 0.0), (likelihood, params.SPEC['birth_clutter_likelihood'], 'target', target_likelihood)
#        assert(likelihood != 0.0), (likelihood)

        return likelihood

    else: #return log likelihood
        log_likelihood = 0.0
        assert(len(measurement_associations) == len(meas_groups))
        for meas_index, meas_association in enumerate(measurement_associations):
            cur_likelihood = -1
            if(meas_association == total_target_count): #birth
                if params.SPEC['birth_clutter_likelihood'] == 'const1':
                    cur_likelihood = params.p_birth_likelihood**len(meas_groups[meas_index])
                elif params.SPEC['birth_clutter_likelihood'] == 'const2':
                    cur_likelihood = params.p_birth_likelihood
                elif params.SPEC['birth_clutter_likelihood'] == 'aprox1':
                    cur_likelihood = birth_clutter_likelihood(meas_groups[meas_index], params, 'birth')*params.p_birth_likelihood
                else:
                    print "Invalid params.SPEC['birth_clutter_likelihood']"
                    sys.exit(1);       
                
            elif(meas_association == -1): #clutter
                if params.SPEC['birth_clutter_likelihood'] == 'const1':
                    cur_likelihood = params.p_clutter_likelihood**len(meas_groups[meas_index])
                elif params.SPEC['birth_clutter_likelihood'] == 'const2':
                    cur_likelihood = params.p_clutter_likelihood
                elif params.SPEC['birth_clutter_likelihood'] == 'aprox1':
                    cur_likelihood = birth_clutter_likelihood(meas_groups[meas_index], params, 'clutter')*params.p_clutter_likelihood
                else:
                    print "Invalid params.SPEC['birth_clutter_likelihood']"
                    sys.exit(1);               

            else:
                assert(meas_association >= 0 and meas_association < total_target_count), (meas_association, total_target_count)
                cur_likelihood = memoized_assoc_likelihood(particle, meas_groups[meas_index], meas_association, params)

#            assert(cur_likelihood > 0.0), (cur_likelihood, log_likelihood, params.SPEC['birth_clutter_likelihood'], 'birth')
            if cur_likelihood > 0.0:
                log_likelihood += ln_zero_approx(cur_likelihood)
            else:#use very small log probability
                log_likelihood -= 500

        return log_likelihood

def memoized_assoc_likelihood(particle, detection_group, target_index, params):
    """
        LSVM and regionlets produced two measurements with the same locations (centers), so using the 
        meas_source_index as part of the key is (sort of) necessary.  Currently also using the score_index, 
        could possibly be removed (not sure if this would improve speed).

        Currently saving more in the value than necessary (from debugging), can eliminate to improve
        performance (possibly noticable)

    Inputs:
    - params: type Parameters, gives prior probabilities and other parameters we are using

    """

#    if((str(detection_group), target_index) in particle.assoc_likelihood_cache):
#        (assoc_likelihood) = particle.assoc_likelihood_cache[(str(detection_group), target_index)]
#        return assoc_likelihood
#    else: #likelihood not cached
    if True:
        target = particle.targets.living_targets[target_index]
        target_cov = np.dot(np.dot(params.H, target.P), params.H.T)
        assert(target.x.shape == (4, 1))

        state_mean_meas_space = np.dot(params.H, target.x)
        state_mean_meas_space = np.squeeze(state_mean_meas_space)



        #get list of detection names present in our detection group
        dets_present = []
        for det_name, detection in detection_group.iteritems():
            dets_present.append(det_name)
        # create array of all detection positions in the group
        all_det_loc = np.zeros(2*len(detection_group))
        # repeat the target location to map it to the #detections * 2 dimension space
        target_loc_repeated = np.zeros(2*len(detection_group))
        for idx, det_name in enumerate(dets_present):
            all_det_loc[idx*2] = detection_group[det_name][0]
            all_det_loc[idx*2+1] = detection_group[det_name][1]

            target_loc_repeated[idx*2] = state_mean_meas_space[0]
            target_loc_repeated[idx*2+1] = state_mean_meas_space[1]


        complete_covariance = np.zeros((2*len(detection_group), 2*len(detection_group)))
        for idx1, det_name1 in enumerate(dets_present):
            for idx2, det_name2 in enumerate(dets_present):
                complete_covariance[idx1*2:(idx1+1)*2,idx2*2:(idx2+1)*2] = params.posOnly_covariance_blocks[(det_name1, det_name2)] + target_cov


        if params.USE_PYTHON_GAUSSIAN:        
            distribution = multivariate_normal(mean=target_loc_repeated, cov=complete_covariance)
            assoc_likelihood = distribution.pdf(all_det_loc)
        else:
            S_det = numpy.linalg.det(complete_covariance)
            S_inv = inv(complete_covariance)
            assert(S_det > 0), S_det
            LIKELIHOOD_DISTR_NORM = 1.0/(math.sqrt(S_det)*(2*math.pi)**(len(target_loc_repeated)/2))

            assert(LIKELIHOOD_DISTR_NORM!=0.0), (S_det, complete_covariance, len(target_loc_repeated))

            offset = all_det_loc - target_loc_repeated
            a = -.5*np.dot(np.dot(offset, S_inv), offset)
            assoc_likelihood = LIKELIHOOD_DISTR_NORM*math.exp(a)

#            if assoc_likelihood == 0.0:
#                print "about to crash, assoc_likelihood = 0"
#                print "the target at position ", state_mean_meas_space
#                print "With width: ", target.width, " and height:", target.height
#                print "was associated with this measurement group:"
#                print detection_group
#            assert(assoc_likelihood != 0.0), (a, offset, S_inv)

#        distribution = multivariate_normal(mean=target_loc_repeated, cov=complete_covariance)
#        assoc_likelihood_compare = distribution.pdf(all_det_loc)
#
#        S_det = numpy.linalg.det(complete_covariance)
#        S_inv = inv(complete_covariance)
#        assert(S_det > 0), S_det
#        LIKELIHOOD_DISTR_NORM = 1.0/(math.sqrt(S_det)*(2*math.pi)**(len(target_loc_repeated)/2))
#        offset = all_det_loc - target_loc_repeated
#        a = -.5*np.dot(np.dot(offset, S_inv), offset)
#        assoc_likelihood = LIKELIHOOD_DISTR_NORM*math.exp(a)
#
#        assert(abs(assoc_likelihood_compare - assoc_likelihood) < .0000001), (assoc_likelihood, assoc_likelihood_compare)


#        if params.USE_PYTHON_GAUSSIAN:
#            distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
#            assoc_likelihood = distribution.pdf(measurement)
#        else:
#            S_det = S[0][0]*S[1][1] - S[0][1]*S[1][0] # a little faster
#            S_inv = inv(S)
#            assert(S_det > 0), S_det
#            LIKELIHOOD_DISTR_NORM = 1.0/math.sqrt((2*math.pi)**2*S_det)
#
#            offset = measurement - state_mean_meas_space
#            a = -.5*np.dot(np.dot(offset, S_inv), offset)
#            assoc_likelihood = LIKELIHOOD_DISTR_NORM*math.exp(a)
#



#        particle.assoc_likelihood_cache[(str(detection_group), target_index)] = (assoc_likelihood)
        return assoc_likelihood


