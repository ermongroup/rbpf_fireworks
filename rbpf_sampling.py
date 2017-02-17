import numpy as np
from numpy.linalg import inv
import random

import math


class Parameters:
    def __init__(self, target_emission_probs, clutter_probabilities,\
                 birth_probabilities, meas_noise_cov, R_default, H,\
                 USE_PYTHON_GAUSSIAN, USE_CONSTANT_R, score_intervals,\
                 p_birth_likelihood, p_clutter_likelihood, CHECK_K_NEAREST_TARGETS,
                 K_NEAREST_TARGETS, scale_prior_by_meas_orderings,\
                 SPEC):
        '''
        Inputs:
        -  score_intervals: list of lists, where score_intervals[i] is a list
            specifying the score intervals for measurement source i.  
            score_intervals[i][j] specifies the lower bound for the jth score
            interval corresponding to measurement source i (0 indexed).
        - CHECK_K_NEAREST_TARGETS: If true only possibly associate each measurement with
            one of its K_NEAREST_TARGETS.  If false measurements may be associated
            with any target.
        '''

        self.target_emission_probs = target_emission_probs
        self.clutter_probabilities = clutter_probabilities
        self.birth_probabilities = birth_probabilities


        #print "checkthis outt..asdfasfwef"
        #print "self.clutter_probabilities", self.clutter_probabilities
        #print "self.birth_probabilitie", self.birth_probabilities


        self.meas_noise_cov = meas_noise_cov
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
        self.SPEC = SPEC


    def get_score_index(self, score, meas_source_index):
        """
        Inputs:
        - score: the score of a detection

        Output:
        - index: output the 0 indexed score interval this score falls into
        """

        index = 0
        for i in range(1, len(self.score_intervals[meas_source_index])):
            if(score > self.score_intervals[meas_source_index][i]):
                index += 1
            else:
                break
                
        assert(score > self.score_intervals[meas_source_index][index]), (score, self.score_intervals[meas_source_index][index], self.score_intervals[meas_source_index][index+1]) 
        if(index < len(self.score_intervals[meas_source_index]) - 1):
            assert(score <= self.score_intervals[meas_source_index][index+1]), (score, self.score_intervals[meas_source_index][index], self.score_intervals[meas_source_index][index+1])
        return index

    def emission_prior(self, meas_source_index, meas_score):
        score_index = self.get_score_index(meas_score, meas_source_index)
        return self.target_emission_probs[meas_source_index][score_index]

    def clutter_prior(self, meas_source_index, meas_score, clutter_count):
        '''
        The prior probability of clutter_count number of clutter measurements with score 
        given by meas_score from the measurement source with index meas_source_index
        '''    
        score_index = self.get_score_index(meas_score, meas_source_index)    
        return self.clutter_probabilities[meas_source_index][score_index][clutter_count]

    def max_clutter_count(self, meas_source_index, meas_score):
        '''
        The maximum clutter count from the specified measurement source and score
        range that has a non-zero prior.
        '''
        score_index = self.get_score_index(meas_score, meas_source_index)    
        return len(self.clutter_probabilities[meas_source_index][score_index]) - 1


    def birth_prior(self, meas_source_index, meas_score, birth_count):
        '''
        The prior probability of birth_count number of births with score given by
        meas_score from the measurement source with index meas_source_index
        '''
        score_index = self.get_score_index(meas_score, meas_source_index)    
        if self.SPEC['set_birth_clutter_prop_equal']:
            return self.clutter_probabilities[meas_source_index][score_index][birth_count]
        else:
            return self.birth_probabilities[meas_source_index][score_index][birth_count]


    def max_birth_count(self, meas_source_index, meas_score):
        '''
        The maximum birth count from the specified measurement source and score
        range that has a non-zero prior.
        '''
        score_index = self.get_score_index(meas_score, meas_source_index)    
        return len(self.birth_probabilities[meas_source_index][score_index]) - 1

    def check_counts(self, clutter_counts_by_score, birth_counts_by_score, meas_source_index):
        assert(len(clutter_counts_by_score) == len(birth_counts_by_score))
        assert(len(clutter_counts_by_score) == len(self.clutter_probabilities[meas_source_index]))

        for i in range(len(clutter_counts_by_score)):
            assert(0 <= clutter_counts_by_score[i] and clutter_counts_by_score[i] <= len(self.clutter_probabilities[meas_source_index][i]) - 1)
            assert(0 <= birth_counts_by_score[i] and birth_counts_by_score[i] <= len(self.birth_probabilities[meas_source_index][i]) - 1), (birth_counts_by_score[i], len(self.birth_probabilities[meas_source_index][i]) - 1, self.birth_probabilities[meas_source_index][i])


    def get_R(self, meas_source_index, meas_score):
        if self.USE_CONSTANT_R:
            return self.R_default
        else:
            score_index = self.get_score_index(meas_score, meas_source_index)    
            return self.meas_noise_cov[meas_source_index][score_index]

def sample_and_reweight(particle, measurement_lists, \
    cur_time, measurement_scores, params):
    """
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle
    - measurement_lists: a list where measurement_lists[i] is a list of all measurements from the current
        time instance from the ith measurement source (i.e. different object detection algorithms
        or different sensors)
    - measurement_scores: a list where measurement_scores[i] is a list containing scores for every measurement in
        measurement_list[i]
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - measurement_associations: A list where measurement_associations[i] is a list of association values
        for each measurements in measurement_lists[i].  Association values correspond to:
        measurement_associations[i][j] = -1 -> measurement is clutter
        measurement_associations[i][j] = particle.targets.living_count -> measurement is a new target
        measurement_associations[i][j] in range [0, particle.targets.living_count-1] -> measurement is of
            particle.targets.living_targets[measurement_associations[i][j]]

    - imprt_re_weight: After processing this measurement the particle's
        importance weight will be:
        new_importance_weight = old_importance_weight * imprt_re_weight
    - targets_to_kill: a list containing the indices of targets that should be killed, beginning
        with the smallest index in increasing order, e.g. [0, 4, 6, 33]
    """

    #get death probabilities for each target in a numpy array
    num_targs = particle.targets.living_count
    p_target_deaths = []
    for target in particle.targets.living_targets:
        p_target_deaths.append(target.death_prob)
        assert(p_target_deaths[len(p_target_deaths) - 1] >= 0 and p_target_deaths[len(p_target_deaths) - 1] <= 1)


    (targets_to_kill, measurement_associations, proposal_probability, unassociated_target_death_probs) = \
        sample_meas_assoc_and_death(particle, measurement_lists, particle.targets.living_count, p_target_deaths, \
                                    cur_time, measurement_scores, params)



    living_target_indices = []
    for i in range(particle.targets.living_count):
        if(not i in targets_to_kill):
            living_target_indices.append(i)

    exact_probability = 1.0
    for meas_source_index in range(len(measurement_lists)):
        cur_likelihood = get_likelihood(particle, meas_source_index, measurement_lists[meas_source_index], \
            particle.targets.living_count, measurement_associations[meas_source_index],\
            measurement_scores[meas_source_index], params)
        cur_assoc_prior = get_assoc_prior(living_target_indices, particle.targets.living_count, len(measurement_lists[meas_source_index]), 
                               measurement_associations[meas_source_index], measurement_scores[meas_source_index], params, meas_source_index)
#        #print "meas_source_index =", meas_source_index, "cur_likelihood =", cur_likelihood, "cur_assoc_prior =", cur_assoc_prior
        exact_probability *= cur_likelihood * cur_assoc_prior


    death_prior = calc_death_prior(living_target_indices, p_target_deaths)
    exact_probability *= death_prior

    assert(num_targs == particle.targets.living_count)
    #double check targets_to_kill is sorted
    assert(all([targets_to_kill[i] <= targets_to_kill[i+1] for i in xrange(len(targets_to_kill)-1)]))

    imprt_re_weight = exact_probability/proposal_probability

    assert(imprt_re_weight != 0.0), (exact_probability, proposal_probability, death_prior)

    particle.likelihood_DOUBLE_CHECK_ME = exact_probability

    return (measurement_associations, targets_to_kill, imprt_re_weight)

def sample_meas_assoc_and_death(particle, measurement_lists, total_target_count, 
                           p_target_deaths, cur_time, measurement_scores, params):
    """
    Try sampling associations with each measurement sequentially
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle
    - measurement_lists: type list, measurement_lists[i] is a list of all measurements from the current
        time instance from the ith measurement source (i.e. different object detection algorithms
        or different sensors)
    - measurement_scores: type list, measurement_scores[i] is a list containing scores for every measurement in
        measurement_list[i]
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - targets_to_kill: a list of targets that have been sampled to die (not killed yet)
    - measurement_associations: type list, measurement_associations[i] is a list of associations for  
        the measurements in measurement_lists[i]
    - proposal_probability: proposal probability of the sampled deaths and associations
        
    """
    assert(len(measurement_lists) == len(measurement_scores))
    measurement_associations = []
    proposal_probability = 1.0
    for meas_source_index in range(len(measurement_lists)):
        (cur_associations, cur_proposal_prob) = associate_measurements_sequentially\
            (particle, meas_source_index, measurement_lists[meas_source_index], \
             total_target_count, p_target_deaths, measurement_scores[meas_source_index],\
             params)
        measurement_associations.append(cur_associations)
        proposal_probability *= cur_proposal_prob

    assert(len(measurement_associations) == len(measurement_lists))

############################################################################################################
    #sample target deaths from unassociated targets
    unassociated_targets = []
    unassociated_target_death_probs = []

    for i in range(total_target_count):
        target_unassociated = True
        for meas_source_index in range(len(measurement_associations)):
            if (i in measurement_associations[meas_source_index]):
                target_unassociated = False
        if target_unassociated:
            unassociated_targets.append(i)
            unassociated_target_death_probs.append(p_target_deaths[i])
        else:
            unassociated_target_death_probs.append(0.0)

    (targets_to_kill, death_probability) =  \
        sample_target_deaths(particle, unassociated_targets, cur_time)

    #probability of sampling all associations
    proposal_probability *= death_probability
    assert(proposal_probability != 0.0)

    #debug
    for meas_source_index in range(len(measurement_associations)):
        for i in range(total_target_count):
            assert(measurement_associations[meas_source_index].count(i) == 0 or \
                   measurement_associations[meas_source_index].count(i) == 1), (measurement_associations[meas_source_index],  measurement_list, total_target_count, p_target_deaths)
    #done debug

    return (targets_to_kill, measurement_associations, proposal_probability, unassociated_target_death_probs)


def associate_measurements_sequentially(particle, meas_source_index, measurement_list, total_target_count, \
    p_target_deaths, measurement_scores, params):

    """
    Try sampling associations with each measurement sequentially
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - measurement_list: a list of all measurements from the current time instance
    - total_target_count: the number of living targets on the previous time instace
    - p_target_deaths: a list of length len(total_target_count) where 
        p_target_deaths[i] = the probability that target i has died between the last
        time instance and the current time instance
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Output:
    - list_of_measurement_associations: list of associations for each measurement
    - proposal_probability: proposal probability of the sampled deaths and associations
        
    """
    list_of_measurement_associations = []
    proposal_probability = 1.0

    #sample measurement associations
    birth_count = [0 for i in range(len(params.score_intervals[meas_source_index]))]
    clutter_count = [0 for i in range(len(params.score_intervals[meas_source_index]))]
    remaining_meas_count = len(measurement_list)

    def get_remaining_meas_count(cur_meas_index, cur_meas_score):
        assert(len(measurement_scores) == len(measurement_list))
        remaining_meas_count = 0
        cur_meas_score_idx = params.get_score_index(cur_meas_score, meas_source_index)
        for idx in range(cur_meas_index+1, len(measurement_list)):
            if(cur_meas_score_idx ==\
               params.get_score_index(measurement_scores[idx], meas_source_index)):
                remaining_meas_count = remaining_meas_count + 1

        return remaining_meas_count


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


    for (index, cur_meas) in enumerate(measurement_list):
        meas_score = measurement_scores[index]
        #create proposal distribution for the current measurement
        #compute target association proposal probabilities
        proposal_distribution_list = []

        if params.CHECK_K_NEAREST_TARGETS:
            targets_to_check = get_k_nearest_targets(cur_meas, params.K_NEAREST_TARGETS)
        else:
            targets_to_check = [i for i in range(total_target_count)]

#        for target_index in range(total_target_count):
        for target_index in targets_to_check:
            cur_target_likelihood = memoized_assoc_likelihood(particle, cur_meas, meas_source_index, target_index, params, meas_score)
            targ_likelihoods_summed_over_meas = 0.0

            for meas_index in range(len(measurement_list)):
                temp_score = measurement_scores[meas_index] #measurement score for the meas_index in this loop
                targ_likelihoods_summed_over_meas += memoized_assoc_likelihood(particle, measurement_list[meas_index], meas_source_index, target_index, params, temp_score)
            
            if((targ_likelihoods_summed_over_meas != 0.0) and (not target_index in list_of_measurement_associations)\
                and p_target_deaths[target_index] < 1.0):
                cur_target_prior = params.emission_prior(meas_source_index, meas_score)*cur_target_likelihood \
                                  /targ_likelihoods_summed_over_meas
            else:
                cur_target_prior = 0.0

            proposal_distribution_list.append(cur_target_likelihood*cur_target_prior)


#        score_index = params.get_score_index(meas_score, meas_source_index)
        remaining_meas_count_by_score = get_remaining_meas_count(index, meas_score)
        #compute birth association proposal probability
        cur_birth_prior = 0.0

        assert(params.get_score_index(meas_score, meas_source_index) < len(birth_count)), (params.get_score_index(meas_score, meas_source_index), len(birth_count))

        for i in range(birth_count[params.get_score_index(meas_score, meas_source_index)]+1, min(params.max_birth_count(meas_source_index, meas_score) + 1, remaining_meas_count_by_score + birth_count[params.get_score_index(meas_score, meas_source_index)] + 2)):
            cur_birth_prior += params.birth_prior(meas_source_index, meas_score, i)*(i - birth_count[params.get_score_index(meas_score, meas_source_index)])/(remaining_meas_count_by_score + 1)
            #print "cur_birth_prior =", cur_birth_prior
            #print "params.birth_prior(meas_source_index, meas_score, i) =", params.birth_prior(meas_source_index, meas_score, i)
            #print "meas_source_index =", meas_source_index
            #print "meas_score =", meas_score
            #print "(i - birth_count[params.get_score_index(meas_score, meas_source_index)])/(remaining_meas_count_by_score + 1) =", (i - birth_count[params.get_score_index(meas_score, meas_source_index)])/(remaining_meas_count_by_score + 1)
        proposal_distribution_list.append(cur_birth_prior*params.p_birth_likelihood)
#        for i in range(birth_count+1, min(len(params.birth_probabilities[meas_source_index][score_index]), remaining_meas_count_by_score + birth_count + 1)):
#            cur_birth_prior += params.birth_probabilities[meas_source_index][score_index][i]*(i - birth_count)/remaining_meas_count_by_score 
#        proposal_distribution_list.append(cur_birth_prior*params.p_birth_likelihood)

#        assert(len(params.birth_probabilities[meas_source_index][score_index]) == params.max_birth_count(meas_source_index, meas_score) + 1), (len(params.birth_probabilities[meas_source_index][score_index]), params.max_birth_count(meas_source_index, meas_score) + 1)

        #compute clutter association proposal probability
        cur_clutter_prior = 0.0
        for i in range(clutter_count[params.get_score_index(meas_score, meas_source_index)]+1, min(params.max_clutter_count(meas_source_index, meas_score) + 1, remaining_meas_count_by_score + clutter_count[params.get_score_index(meas_score, meas_source_index)] + 2)):
            cur_clutter_prior += params.clutter_prior(meas_source_index, meas_score, i)*(i - clutter_count[params.get_score_index(meas_score, meas_source_index)])/(remaining_meas_count_by_score + 1)
            #print "cur_clutter_prior =", cur_clutter_prior
            #print "params.clutter_prior(meas_source_index, meas_score, i) =", params.clutter_prior(meas_source_index, meas_score, i)            
            #print "meas_source_index =", meas_source_index
            #print "meas_score =", meas_score
            #print "(i - birth_count[params.get_score_index(meas_score, meas_source_index)])/(remaining_meas_count_by_score + 1) =", (i - birth_count[params.get_score_index(meas_score, meas_source_index)])/(remaining_meas_count_by_score + 1)

        proposal_distribution_list.append(cur_clutter_prior*params.p_clutter_likelihood)
#        for i in range(clutter_count+1, min(len(params.clutter_probabilities[meas_source_index][score_index]), remaining_meas_count_by_score + clutter_count + 1)):
#            cur_clutter_prior += params.clutter_probabilities[meas_source_index][score_index][i]*(i - clutter_count)/remaining_meas_count_by_score 
#        proposal_distribution_list.append(cur_clutter_prior*params.p_clutter_likelihood)



        #normalize the proposal distribution
        proposal_distribution = np.asarray(proposal_distribution_list)
        assert(np.sum(proposal_distribution) != 0.0), (index, remaining_meas_count, len(proposal_distribution), proposal_distribution, birth_count, clutter_count, len(measurement_list), total_target_count)



        proposal_distribution /= float(np.sum(proposal_distribution))
        if params.CHECK_K_NEAREST_TARGETS:
            proposal_length = min(params.K_NEAREST_TARGETS+2, total_target_count+2)
            assert(len(proposal_distribution) == proposal_length), (proposal_length, len(proposal_distribution))

        else:
            assert(len(proposal_distribution) == total_target_count+2), len(proposal_distribution)


        sampled_assoc_idx = np.random.choice(len(proposal_distribution),
                                                p=proposal_distribution)

        #print proposal_distribution
#        sleep(5)

        if params.CHECK_K_NEAREST_TARGETS:
            possible_target_assoc_count = min(params.K_NEAREST_TARGETS, total_target_count)
            if(sampled_assoc_idx <= possible_target_assoc_count): #target or birth association
                if(sampled_assoc_idx == possible_target_assoc_count): #birth
                    birth_count[params.get_score_index(meas_score, meas_source_index)] += 1
                    list_of_measurement_associations.append(total_target_count)
                else: #target
                    list_of_measurement_associations.append(targets_to_check[sampled_assoc_idx])

            else: #clutter association
                assert(sampled_assoc_idx == possible_target_assoc_count+1)
                list_of_measurement_associations.append(-1)
                clutter_count[params.get_score_index(meas_score, meas_source_index)] += 1

        else: #we considered association with all targets
            if(sampled_assoc_idx <= total_target_count): #target or birth association
                list_of_measurement_associations.append(sampled_assoc_idx)
                if(sampled_assoc_idx == total_target_count):
                    birth_count[params.get_score_index(meas_score, meas_source_index)] += 1
            else: #clutter association
                assert(sampled_assoc_idx == total_target_count+1)
                list_of_measurement_associations.append(-1)
                clutter_count[params.get_score_index(meas_score, meas_source_index)] += 1

        proposal_probability *= proposal_distribution[sampled_assoc_idx]

        remaining_meas_count -= 1

        assert(clutter_count[params.get_score_index(meas_score, meas_source_index)] <= params.max_clutter_count(meas_source_index, meas_score))
        assert(birth_count[params.get_score_index(meas_score, meas_source_index)] <= params.max_birth_count(meas_source_index, meas_score)), (proposal_distribution, sampled_assoc_idx, birth_count, params.max_birth_count(meas_source_index, meas_score), index, remaining_meas_count, len(proposal_distribution), birth_count, clutter_count, len(measurement_list), total_target_count)

    assert(remaining_meas_count == 0)
    return(list_of_measurement_associations, proposal_probability)


def sample_target_deaths(particle, unassociated_targets, cur_time):
    """
    Sample target deaths, given they have not been associated with a measurement, using probabilities
    learned from data.
    Also kill all targets that are offscreen.

    Inputs:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle     
    - unassociated_targets: a list of target indices that have not been associated with a measurement

    Output:
    - targets_to_kill: a list of targets that have been sampled to die (not killed yet)
    - probability_of_deaths: the probability of the sampled deaths
    """
    targets_to_kill = []
    probability_of_deaths = 1.0

    for target_idx in range(len(particle.targets.living_targets)):
        #kill offscreen targets with probability 1.0
        if(particle.targets.living_targets[target_idx].offscreen == True):
            targets_to_kill.append(target_idx)
        elif(target_idx in unassociated_targets):
            cur_death_prob = particle.targets.living_targets[target_idx].death_prob
            if(random.random() < cur_death_prob):
                targets_to_kill.append(target_idx)
                probability_of_deaths *= cur_death_prob
            else:
                probability_of_deaths *= (1 - cur_death_prob)
    return (targets_to_kill, probability_of_deaths)

def calc_death_prior(living_target_indices, p_target_deaths):
    death_prior = 1.0
    for (cur_target_index, cur_target_death_prob) in enumerate(p_target_deaths):
        if cur_target_index in living_target_indices:
            death_prior *= (1.0 - cur_target_death_prob)
            assert((1.0 - cur_target_death_prob) != 0.0), cur_target_death_prob
        else:
            death_prior *= cur_target_death_prob
            assert((cur_target_death_prob) != 0.0), cur_target_death_prob

    return death_prior

def nCr(n,r):
    return math.factorial(n) / math.factorial(r) / math.factorial(n-r)

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


def get_assoc_prior(living_target_indices, total_target_count, number_measurements, 
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


def get_likelihood(particle, meas_source_index, measurement_list, total_target_count,
                   measurement_associations, measurement_scores, params):
    """
    REDOCUMENT, BELOW INCORRECT, not including death probability now
    Calculate p(data, associations, #measurements, deaths) as:
    p(data|deaths, associations, #measurements)*p(deaths)*p(associations, #measurements|deaths)
    Input:
    - particle: type Particle, we will perform sampling and importance reweighting on this particle             
    - measurement_list: a list of all measurements from the current time instance, from the measurement
        source with index meas_source_index
    - total_target_count: the number of living targets on the previous time instace
    - measurement_associations: a list of association values for each measurement. Each association has the value
        of a living target index (index from last time instance), target birth (total_target_count), 
        or clutter (-1)
    - params: type Parameters, gives prior probabilities and other parameters we are using

    Return:
    - p(data, associations, #measurements, deaths)

    """

    likelihood = 1.0
    assert(len(measurement_associations) == len(measurement_list))
    for meas_index, meas_association in enumerate(measurement_associations):
        if(meas_association == total_target_count): #birth
            likelihood *= params.p_birth_likelihood
        elif(meas_association == -1): #clutter
            likelihood *= params.p_clutter_likelihood
        else:
            assert(meas_association >= 0 and meas_association < total_target_count), (meas_association, total_target_count)
            meas_score = measurement_scores[meas_index]
            likelihood *= memoized_assoc_likelihood(particle, measurement_list[meas_index], meas_source_index, \
                                                         meas_association, params, meas_score)

    assert(likelihood != 0.0), (likelihood)

    return likelihood

def memoized_assoc_likelihood(particle, measurement, meas_source_index, target_index, params, meas_score):
    """
        LSVM and regionlets produced two measurements with the same locations (centers), so using the 
        meas_source_index as part of the key is (sort of) necessary.  Currently also using the score_index, 
        could possibly be removed (not sure if this would improve speed).

        Currently saving more in the value than necessary (from debugging), can eliminate to improve
        performance (possibly noticable)

    Inputs:
    - params: type Parameters, gives prior probabilities and other parameters we are using

    """


    if((measurement[0], measurement[1], target_index, meas_source_index, meas_score) in particle.assoc_likelihood_cache):
        (assoc_likelihood, cached_score_index, cached_measurement, cached_meas_source_index) = particle.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, meas_score)]
        assert(cached_score_index == meas_score), (cached_score_index, meas_score, measurement, cached_measurement, target_index, meas_noise_cov, cached_meas_source_index, meas_source_index)
        assert(cached_meas_source_index == meas_source_index), (cached_score_index, meas_score, measurement, cached_measurement, target_index, meas_noise_cov, cached_meas_source_index, meas_source_index)
        return assoc_likelihood
    else: #likelihood not cached
        R = params.get_R(meas_source_index, meas_score)
        target = particle.targets.living_targets[target_index]
        S = np.dot(np.dot(params.H, target.P), params.H.T) + R
        assert(target.x.shape == (4, 1))

        state_mean_meas_space = np.dot(params.H, target.x)
        state_mean_meas_space = np.squeeze(state_mean_meas_space)


        if params.USE_PYTHON_GAUSSIAN:
            distribution = multivariate_normal(mean=state_mean_meas_space, cov=S)
            assoc_likelihood = distribution.pdf(measurement)
        else:
            S_det = S[0][0]*S[1][1] - S[0][1]*S[1][0] # a little faster
            S_inv = inv(S)
            assert(S_det > 0), S_det
            LIKELIHOOD_DISTR_NORM = 1.0/math.sqrt((2*math.pi)**2*S_det)

            offset = measurement - state_mean_meas_space
            a = -.5*np.dot(np.dot(offset, S_inv), offset)
            assoc_likelihood = LIKELIHOOD_DISTR_NORM*math.exp(a)

        particle.assoc_likelihood_cache[(measurement[0], measurement[1], target_index, meas_source_index, meas_score)] = (assoc_likelihood, meas_score, measurement, meas_source_index)
        return assoc_likelihood


