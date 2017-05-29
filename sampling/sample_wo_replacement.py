import numpy as np
import copy
import cProfile
import math
import time
from itertools import product
#p = [.5, .3, .1, .05, .03, .02]
p = np.random.random(100)
p /= np.sum(p)
print sum(p)

prop_probs = []

###########for i in range(len(p)):
###########    print 'p(x_i) =', p[i]
###########    cur_prop_prob = 1.0
###########    for j in range(len(p)):
###########        if(j != i):
###########            temp = 1.0
###########            for k in range(len(p)):
###########                if(k != i and k != j):
###########                    temp += p[k]/(1 - p[j] - p[k])
###########
###########            cur_prop_prob += (p[j]/(1-p[j]))*temp
###########    cur_prop_prob*=p[i]
###########    prop_probs.append(cur_prop_prob)
###########
###########print prop_probs
###########prop_probs = np.asarray(prop_probs)

#prop_probs /= np.sum(prop_probs)
#print prop_probs

#for i in range(len(p)):
#   for j in range(i+1, len(p)):
#       for k in range(j+1, len(p)):
#           print 'sampled', i, ',', j, 'and', k
#           print 'weight samples by normalized probs, p(i) =', p[i]/(p[i]+p[j]+p[k]), 'p(j) =', p[j]/(p[i]+p[j]+p[k]), 'and p(k) =', p[k]/(p[i]+p[j]+p[k])
#           print 'weight samples by prob/prop_prob, p(i) =', p[i]/prop_probs[i], 'p(j) =', p[j]/prop_probs[j], 'and p(k) =', p[k]/prop_probs[k]

def rec_helper(p, init_denom, indices_to_avoid, num_samples):
    assert(len(indices_to_avoid) <= num_samples - 1)
    if len(indices_to_avoid) == num_samples - 1:
        ret_val = 1.0
        for cur_idx in range(len(p)):
            if not cur_idx in indices_to_avoid:
                ret_val += p[cur_idx]/(init_denom-p[cur_idx])
        return ret_val
    else:
        ret_val = 1.0
        for cur_idx in range(len(p)):
            if not cur_idx in indices_to_avoid:
                new_denom = init_denom-p[cur_idx]               
                cur_val = p[cur_idx]/new_denom
                new_indices_to_avoid = copy.copy(indices_to_avoid) #shallow copy!
                new_indices_to_avoid.append(cur_idx)
                cur_val *= rec_helper(p,new_denom,new_indices_to_avoid,num_samples)
                ret_val += cur_val
        return ret_val

def calc_prop_prob(p, idx, num_samples):
    '''
    Consider sampling a set of elements from the probability distribution
    p without replacement.  This function computes the probability that
    a particular element of p is in this set, when the set size (or number of
    elements that are sampled) is num_samples.

    Inputs:
    - p: numpy array, the probability distribution samples were drawn from
    - idx: integer in [0, len(p) - 1], index in p of a particular element 
    - num_samples: the number of samples we draw without replacement from p

    Outputs:
    - marginal_prob: Marginal probability that the element in p represented
        by idx will be in the set of sampled elements.  We are marginalizing
        over every ordering of the sampled elements and which elements are in
        the sampled set (other than idx).
    '''
    assert(num_samples >= 1)
    ret_val = p[idx]
    if num_samples > 1:
        ret_val *= rec_helper(p=p, init_denom=1.0, indices_to_avoid=[idx], num_samples=num_samples)
    return ret_val

def calc_prop_prob_by_sampling(p, num_samples):
    '''
    Solves same problem as calc_prop_prob, but calculate probabilities for all indices in the
    distribution p and uses sampling instead of calculating the exact probability.  
    Use for testing calc_prop_prob
    '''
    ret_probs = [0.0 for i in range(len(p))]
    NUMBER_SAMPLES_FOR_TESTING = 100000
    for itr in range(NUMBER_SAMPLES_FOR_TESTING):
        cur_samples = np.random.choice(len(p), size=(num_samples), replace=False, p=p)
        for sample_idx in cur_samples:
            ret_probs[sample_idx] += 1
    for idx in range(len(ret_probs)):
        ret_probs[idx] /= NUMBER_SAMPLES_FOR_TESTING
    return ret_probs






if __name__ == "__main__":
    l = [i for i in range(100)]
    sum = 0
    t1 = time.time()
    for ints in product(*[l,l,l]):
        sum += 1
    print sum
    t2 = time.time()
    print t2-t1, "seconds"

    sleep(5)

    NUM_SAMPLES = 5
    PROFILE = True
    idx = 1

    fac_sum = 0
    for i in range(math.factorial(10)):
        fac_sum += i
    print fac_sum

    if PROFILE:
        cProfile.runctx('calc_prop_prob(p, idx, NUM_SAMPLES)', {'p': p, 'idx': idx,
            'NUM_SAMPLES':NUM_SAMPLES,  'calc_prop_prob':calc_prop_prob}, {})

    else:
#        prop_probs2 = [calc_prop_prob(p, idx, NUM_SAMPLES) for idx in range(len(p))]
        prop_probs2 = [calc_prop_prob(p, idx, NUM_SAMPLES) for idx in range(NUM_SAMPLES)]
        print prop_probs2

        prop_probs3 = calc_prop_prob_by_sampling(p, NUM_SAMPLES)
        print prop_probs3
