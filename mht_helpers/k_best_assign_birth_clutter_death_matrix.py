import numpy as np
import sys
import cProfile

from munkres import Munkres, print_matrix
from scipy.optimize import linear_sum_assignment
#use edited version of scipy
sys.path.insert(0, "/Users/jkuck/tracking_research/scipy/scipy/optimize")
#from _hungarian import linear_sum_assignment

from pymatgen.optimization import linear_assignment

import itertools
import math
import random
import time
from operator import itemgetter
sys.path.insert(0, "../")
from rbpf_sampling_manyMeasSrcs import convert_assignment_matrix3
from rbpf_sampling_manyMeasSrcs import convert_assignment_pairs_to_matrix3
from rbpf_sampling_manyMeasSrcs import convert_assignment_pairs_to_associations3
np.random.seed(1)
random.seed(1)
#if True, check that we don't return two assignment matrices that correspond
#to the same associations and deaths, due to filler entries in matrix
CHECK_NO_DUPLICATES = True

#if true use Munkres from munkres to solve the assignment problem
#if false use linear_sum_assignment from scipy to solve the assignment problem (generally faster)

#'pymatgen' should be fastest, significantly
#pick from ['munkres', 'scipy', 'pymatgen'], 
ASSIGNMENT_SOLVER = 'pymatgen'
DEBUG = False
DEBUG1 = False
DEBUG2 = False
PROFILE = False

#if 'delete', delete required rows, WORKS :)
#if 'fixed', keep the same size but set cost of required entry to .00000001, 
#other entries in row/col to INFEASIBLE_COST, THIS DOESN"T WORK CURRENTLY
#'copy' NOT IMPLEMENTED
REMAINING_COST_MATRIX_CONSTRUNCTION = 'delete'
#Set other entries in the same row/col as a required cell to this value, should be big
#but don't want overflow issues when further transforming the matrix, more principled number?
INFEASIBLE_COST = 9999999999999999
#This is different that the implementation in k_best_assignment in that when we are finding
#the k_best assignments for a measurement association/target death matrix, the bottom right
#portion of the matrix is filled with zeros and we do not care about getting multiple assignments
#that only differ in this bottom right corner, so we change the partition of nodes to exclude
#getting additional assignments that only differ in this way.
#
#
#References:
# [1] K. G. Murty, "Letter to the Editor--An Algorithm for Ranking all the Assignments in Order of
#     Increasing Cost," Oper. Res., vol. 16, no. May 2016, pp. 682-687, 1968.
#
# [2] I. J. Cox and M. L. Miller, "On finding ranked assignments with application to multitarget
#     tracking and motion correspondence," IEEE Trans. Aerosp. Electron. Syst., vol. 31, no. 1, pp.
#     486-489, Jan. 1995.



#BRUTE FORCE TEST ME WITH RANDOM MATRICES
def k_best_assign_mult_cost_matrices(k, cost_matrices, matrix_costs, M):
    '''
    Find the k lowest cost assignments for any of the cost matrices.  That is, the lowest cost will
    be the lowest cost assignment with costs specified by ANY of the cost matrices.  This is 
    useful for multiple hypothesis tracking, where we want to find the k lowest costs and have
    k cost matrices.  Rather than solving k k_best_assignment problems, generating k^2 costs, and
    picking the k smallest costs we will instead initialize Murty's algorithm with a set of 
    cost matrices as explained in [2] on pp. 487-488.

    The cost of an assignment is given by the assignment to the specified cost matrix PLUS the
    cost for the matrix given in matrix_costs
    Inputs:
    - k: (integer), find top k best assignments   
    - cost_matrices: (list of numpy arrays)  
    - matrix_costs: (list of floats) same length as cost_matrices.  add this to every assignment cost for the corresponding matrix
        in cost_matrices
    - M: number of measurements 

    cost_matrices have dimensions (2*M + 2*T)x(2*M + 2*T), where T = number of targets and may differ
    between cost_matrices


    Output:
    - best_assignments: (list of triplets) best_assignments[i][0] is the cost of the ith best
        assignment.  best_assignments[i][1] is the ith best assignment, which is a list of pairs
        where each pair represents an association in the assignment (1's in assignment matrix),
        best_assignments[i][2] is the index in the input cost_matrices of the cost matrix used
        for the ith best assignment
    '''
    for cur_cost_matrix in cost_matrices:
#        assert(cur_cost_matrix.shape == (2*M + 2*T, 2*M + 2*T)), (cur_cost_matrix.shape, M, T)
        assert(cur_cost_matrix.shape[0] == cur_cost_matrix.shape[1]), (cur_cost_matrix.shape, M)
        for i in range(cur_cost_matrix.shape[0]):
            for j in range(cur_cost_matrix.shape[1]):
                assert(sys.maxint > cur_cost_matrix[i][j])
    best_assignments = []
    cur_partition = []
    for (idx, cur_cost_matrix) in enumerate(cost_matrices):
        T = cur_cost_matrix.shape[0]/2 - M
        assert(cur_cost_matrix.shape == (2*M + 2*T, 2*M + 2*T)), (cur_cost_matrix.shape, M, T)        
        cur_partition.append(Node(cur_cost_matrix, [], [], idx, M, T, matrix_costs[idx]))

    for itr in range(0, k):
        if DEBUG2:
            print '-'*80
            print 'iteration:', itr
            print "best assignments:"
            for best_assign in best_assignments:
                print 'total cost =', best_assign[0], 'cost_matrix_index =', best_assign[2]
            print "partition:"
            for node in cur_partition:
                print 'matrix cost =', node.matrix_cost, ', total cost =', node.minimum_cost



        min_cost = sys.maxint
        min_cost_idx = -1
        np.set_printoptions(linewidth=300)

        if DEBUG1:
            print len(cur_partition), "nodes in partition with remaing cost matrices:"
        for cur_idx, cur_node in enumerate(cur_partition):
            if DEBUG1:
                print cur_node.minimum_cost
                print cur_node.remaining_cost_matrix   
                print       
            if cur_node.minimum_cost < min_cost:
                min_cost = cur_node.minimum_cost
                min_cost_idx = cur_idx
        if DEBUG1:            
            print '$'*80

        if DEBUG1:
            debug_total_assign_count = 0
            for cur_idx, cur_node in enumerate(cur_partition):
                rcm = cur_node.remaining_cost_matrix
                assert(rcm.shape[0] == rcm.shape[1])
                rmc_assign_count = 0
                for row in range(rcm.shape[0]):
                    if row == 0: #count excluded cells
                        excluded_cell_count = 0
                        for col in range(rcm.shape[1]):
                            if rcm[(row, col)] == sys.maxint:
                                excluded_cell_count += 1
                        rmc_assign_count = rcm.shape[0] - excluded_cell_count
                    else:
                        excluded_cell_count = 0
                        for col in range(rcm.shape[1]):
                            if rcm[(row, col)] == sys.maxint:
                                excluded_cell_count += 1
                        assert(excluded_cell_count == 0)
                rmc_assign_count *= math.factorial(rcm.shape[0] - 1)
                debug_total_assign_count += rmc_assign_count                        
            print "we've kept track of assignments:", (itr + debug_total_assign_count)
            print '&'*80
            print
            print
        if min_cost_idx != -1:
            best_assignments.append(cur_partition[min_cost_idx].get_min_cost_assignment())
            if DEBUG:
                print "iter", itr, "-"*80
                print "best assignment: ", best_assignments[-1]
            cur_partition.extend(cur_partition[min_cost_idx].partition())

            if DEBUG:
                print "all assignment costs in partition: ",
                for node in cur_partition:
                    node_min_cost_assign = node.get_min_cost_assignment()
                    print node_min_cost_assign[0],
                print
                print "required_cells: ", cur_partition[min_cost_idx].required_cells
                print "min_cost_associations: ", cur_partition[min_cost_idx].min_cost_associations
                print

            del cur_partition[min_cost_idx]
        else: #out of assignments early
            break

    if CHECK_NO_DUPLICATES:
        check_for_duplicates(best_assignments, M, cost_matrices[0])

    return best_assignments

def check_for_duplicates(best_assignments, M, cost_matrix_example):
    '''
    Check the assignments differ in entries that are meaningful
    '''
    print 'checking for duplicates'
    print(cost_matrix_example.shape, cost_matrix_example)
    unique_assignments = []
    for assignment in best_assignments:
        T = len(assignment[1])/2 - M
        assert(len(assignment[1]) == 2*M + 2*T), (len(assignment[1]), M, T)                

        (meas_grp_associations, dead_target_indices) = convert_assignment_pairs_to_associations3(assignment[1], M, T)
        
#        cur_assignment_matrix = convert_assignment_pairs_to_matrix3(assignment[1], M, T)
#        (meas_grp_associations1, dead_target_indices1) = convert_assignment_matrix3(cur_assignment_matrix, M, T)
#        assert(meas_grp_associations == meas_grp_associations1)
#        assert(dead_target_indices == dead_target_indices1)

        cur_tuple = (tuple(meas_grp_associations), tuple(dead_target_indices), assignment[0])
        assert(not cur_tuple in unique_assignments), (assignment, best_assignments, unique_assignments)
        unique_assignments.append(cur_tuple)

#def check_assignments_differ(assignA, assignB, M, T):
#    '''
#    Check the assignments differ in entries that are meaningful
#    '''
#    assignments_differ = False
#    for ()

class Node:
    def __init__(self, orig_cost_matrix, required_cells, excluded_cells, orig_cost_matrix_index, M, T, matrix_cost):
        '''
        Following the terminology used by [1], a node is defined to be a nonempty subset of possible
        assignments to a cost matrix.  Every assignment in node N is required to contain
        required_cells and exclude excluded_cells.

        Inputs:
        - orig_cost_matrix: (2d numpy array) the original cost matrix
        - required_cells: (list of pairs) where each pair represents a (zero indexed) location
            in the assignment matrix that must be a 1
        - excluded_cells: list of pairs) where each pair represents a (zero indexed) location
            in the assignment matrix that must be a 0
        - orig_cost_matrix_index: index of the cost matrix this Node is descended from, used when
            when finding the k lowest cost assignments among a group of assignment matrices
            (k_best_assign_mult_cost_matrices)
        - M: number of measurements 
        - T: number of targets 
        - matrix_cost: cost of this matrix, must be added to every assignment cost we compute
        orig_cost_matrix has dimensions (2*M + 2*T)x(2*M + 2*T) 

        '''
        self.orig_cost_matrix = np.array(orig_cost_matrix, copy=True)
        self.required_cells = required_cells[:]
        self.excluded_cells = excluded_cells[:]
        self.orig_cost_matrix_index = orig_cost_matrix_index
        self.M = M
        self.T = T
        if DEBUG:
            print "New Node:"
            print "self.required_cells:", self.required_cells 
            print "self.excluded_cells:", self.excluded_cells 

        self.matrix_cost = matrix_cost
        self.minimum_cost = matrix_cost

        if orig_cost_matrix.size > 0:
            #we will transform the cost matrix into the "remaining cost matrix" as described in [1]
            if REMAINING_COST_MATRIX_CONSTRUNCTION == 'fixed':
                self.remaining_cost_matrix = self.construct_fixed_size_remaining_cost_matrix()
            elif REMAINING_COST_MATRIX_CONSTRUNCTION == 'delete':
                self.remaining_cost_matrix = self.construct_remaining_cost_matrix()
            else:
                self.remaining_cost_matrix = self.construct_remaining_cost_matrix()

            assert((self.remaining_cost_matrix > 0).all()), self.remaining_cost_matrix
            #solve the assignment problem for the remaining cost matrix
            if ASSIGNMENT_SOLVER == 'munkres':
                hm = Munkres()
                # we get a list of (row, col) associations, or 1's in the minimum assignment matrix
                association_list = hm.compute(self.remaining_cost_matrix.tolist())
            elif ASSIGNMENT_SOLVER == 'scipy':
                row_ind, col_ind = linear_sum_assignment(self.remaining_cost_matrix)
                assert(len(row_ind) == len(col_ind))
                association_list = zip(row_ind, col_ind)
            else:
                assert(ASSIGNMENT_SOLVER == 'pymatgen')
                lin_assign = linear_assignment.LinearAssignment(self.remaining_cost_matrix)
                solution = lin_assign.solution
                association_list = zip([i for i in range(len(solution))], solution)
#                association_list = [(i, i) for i in range(orig_cost_matrix.shape[0])]

            if DEBUG:
                print "remaining cost matrix:"
                print self.remaining_cost_matrix
                print "association_list"
                print association_list

            if REMAINING_COST_MATRIX_CONSTRUNCTION == 'fixed':
                for (row,col) in association_list:
                    self.minimum_cost += np.asscalar(self.orig_cost_matrix[row][col])
            elif REMAINING_COST_MATRIX_CONSTRUNCTION == 'delete':
                #compute the minimum cost assignment for the node
                for (row,col) in association_list:
        #            print 'a', self.minimum_cost, type(self.minimum_cost)
        #            print 'b', self.remaining_cost_matrix[row][col], type(self.remaining_cost_matrix[row][col])
        #            print 'c', self.minimum_cost +self.remaining_cost_matrix[row][col], type(self.minimum_cost +self.remaining_cost_matrix[row][col])
                    #np.asscalar important for avoiding overflow problems
                    self.minimum_cost += np.asscalar(self.remaining_cost_matrix[row][col])
                for (row, col) in self.required_cells:
                    #np.asscalar important for avoiding overflow problems
                    self.minimum_cost += np.asscalar(orig_cost_matrix[row][col])

            else:
                implement_me = False
            #store the minimum cost associations with indices consistent with the original cost matrix
            if REMAINING_COST_MATRIX_CONSTRUNCTION == 'fixed':
                self.min_cost_associations = association_list
            else:
                self.min_cost_associations = self.get_orig_indices(association_list)

        else:
            self.min_cost_associations = []
        if DEBUG:
            print "New Node:"
            print "self.required_cells:", self.required_cells 
            print "self.excluded_cells:", self.excluded_cells 
            print
            print

    def get_min_cost_assignment(self):
        if REMAINING_COST_MATRIX_CONSTRUNCTION == 'fixed':
            min_cost_assignment = self.min_cost_associations
        else:
            min_cost_assignment = self.required_cells[:]
            min_cost_assignment.extend(self.min_cost_associations)
        if DEBUG:
            return (self.minimum_cost, min_cost_assignment, self.excluded_cells, self.required_cells[:], self.min_cost_associations)
        else:
            return (self.minimum_cost, min_cost_assignment, self.orig_cost_matrix_index)

    def partition(self):
        '''
        Partition this node by its minimum assignment, as described in [1]

        Output:
        - partition: a list of mutually disjoint Nodes, whose union with the minimum assignment
            of this node forms the set of possible assignments represented by this node (that are different in our assignment context)
        '''
        partition = []
        cur_required_cells = self.required_cells[:]

        if DEBUG:
            print '!'*40, 'Debug partition()', '!'*40
            print len(self.min_cost_associations) - 1

        if DEBUG2:
            print 'partition called, len(self.min_cost_associations)  - 1 =', len(self.min_cost_associations)  - 1

#        for idx in range(len(self.min_cost_associations)  - 1):
        #this seems to be corret from testing, without -1, but seems like we might have two identical associations in the partition
        #think about more some time
        for idx in range(len(self.min_cost_associations)):
            cur_assoc = self.min_cost_associations[idx]
            row_idx = cur_assoc[0]
            col_idx = cur_assoc[1]
            #only partition by cells that will result in a different assignment
            if((row_idx < self.M+2*self.T and col_idx < self.T) or (row_idx < self.M and col_idx < self.T+2*self.M)):
                cur_excluded_cells = self.excluded_cells[:]
                cur_excluded_cells.append(cur_assoc)
                if DEBUG:
                    print "idx:", idx
                    print "cur_required_cells:", cur_required_cells
                    print "cur_excluded_cells:", cur_excluded_cells
                    print "self.excluded_cells: ", self.excluded_cells
                    print "self.required_cells: ", self.required_cells
                    #check we haven't made a mistake
                    for assoc in cur_excluded_cells:
                        assert(not(assoc in cur_required_cells))
                    for i in range(len(cur_required_cells)):
                        for j in range(i+1, len(cur_required_cells)):
                            assert(cur_required_cells[i][0] != cur_required_cells[j][0] and
                                   cur_required_cells[i][1] != cur_required_cells[j][1])
                                    
                partition.append(Node(self.orig_cost_matrix, cur_required_cells, cur_excluded_cells,
                                      self.orig_cost_matrix_index, self.M, self.T, self.matrix_cost))
                cur_required_cells.append(cur_assoc)

            elif DEBUG2:
                print "no need to partition by association", cur_assoc, ', M =', self.M, ', T =', self.T

        return partition

    #transform the cost matrix into the "remaining cost matrix" as described in [1]
    def construct_remaining_cost_matrix(self):
        remaining_cost_matrix = np.array(self.orig_cost_matrix, copy=True)
      
        #replace excluded_cell locations with infinity in the remaining cost matrix
        for (row, col) in self.excluded_cells:
            remaining_cost_matrix[row][col] = sys.maxint

        rows_to_delete = []
        cols_to_delete = []
        for (row, col) in self.required_cells: #remove required rows and columns
            rows_to_delete.append(row)
            cols_to_delete.append(col)

        #create sorted lists of rows and columns to delete, where indices are sorted in increasing
        #order, e.g. [1, 4, 5, 9]
        sorted_rows_to_delete = sorted(rows_to_delete)
        sorted_cols_to_delete = sorted(cols_to_delete)

        #delete rows and cols, starting with LARGEST indices to preserve validity of smaller indices
        for row in reversed(sorted_rows_to_delete):
            remaining_cost_matrix = np.delete(remaining_cost_matrix, row, 0)
        for col in reversed(sorted_cols_to_delete):
            remaining_cost_matrix = np.delete(remaining_cost_matrix, col, 1)


        return remaining_cost_matrix

    #transform the cost matrix into the "remaining cost matrix" as described in [1]
    def construct_fixed_size_remaining_cost_matrix(self):
        remaining_cost_matrix = np.array(self.orig_cost_matrix, copy=True)
      
        #replace excluded_cell locations with infinity in the remaining cost matrix
        for (row, col) in self.excluded_cells:
            remaining_cost_matrix[row][col] = sys.maxint

        rows_to_delete = []
        cols_to_delete = []
        for (row, col) in self.required_cells: #remove required rows and columns
            remaining_cost_matrix[row, :] = INFEASIBLE_COST
            remaining_cost_matrix[:, col] = INFEASIBLE_COST
            remaining_cost_matrix[row, col] = .00000001

        return remaining_cost_matrix


    def get_orig_indices(self, rcm_indices):
        '''
        Take a list of indices in the remaining cost matrix and transform them into indices
        in the original cost matrix

        Inputs:
        - rcm_indices: (list of pairs) indices in the remaining cost matrix

        Outputs:
        - orig_indices: (list of pairs) converted indices in the original cost matrix
        '''

        orig_indices = rcm_indices

        deleted_rows = []
        deleted_cols = []
        for (row, col) in self.required_cells: #remove required rows and columns
            deleted_rows.append(row)
            deleted_cols.append(col)

        #create sorted lists of rows and columns that were deleted in the remaining cost matrix,
        #where indices are sorted in increasing order, e.g. [1, 4, 5, 9]
        sorted_deleted_rows = sorted(deleted_rows)
        sorted_deleted_cols = sorted(deleted_cols)

        for deleted_row in sorted_deleted_rows:
            for idx, (row, col) in enumerate(orig_indices):
                if deleted_row <= row:
                    orig_indices[idx] = (orig_indices[idx][0] + 1, orig_indices[idx][1])

        for deleted_col in sorted_deleted_cols:
            for idx, (row, col) in enumerate(orig_indices):
                if deleted_col <= col:
                    orig_indices[idx] = (orig_indices[idx][0], orig_indices[idx][1] + 1)

        return orig_indices


def brute_force_k_best_assignments(k, cost_matrices, matrix_costs, M):
    '''
    The cost of an assignment is given by the assignment to the specified cost matrix PLUS the
    cost for the matrix given in matrix_costs
    Inputs:
    - k: (integer), find top k best assignments
    - cost_matrices: (list of numpy arrays)  
    - matrix_costs: (list of floats) same length as cost_matrices.  add this to every assignment cost for the corresponding matrix
        in cost_matrices
    - M: number of measurements 


    Output:
    - best_assignments: (list of triplets) best_assignments[i][0] is the cost of the ith best
        assignment.  best_assignments[i][1] is the ith best assignment, which is a list of pairs
        where each pair represents an association in the assignment (1's in assignment matrix),
        best_assignments[i][2] is the index in the input cost_matrices of the cost matrix used
        for the ith best assignment

    '''
    potential_best_assignments = [] 

    for cm_idx, cost_matrix in enumerate(cost_matrices):
        T = len(cost_matrix[1])/2 - M
        assert(len(cost_matrix[1]) == 2*M + 2*T), (len(cost_matrix[1]), M, T)                
        assert(cost_matrix.shape[0] == cost_matrix.shape[1])

        n = cost_matrix.shape[0]
        all_perm_mats = gen_permutation_matrices(n)
        costs = []
        for pm in all_perm_mats:
            costs.append(matrix_costs[cm_idx]+np.trace(np.dot(pm, np.transpose(cost_matrix))))

        cur_best_assignments = [] 
        unique_assoc_deaths = []
        while len(cur_best_assignments) < k and len(costs) > 0:
            (min_key, min_cost) = min(enumerate(costs), key=itemgetter(1)) #find the next smallest cost
            if min_cost < INFEASIBLE_COST: #invalid assignment
                #check this assignment is new
                (meas_grp_associations, dead_target_indices) = convert_assignment_matrix3(all_perm_mats[min_key], M, T)
                cur_tuple = (tuple(meas_grp_associations), tuple(dead_target_indices))
                if not cur_tuple in unique_assoc_deaths:
                    unique_assoc_deaths.append(cur_tuple)
                    cur_best_assignments.append((min_cost, convert_perm_array_to_list(all_perm_mats[min_key]), cm_idx))
                    if DEBUG2:
                        print 'found new potential assoc,', cur_tuple, ', M=', M, ', T=',T
                del all_perm_mats[min_key]
                del costs[min_key]
            else:
                break

        potential_best_assignments.extend(cur_best_assignments)

    best_assignments = sorted(potential_best_assignments, key=itemgetter(0))[0:k]


    return best_assignments


def convert_perm_list_to_array(list_):
    '''
    Input:
    - list_: a list of length n, where each element is a pair representing an element in an nxn
    permutation matrix that is a 1. 

    Output:
    - matrix: numpy array of the permutation matrix
    '''
    array = np.zeros((len(list_), len(list_)))
    for indices in list_:
        array[indices] = 1
    return array

def convert_perm_array_to_list(arr):
    '''
    Input:
    - matrix: numpy array of the permutation matrix

    Output:
    - list_: a list of length n, where each element is a pair representing an element in an nxn
    permutation matrix that is a 1. 

    '''
    list_ = []
    assert(arr.shape[0] == arr.shape[1])
    n = arr.shape[0]
    for row in range(n):
        for col in range(n):
            if(arr[(row, col)] == 1):
                list_.append((row,col))
    assert(len(list_) == n)
    return list_

def check_assignments_match(best_assignments1, best_assignments2):
    for (index, (cost, assignment_list, cost_matrix_index)) in enumerate(best_assignments1):
        assert(len(best_assignments1) == len(best_assignments2)), (len(best_assignments1), len(best_assignments2), best_assignments1, best_assignments2)
        #assert(cost == best_assignments2[index][0]), (cost, best_assignments2[index][0])
#        np.testing.assert_allclose(cost, best_assignments2[index][0], rtol=1e-5, atol=0), (cost, best_assignments2[index][0], best_assignments1, best_assignments2)
        assert(np.abs(cost - best_assignments2[index][0]) < .00001), (cost, best_assignments2[index][0], best_assignments1, best_assignments2)
        assert((convert_perm_list_to_array(assignment_list) == convert_perm_list_to_array(best_assignments2[index][1])).all), (assignment_list, best_assignments1[index][1])
        assert(cost_matrix_index == best_assignments2[index][2]), (cost_matrix_index, best_assignments2[index][2], cost, best_assignments2[index][0], convert_perm_list_to_array(assignment_list), best_assignments2[index][1])

def gen_permutation_matrices(n):
    '''
    return a list of all nxn permutation matrices (numpy arrays)
    '''
    all_permutation_matrices = []
    for cur_permutation in itertools.permutations([i for i in range(n)], n):
        cur_perm_mat = np.zeros((n,n))
        for row, col in enumerate(cur_permutation):
            cur_perm_mat[row][col] = 1
        all_permutation_matrices.append(cur_perm_mat)
    return all_permutation_matrices


def construct_random_costs_matrix(M, T):
    '''
    Inputs:
    - M: (int), #measurements
    - T: (int), #targets

    Outputs:
    - cost_matrix: numpy matrix with dimensions (2*M+2*T)x(2*M+2*T) of costs
    '''
    #construct a (2*M+2*T)x(2*M+2*T) matrix of log probabilities
    cost_matrix = np.ones((2*M + 2*T, 2*T + 2*M))
    cost_matrix *= INFEASIBLE_COST

    #generate random costs for target association entries in the cost matrix
    for t_idx in range(T):
        for m_idx in range(M):
        #generate random costs for measurement-target association entries in the cost matrix
            cost_matrix[m_idx][t_idx] = np.random.rand(1)[0]*10

        #generate random costs for target doesn't emit and lives/dies entries in the cost matrix
        lives_row_idx = M + 2*t_idx
        dies_row_idx = M + 1 + 2*t_idx
        cost_matrix[lives_row_idx][t_idx] = np.random.rand(1)[0]*10
        cost_matrix[dies_row_idx][t_idx] = np.random.rand(1)[0]*10


    #add birth/clutter measurement association entries to the cost matrix
    for m_idx in range(M):
        clutter_col = T + 2*m_idx
        birth_col = T + 1 + 2*m_idx

        cost_matrix[m_idx][clutter_col] = np.random.rand(1)[0]*10
        cost_matrix[m_idx][birth_col] = np.random.rand(1)[0]*10

    #set bottom right quadrant to 0's
    for row_idx in range(M, 2*M+2*T):
        for col_idx in range(T, 2*T+2*M):
            cost_matrix[row_idx][col_idx] = 0.0

    cost_matrix = cost_matrix + 2.0 #in Node we want all entries to be great than zero, not sure offhand why

    return cost_matrix

def test_against_brute_force(M,k,num_cost_matrices,iters):
    '''
    Test our implementation to find the k best assignments for a set of cost
    matrices, each with an associated cost, against a brute force approach.
    Inputs:
    - M: use a random cost matrix of size (2*M + 2*T)x(2*M + 2*T) with this M and random 
        T in range 0, M+1
    - k: find k best solutions
    - iters: number of random problems to solve and check
    - num_cost_matrices: integer, the number of cost matrices to generate
    '''
    for test_iter in range(iters):
        #create cost matrices and associated costs
        cost_matrices = []
        matrix_costs = []
        for m_idx in range(num_cost_matrices):
            T = random.randrange(M+1) + 1
            cost_matrix = construct_random_costs_matrix(M, T)
            matrix_cost = np.random.rand(1)[0]*1000
            cost_matrices.append(cost_matrix)
            matrix_costs.append(matrix_cost)

        best_assignments = k_best_assign_mult_cost_matrices(k, cost_matrices, matrix_costs, M)
        print "calculated with Hungarian"        
        if DEBUG:
            for (idx, assignment) in enumerate(best_assignments):
                print idx, ":   ", assignment
            print

        best_assignments_brute_force = brute_force_k_best_assignments(k, cost_matrices, matrix_costs, M)
        print "calculated with brute force"

        if DEBUG:
            for (idx, (cost, perm)) in enumerate(best_assignments_brute_force):
                print idx, ":   ", (cost, convert_perm_array_to_list(perm))
        check_assignments_match(best_assignments, best_assignments_brute_force)
        print "match!"


def speed_test(M,k,num_cost_matrices,iters):
    '''
    Time our implementation to find the k best assignments for a set of cost
    matrices, each with an associated cost
    Inputs:
    - M: use a random cost matrix of size (2*M + 2*T)x(2*M + 2*T) with this M and random 
        T in range 0, M+1
    - k: find k best solutions
    - iters: number of random problems to solve and check
    - num_cost_matrices: integer, the number of cost matrices to generate
    '''
    cost_matrices = []
    matrix_costs = []
    for m_idx in range(num_cost_matrices):
        T = M
        cost_matrix = construct_random_costs_matrix(M, T)
        matrix_cost = np.random.rand(1)[0]*1000
        cost_matrices.append(cost_matrix)
        matrix_costs.append(matrix_cost)

    t1 = time.time()

    for test_iter in range(iters):
        #create cost matrices and associated costs

        best_assignments = k_best_assign_mult_cost_matrices(k, cost_matrices, matrix_costs, M)

    t2 = time.time()

    print "calculation took", t2-t1, "seconds"


if __name__ == "__main__":
    M = 1 # number of measurements in cost matrices of size 
    k = 5 # find k lowest cost solutions
    num_cost_matrices = 10
    test_against_brute_force(M,k,num_cost_matrices,100)

    M = 15 # number of measurements in cost matrices of size 
    k = 100 # find k lowest cost solutions
    num_cost_matrices = 10
    iters = 10
    if PROFILE:
        cProfile.runctx('speed_test(M,k,num_cost_matrices,iters)', {'M': M, 'k': k,
            'num_cost_matrices':num_cost_matrices, 'iters':iters, 'speed_test':speed_test}, {})

    else:
        speed_test(M,k,num_cost_matrices,iters)

