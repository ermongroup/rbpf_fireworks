import numpy as np
from munkres import Munkres, print_matrix
import sys
import itertools
import math
from operator import itemgetter

DEBUG = False
DEBUG1 = False
#
#
#References:
# [1] K. G. Murty, "Letter to the Editor--An Algorithm for Ranking all the Assignments in Order of
#     Increasing Cost," Oper. Res., vol. 16, no. May 2016, pp. 682-687, 1968.
#
# [2] I. J. Cox and M. L. Miller, "On finding ranked assignments with application to multitarget
#     tracking and motion correspondence," IEEE Trans. Aerosp. Electron. Syst., vol. 31, no. 1, pp.
#     486-489, Jan. 1995.

def k_best_assignments(k, cost_matrix):
    '''
    Find the k best assignments for the given cost matrix

    Inputs:
    - k: (integer) 
    - cost_matrix: (numpy array)  

    Output:
    - best_assignments: (list of pairs) best_assignments[i][0] is the cost of the ith best
        assignment.  best_assignments[i][1] is the ith best assignment, which is a list of pairs
        where each pair represents an association in the assignment (1's in assignment matrix)
    '''
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            assert(sys.maxint > cost_matrix[i][j])

    init_node = Node(cost_matrix, [], [], 0)
    best_assignments = [init_node.get_min_cost_assignment()[:2]]
    cur_partition = init_node.partition()
    for itr in range(1, k):
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

        best_assignments.append(cur_partition[min_cost_idx].get_min_cost_assignment()[:2])
        assert(len(best_assignments[-1])==2)
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
    return best_assignments


def k_best_assign_mult_cost_matrices(k, cost_matrices):
    '''
    Find the k lowest cost assignments for any of the cost matrices.  That is, the lowest cost will
    be the lowest cost assignment with costs specified by ANY of the cost matrices.  This is 
    useful for multiple hypothesis tracking, where we want to find the k lowest costs and have
    k cost matrices.  Rather than solving k k_best_assignment problems, generating k^2 costs, and
    picking the k smallest costs we will instead initialize Murty's algorithm with a set of 
    cost matrices as explained in [2] on pp. 487-488.


    Inputs:
    - k: (integer) 
    - cost_matrices: (list of numpy arrays)  

    Output:
    - best_assignments: (list of triplets) best_assignments[i][0] is the cost of the ith best
        assignment.  best_assignments[i][1] is the ith best assignment, which is a list of pairs
        where each pair represents an association in the assignment (1's in assignment matrix),
        best_assignments[i][2] is the index in the input cost_matrices of the cost matrix used
        for the ith best assignment
    '''
    for cur_cost_matrix in cost_matrices:
        for i in range(cur_cost_matrix.shape[0]):
            for j in range(cur_cost_matrix.shape[1]):
                assert(sys.maxint > cur_cost_matrix[i][j])
    best_assignments = []
    cur_partition = []
    for (idx, cur_cost_matrix) in enumerate(cost_matrices):
        cur_partition.append(Node(cur_cost_matrix, [], [], idx))

    for itr in range(0, k):
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
    return best_assignments


class Node:
    def __init__(self, orig_cost_matrix, required_cells, excluded_cells, orig_cost_matrix_index):
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
        '''
        self.orig_cost_matrix = np.array(orig_cost_matrix, copy=True)
        self.required_cells = required_cells[:]
        self.excluded_cells = excluded_cells[:]
        self.orig_cost_matrix_index = orig_cost_matrix_index
        if DEBUG:
            print "New Node:"
            print "self.required_cells:", self.required_cells 
            print "self.excluded_cells:", self.excluded_cells 

        #we will transform the cost matrix into the "remaining cost matrix" as described in [1]
        self.remaining_cost_matrix = self.construct_remaining_cost_matrix()
        assert((self.remaining_cost_matrix > -.000001).all()), self.remaining_cost_matrix
        #solve the assignment problem for the remaining cost matrix
        hm = Munkres()
        # we get a list of (row, col) associations, or 1's in the minimum assignment matrix
        association_list = hm.compute(self.remaining_cost_matrix.tolist())
        if DEBUG:
            print "remaining cost matrix:"
            print self.remaining_cost_matrix
            print "association_list"
            print association_list


        #compute the minimum cost assignment for the node
        self.minimum_cost = 0
        for (row,col) in association_list:
#            print 'a', self.minimum_cost, type(self.minimum_cost)
#            print 'b', self.remaining_cost_matrix[row][col], type(self.remaining_cost_matrix[row][col])
#            print 'c', self.minimum_cost +self.remaining_cost_matrix[row][col], type(self.minimum_cost +self.remaining_cost_matrix[row][col])
            #np.asscalar important for avoiding overflow problems
            self.minimum_cost += np.asscalar(self.remaining_cost_matrix[row][col])
        for (row, col) in self.required_cells:
            #np.asscalar important for avoiding overflow problems
            self.minimum_cost += np.asscalar(orig_cost_matrix[row][col])

        #store the minimum cost associations with indices consistent with the original cost matrix
        self.min_cost_associations = self.get_orig_indices(association_list)

        if DEBUG:
            print "New Node:"
            print "self.required_cells:", self.required_cells 
            print "self.excluded_cells:", self.excluded_cells 
            print
            print

    def get_min_cost_assignment(self):
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
            of this node forms the set of possible assignments represented by this node
        '''
        partition = []
        cur_required_cells = self.required_cells[:]

        if DEBUG:
            print '!'*40, 'Debug partition()', '!'*40
            print len(self.min_cost_associations) - 1

        for idx in range(len(self.min_cost_associations)  - 1):
            cur_assoc = self.min_cost_associations[idx]
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
                                  self.orig_cost_matrix_index))
            cur_required_cells.append(cur_assoc)


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

def brute_force_k_best_assignments(k, cost_matrix):
    assert(cost_matrix.shape[0] == cost_matrix.shape[1])
    n = cost_matrix.shape[0]
    all_perm_mats = gen_permutation_matrices(n)
    costs = []
    for pm in all_perm_mats:
        costs.append(np.trace(np.dot(pm, np.transpose(cost_matrix))))

    min_costs = [] #list of triples (smallest k costs, corresponding permutation matrix, 0)
    for i in range(k):
        (min_key, min_cost) = min(enumerate(costs), key=itemgetter(1)) #find the next smallest cost
        min_costs.append((min_cost, all_perm_mats[min_key], 0))
        del all_perm_mats[min_key]
        del costs[min_key]

    return min_costs


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
        #assert(cost == best_assignments2[index][0]), (cost, best_assignments2[index][0])
        np.testing.assert_allclose(cost, best_assignments2[index][0], rtol=1e-5, atol=0), (cost, best_assignments2[index][0])
#        assert((convert_perm_list_to_array(assignment_list) == best_assignments2[index][1]).all), (assignment_list, best_assignments1[index][1])
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


def test_against_brute_force(N,k,iters):
    '''
    Test our implementation of Murty's algorithm to find the k best assignments for a given cost
    matrix against a brute force approach.
    Inputs:
    - N: use a random cost matrix of size (NxN)
    - k: find k best solutions
    - iters: number of random problems to solve and check
    '''
    for test_iter in range(iters):
        cost_matrix = np.random.rand(N,N)*1000

        best_assignments = k_best_assignments(k, cost_matrix)
        if DEBUG:
            for (idx, assignment) in enumerate(best_assignments):
                print idx, ":   ", assignment
            print
        print "calculated with Hungarian"
        best_assignments_brute_force = brute_force_k_best_assignments(k, cost_matrix)
        print "calculated with brute force"

        if DEBUG:
            for (idx, (cost, perm)) in enumerate(best_assignments_brute_force):
                print idx, ":   ", (cost, convert_perm_array_to_list(perm))
        check_assignments_match(best_assignments, best_assignments_brute_force)
        print "match!"

def test_mult_cost_matrices(num_cost_matrices, N,k,iters):
    '''
    Inputs:
    - num_cost_matrices: number of cost matrices to use
    - N: use a random cost matrices of size (NxN)
    - k: find k best solutions
    - iters: number of random problems to solve and check
    '''
    for test_iter in range(iters):
        cost_matrices = []
        for i in range(num_cost_matrices):
            cost_matrices.append(np.random.rand(N,N))

        best_assignments_mult = k_best_assign_mult_cost_matrices(k, cost_matrices)
#        print best_assignments_mult
        print 'calculated'
        #now try using k_best_assignments k times
        best_assignments_naive = []
        for (idx, cur_cost_matrix) in enumerate(cost_matrices):
            cur_best_assignments = k_best_assignments(k, cur_cost_matrix)
            for (idx1, cur_assignment) in enumerate(cur_best_assignments):
                cur_best_assignments[idx1] = (cur_assignment[0], cur_assignment[1], idx)
            best_assignments_naive.extend(cur_best_assignments)
        best_assignments_naive.sort(key=itemgetter(0))
        best_assignments_naive = best_assignments_naive[0:k]
        print "calculated naive"

#        print
#        print best_assignments_naive
        check_assignments_match(best_assignments_mult, best_assignments_naive)
        print 'match!'
#        if DEBUG:
#            for (idx, assignment) in enumerate(best_assignments):
#                print idx, ":   ", assignment
#            print
#        print "calculated with Hungarian"
#        best_assignments_brute_force = brute_force_k_best_assignments(k, cost_matrix)
#        print "calculated with brute force"
#
#        if DEBUG:
#            for (idx, (cost, perm)) in enumerate(best_assignments_brute_force):
#                print idx, ":   ", (cost, convert_perm_array_to_list(perm))
#        check_assignments_match(best_assignments, best_assignments_brute_force)
#        print "match!"






if __name__ == "__main__":
    N = 7 # cost matrices of size (NxN)
    k = 50 # find k lowest cost solutions
    num_cost_matrices = 10
    test_against_brute_force(N,k,100)
#    test_mult_cost_matrices(num_cost_matrices, N, k, 100)



