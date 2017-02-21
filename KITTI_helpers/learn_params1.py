#Copied from KITTI devkit_tracking/python/evaluate_tracking.py and then edited


#!/usr/bin/env python
# encoding: utf-8
from __future__ import division
import numpy as np
import sys,os,copy,math
import os.path
from munkres import Munkres
from collections import defaultdict
from sets import ImmutableSet
from numpy.linalg import inv

#try:
#    from ordereddict import OrderedDict # can be installed using pip
#except:
#    from collections import OrderedDict # only included from python 2.7 on

import mailpy
#from learn_Q import run_EM_on_Q_multiple_targets
#from learn_Q import Target
#from learn_Q import default_time_step
import pickle

sys.path.insert(0, "../")
from cluster_config import RBPF_HOME_DIRECTORY

LEARN_Q_FROM_ALL_GT = False
SKIP_LEARNING_Q = True

#load ground truth data and detection data, when available, from saved pickle file
#to cut down on load time
#Be careful!! If changing data representation, e.g. class gtObject, need to delete pickled data!!
USE_PICKLED_DATA = True
PICKELD_DATA_DIRECTORY = "%sKITTI_helpers/learn_params1_pickled_data" % RBPF_HOME_DIRECTORY
DATA_PATH = "%sKITTI_helpers/data" % RBPF_HOME_DIRECTORY
#DATA_PATH = "./data"

CAMERA_PIXEL_WIDTH = 1242
CAMERA_PIXEL_HEIGHT = 375
#########################################################################
# function that does the evaluation
# input:
#   - det_method (method used for frame by frame detection and the name of the folder
#       where the detections are stored)
#   - mail (messenger object for output messages sent via email and to cout)
# output:
#   - True if at least one of the sub-benchmarks could be processed successfully
#   - False otherwise
# data:
#   - the results shall be saved as follows
#     -> summary statistics of the method: results/<det_method>/stats_task.txt
#        here task refers to the sub-benchmark (e.g., um_lane, uu_road etc.)
#        file contents: numbers for main table, format: %.6f (single space separated)
#        note: only files with successful sub-benchmark evaluation must be created
#     -> detailed results/graphics/plots: results/<det_method>/subdir
#        with appropriate subdir and file names (all subdir's need to be created)

class tData:
    def __init__(self,frame=-1,obj_type="unset",truncation=-1,occlusion=-1,\
                 obs_angle=-10,x1=-1,y1=-1,x2=-1,y2=-1,w=-1,h=-1,l=-1,\
                 X=-1000,Y=-1000,Z=-1000,yaw=-10,score=-1000,track_id=-1):

        # init object data
        self.frame      = frame
        self.track_id   = track_id
        self.obj_type   = obj_type
        self.truncation = truncation
        self.occlusion  = occlusion
        self.obs_angle  = obs_angle
        self.x1         = x1
        self.y1         = y1
        self.x2         = x2
        self.y2         = y2
        self.w          = w
        self.h          = h
        self.l          = l
        self.X          = X
        self.Y          = Y
        self.Z          = Z
        self.yaw        = yaw
        self.score      = score
        self.ignored    = False
        self.valid      = False
        self.tracker    = -1

    def __str__(self):
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())

class gtObject:
    def __init__(self, x1, x2, y1, y2, track_id):
        """
        Ground truth object (occurence of a ground truth track in a single frame)

        Input:
        - x1: left edge of bounding box (smaller value)
        - x2: right edge of bounding box (larger value)
        - y1: upper edge of bounding box (smaller value)
        - y2: lower edge of bounding box  (larger value)
        """
        assert(x2 > x1 and y2 > y1)
        self.x = float(x1+x2)/2.0 #x-coord of bounding box center
        self.y = float(y1+y2)/2.0 #y-coord of bounding box center

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1


        #id of the track this object belongs to
        self.track_id = track_id

        #dictionary of all associated detections
        #assoc_dets['det_name'] is the detection of type 'det_name' associated
        #with this ground truth object
        self.assoc_dets = {}

        #This will be the detObj this ground truth object is associated with,
        #if this gtObject is associated with any detection
        self.associated_detection = None

        if (x1 < 10 or x2 > CAMERA_PIXEL_WIDTH - 15 or y1 < 10 or y2 > CAMERA_PIXEL_HEIGHT - 15):
            self.near_border = True
        else:
            self.near_border = False

class detObject:
    def __init__(self, x1, x2, y1, y2, assoc, score):
        """
        Detected object (in a single frame)

        Input:
        - x1: left edge of bounding box (smaller value)
        - x2: right edge of bounding box (larger value)
        - y1: upper edge of bounding box (smaller value)
        - y2: lower edge of bounding box  (larger value)
        """
        assert(x2 > x1 and y2 > y1)
        self.x = float(x1+x2)/2.0 #x-coord of bounding box center
        self.y = float(y1+y2)/2.0 #y-coord of bounding box center

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1


        # _id of the ground truth track this detection is associated with
        # --OR--
        # -1 if this detection is unassociated (clutter detection)
        self.assoc = assoc

        #the detection score
        self.score = score


class trackingEvaluation(object):
    """ tracking statistics (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
             MOTA	- Multi-object tracking accuracy in [0,100]
             MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
             MOTAL	- Multi-object tracking accuracy in [0,100] with log10(id-switches)

             id-switches - number of id switches
             fragments   - number of fragmentations

             MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories

             recall	        - recall = percentage of detected targets
             precision	    - precision = percentage of correctly detected targets
             FAR		    - number of false alarms per frame
             falsepositives - number of false positives (FP)
             missed         - number of missed targets (FN)
    """

    def __init__(self, cutoff_score, det_method, gt_path=DATA_PATH + "/training_ground_truth", min_overlap=0.5, max_truncation = 0.15, mail=None, cls="car"):
        #jdk parameters to learn
        self.cutoff_score = cutoff_score
        self.clutter_count_list = []
        self.p_target_emission = -1
        self.gt_birth_count_probs = []
        self.detection_birth_count_probs = []
        self.discontinuous_target_count = 0
        self.discontinuous_target_ids = [] #these are ignored when computing death probabilities
        self.targ_cnt_still_living_but_unassoc_after_n = [-99]

        #freq_unassociated_frames_before_target_death[i] = j represents that j targets are unassociated
        #for i frames before dying
        self.freq_unassociated_frames_before_target_death = {}
        #targ_cnt_dead_at_n[i] is the number of targets that were:
        # -alive and associated with a measurement at time t
        # -alive and unassociated with a measurement at [t+1, t+i-1] (for i > 1)
        # -dead at time t+i
        self.targ_cnt_dead_at_n = [-99]

        #for i > 0, target_death_probabilities[i] = 
        #targ_cnt_dead_at_n[i] / (targ_cnt_dead_at_n[i] + targ_cnt_still_living_but_unassoc_after_n[i])
        #This is the probability that a target dies i frames after its last association
        self.target_death_probabilities = [-99]

        self.total_death_count = 0

        #groundTruthLocations[i] is a TargetSet of ground truth targets in sequence i
        self.groundTruthTargetSets = []
        self.meas_errors = []
        self.gt_widths = []
        self.gt_heights = []

        self.all_gt_targets_dict = {}
        self.Q_estimate = -99
        self.measurementTargetSetsBySequence = []



        #begin jdk clean



        #end jdk clean

        # get number of sequences and
        # get number of frames per sequence from test mapping
        # (created while extracting the benchmark)
        filename_test_mapping = DATA_PATH + "/evaluate_tracking.seqmap"
        self.n_frames         = []
        self.sequence_name    = []
        with open(filename_test_mapping, "r") as fh:
            for i,l in enumerate(fh):
                fields = l.split(" ")
                self.sequence_name.append("%04d" % int(fields[0]))
                self.n_frames.append(int(fields[3]) - int(fields[2])+1) #jdk, why is there a +1 ???
        fh.close()                                
        self.n_sequences = i+1

        # mail object
        self.mail = mail

        # class to evaluate
        self.cls = cls

        # data and parameter
        self.gt_path           = os.path.join(gt_path, "label_02")

        self.det_method = det_method
        self.t_path            = os.path.join(DATA_PATH + "/object_detections", self.det_method, "training/det_02")
        self.n_gt              = 0
        self.n_gt_trajectories = 0
        self.n_gt_seq          = []
        self.n_tr              = 0
        self.n_tr_trajectories = 0
        self.n_tr_seq          = []
        self.min_overlap       = min_overlap # minimum bounding box overlap for 3rd party metrics
        self.max_truncation    = max_truncation # maximum truncation of an object for evaluation
        self.n_sample_points   = 500
        # figures for evaluation
        self.MOTA              = 0
        self.MOTP              = 0
        self.MOTAL             = 0
        self.MODA              = 0
        self.MODP              = 0
        self.MODP_t            = []
        self.recall            = 0
        self.precision         = 0
        self.F1                = 0
        self.FAR               = 0
        self.total_cost        = 0
        self.tp                = 0
        self.fn                = 0
        self.fp                = 0
        self.mme               = 0
        self.fragments         = 0
        self.id_switches       = 0
        self.MT                = 0
        self.PT                = 0
        self.ML                = 0
        self.distance          = []
        self.seq_res           = []
        self.seq_output        = []
        # this should be enough to hold all groundtruth trajectories
        # is expanded if necessary and reduced in any case
        self.gt_trajectories   = [[] for x in xrange(self.n_sequences)] 
        self.ign_trajectories  = [[] for x in xrange(self.n_sequences)]

    def createEvalDir(self):
        """Creates directory to store evaluation results and data for visualization"""
        self.eval_dir = os.path.join("./data/object_detections", self.det_method, "eval", self.cls)
        if not os.path.exists(self.eval_dir):
            print "create directory:", self.eval_dir,
            os.makedirs(self.eval_dir)
            print "done"

    def loadGroundtruth(self, include_dontcare_in_gt):
        """Helper function to load ground truth"""
        try:
            self._loadData(self.gt_path, cls=self.cls, include_dontcare_in_gt=include_dontcare_in_gt, loading_groundtruth=True)
        except IOError:
            return False
        return True

    def loadDetections(self):
        """Helper function to load tracker data"""
        try:
            if not self._loadData(self.t_path, cls=self.cls, loading_groundtruth=False):
                return False
        except IOError:
            return False
        return True

    def _loadData(self, root_dir, cls, include_dontcare_in_gt=False, min_score=-1000, loading_groundtruth=False):
        """
            Generic loader for ground truth and tracking data.
            Use loadGroundtruth() or loadDetections() to load this data.
            Loads detections in KITTI format from textfiles.
        """
        # construct objectDetections object to hold detection data
        print "hi1, loading_groundtruth =", loading_groundtruth
        print "root_dir =", root_dir
        t_data  = tData()
        data    = []
        eval_2d = True
        eval_3d = True

        seq_data           = []
        n_trajectories     = 0
        n_trajectories_seq = []

        if not loading_groundtruth:
            fake_track_id = 0 #we'll assign a unique id to every detected object

        for seq, s_name in enumerate(self.sequence_name):
            i              = 0
            filename       = os.path.join(root_dir, "%s.txt" % s_name)
            f              = open(filename, "r") 

            f_data         = [[] for x in xrange(self.n_frames[seq])] # current set has only 1059 entries, sufficient length is checked anyway
            ids            = []
            n_in_seq       = 0
            id_frame_cache = []
            for line in f:
                if not loading_groundtruth:
                    fake_track_id += 1
                # KITTI tracking benchmark data format:
                # (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
                line = line.strip()
                fields            = line.split(" ")
                # classes that should be loaded (ignored neighboring classes)
                if "car" in cls.lower():
                    classes = ["car","van"]
                elif "pedestrian" in cls.lower():
                    classes = ["pedestrian","person_sitting"]
                else:
                    classes = [cls.lower()]
                classes += ["dontcare"]
                if not any([s for s in classes if s in fields[2].lower()]):
                    continue
                # get fields from table
                t_data.frame        = int(float(fields[0]))     # frame
                if loading_groundtruth:
                    t_data.track_id     = int(float(fields[1]))     # id
                else: 
                    t_data.track_id = fake_track_id
                t_data.obj_type     = fields[2].lower()         # object type [car, pedestrian, cyclist, ...]
                t_data.truncation   = float(fields[3])          # truncation [0..1]
                t_data.occlusion    = int(float(fields[4]))     # occlusion  [0,1,2]
                t_data.obs_angle    = float(fields[5])          # observation angle [rad]
                t_data.x1           = float(fields[6])          # left   [px]
                t_data.y1           = float(fields[7])          # top    [px]
                t_data.x2           = float(fields[8])          # right  [px]
                t_data.y2           = float(fields[9])          # bottom [px]
                t_data.h            = float(fields[10])         # height [m]
                t_data.w            = float(fields[11])         # width  [m]
                t_data.l            = float(fields[12])         # length [m]
                t_data.X            = float(fields[13])         # X [m]
                t_data.Y            = float(fields[14])         # Y [m]
                t_data.Z            = float(fields[15])         # Z [m]
                t_data.yaw          = float(fields[16])         # yaw angle [rad]
                if not loading_groundtruth:
                    if len(fields) == 17:
                        t_data.score = -1
                    elif len(fields) == 18:
                        t_data.score  = float(fields[17])     # detection score
                        if t_data.score <= self.cutoff_score:
                            continue
                    else:
                        self.mail.msg("file is not in KITTI format")
                        return

                # do not consider objects marked as invalid
                if t_data.track_id is -1 and t_data.obj_type != "dontcare":
                    continue

                idx = t_data.frame
                # check if length for frame data is sufficient
                if idx >= len(f_data):
                    print "extend f_data", idx, len(f_data)
                    f_data += [[] for x in xrange(max(500, idx-len(f_data)))]
                try:
                    id_frame = (t_data.frame,t_data.track_id)
                    if id_frame in id_frame_cache and not loading_groundtruth:
                        self.mail.msg("track ids are not unique for sequence %d: frame %d" % (seq,t_data.frame))
                        self.mail.msg("track id %d occured at least twice for this frame" % t_data.track_id)
                        self.mail.msg("Exiting...")
                        #continue # this allows to evaluate non-unique result files
                        return False
                    id_frame_cache.append(id_frame)
                    f_data[t_data.frame].append(copy.copy(t_data))
                except:
                    print len(f_data), idx
                    raise

                if t_data.track_id not in ids and t_data.obj_type!="dontcare":
                    ids.append(t_data.track_id)
                    n_trajectories +=1
                    n_in_seq +=1

                # check if uploaded data provides information for 2D and 3D evaluation
                if not loading_groundtruth and eval_2d is True and(t_data.x1==-1 or t_data.x2==-1 or t_data.y1==-1 or t_data.y2==-1):
                    eval_2d = False
                if not loading_groundtruth and eval_3d is True and(t_data.X==-1000 or t_data.Y==-1000 or t_data.Z==-1000):
                    eval_3d = False

            # only add existing frames
            n_trajectories_seq.append(n_in_seq)
            seq_data.append(f_data)
            f.close()

        if not loading_groundtruth:
            print "hi2!"
            self.tracker=seq_data
            self.n_tr_trajectories=n_trajectories
            self.eval_2d = eval_2d
            self.eval_3d = eval_3d
            self.n_tr_seq = n_trajectories_seq
            if self.n_tr_trajectories==0:
                return False
        else: 
            # split ground truth and DontCare areas
            self.dcareas     = []
            self.groundtruth = []
            for seq_idx in range(len(seq_data)):
                seq_gt = seq_data[seq_idx]
                s_g, s_dc = [],[]
                for f in range(len(seq_gt)):
                    all_gt = seq_gt[f]
                    g,dc = [],[]
                    for gg in all_gt:
                        if include_dontcare_in_gt:
                            g.append(gg)
                        else:
                            if gg.obj_type=="dontcare":
                                dc.append(gg)
                            else:
                                g.append(gg)
                    s_g.append(g)
                    s_dc.append(dc)
                self.dcareas.append(s_dc)
                self.groundtruth.append(s_g)
            self.n_gt_seq=n_trajectories_seq
            self.n_gt_trajectories=n_trajectories
        return True
            
            
    def boxoverlap(self,a,b,criterion="union"):
        """
            boxoverlap computes intersection over union for bbox a and b in KITTI format.
            If the criterion is 'union', overlap = (a inter b) / a union b).
            If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
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
        elif criterion.lower()=="a":
            o = float(inter) / float(aarea)
        else:
            raise TypeError("Unkown type for criterion")
        return o

    def compute3rdPartyMetrics(self, include_ignored_gt, include_dontcare_in_gt, include_ignored_detections):
        """
            Computes the metrics defined in 
                - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics
                  MOTA, MOTAL, MOTP
                - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows
                  MT/PT/ML
        """

        # construct Munkres object for Hungarian Method association
        hm = Munkres()
        max_cost = 1e9


        #gt_objects[i][j] is a list of all ground truth objects in the jth frame of the ith video sequence
        gt_objects = []
        #det_objects[i][j] is a list of all detected objects in the jth frame of the ith video sequence
        det_objects = []

        # go through all frames and associate ground truth and tracker results
        # groundtruth and tracker contain lists for every single frame containing lists of KITTI format detections
        fr, ids = 0,0 
        for seq_idx in range(len(self.groundtruth)):
            x_min = 9999999
            x_max = -9999999
            y_min = 9999999
            y_max = -9999999

            gt_objects.append([]) #jdk
            det_objects.append([]) #jdk

            seq_gt           = self.groundtruth[seq_idx]
            seq_dc           = self.dcareas[seq_idx]
            seq_tracker      = self.tracker[seq_idx]
            seq_trajectories = defaultdict(list)
            seq_ignored      = defaultdict(list)
            seqtp            = 0
            seqfn            = 0
            seqfp            = 0
            seqcost          = 0

            last_ids = [[],[]]
            tmp_frags = 0


            for f in range(len(seq_gt)):
                gt_objects[-1].append([]) #jdk
                det_objects[-1].append([]) #jdk

                g = seq_gt[f]
                dc = seq_dc[f]
                        
                t = seq_tracker[f]
                # counting total number of ground truth and tracker objects
                self.n_gt += len(g)
                self.n_tr += len(t)

                # use hungarian method to associate, using boxoverlap 0..1 as cost
                # build cost matrix
                cost_matrix = []
                this_ids = [[],[]]


                for gg in g:

                    # save current ids
                    this_ids[0].append(gg.track_id)
                    this_ids[1].append(-1)
                    gg.tracker       = -1
                    gg.id_switch     = 0
                    gg.fragmentation = 0
                    cost_row         = []
                    for tt in t:
                        # overlap == 1 is cost ==0
                        c = 1-self.boxoverlap(gg,tt)
                        # gating for boxoverlap
                        if c<=self.min_overlap:
                            cost_row.append(c)
                        else:
                            cost_row.append(max_cost)
                    cost_matrix.append(cost_row)
                    # all ground truth trajectories are initially not associated
                    # extend groundtruth trajectories lists (merge lists)
                    seq_trajectories[gg.track_id].append(-1)
                    seq_ignored[gg.track_id].append(False)

                
                if len(g) is 0:
                    cost_matrix=[[]]
                # associate
                association_matrix = hm.compute(cost_matrix)

                # mapping for tracker ids and ground truth ids
                tmptp = 0
                tmpfp = 0
                tmpfn = 0
                tmpc  = 0
                this_cost = [-1]*len(g)

                associated_detections = []

                for row,col in association_matrix:
                    # apply gating on boxoverlap
                    c = cost_matrix[row][col]
                    if c < max_cost:

#                        associated_detections.append(col)
#                        det_objects[-1][-1].append(detObject(t[col].x1, t[col].x2, t[col].y1, t[col].y2, g[row].track_id, t[col].score))


                        g[row].tracker   = t[col].track_id
                        this_ids[1][row] = t[col].track_id
                        t[col].valid     = True
                        g[row].distance  = c
                        self.total_cost += 1-c
                        seqcost         += 1-c
                        tmpc            += 1-c
                        seq_trajectories[g[row].track_id][-1] = t[col].track_id

                        # true positives are only valid associations
                        self.tp += 1
                        tmptp   += 1
                        this_cost.append(c)
                    else:
                        g[row].tracker = -1
                        self.fn       += 1
                        tmpfn         += 1


#                for det_obj_idx in range(len(t)):
#                    if(not det_obj_idx in associated_detections):
#                        det_objects[-1][-1].append(detObject(t[det_obj_idx].x1, t[det_obj_idx].x2, t[det_obj_idx].y1, t[det_obj_idx].y2, assoc=-1, score=t[det_obj_idx].score))



                # associate tracker and DontCare areas
                # ignore tracker in neighboring classes
                nignoredtracker = 0
                for tt in t:
                    if (self.cls=="car" and tt.obj_type=="van") or (self.cls=="pedestrian" and tt.obj_type=="person_sitting"):
                        nignoredtracker+= 1
                        tt.ignored      = True
                        continue
                    for d in dc:
                        overlap = self.boxoverlap(tt,d,"a")
                        if overlap>0.5 and not tt.valid:
                            tt.ignored      = True
                            nignoredtracker+= 1
                            break

                # check for ignored FN/TP (truncation or neighboring object class)
                ignoredfn  = 0
                nignoredtp = 0
                for gg in g:
                    if gg.tracker < 0:
                        # ignored FN due to truncation
                        if gg.truncation>self.max_truncation:
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            ignoredfn += 1
                        # ignored FN due to neighboring object class
                        elif (self.cls=="car" and gg.obj_type=="van") or (self.cls=="pedestrian" and gg.obj_type=="person_sitting"):
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            ignoredfn += 1
                    elif gg.tracker>=0:
                        # ignored TP due to truncation
                        if gg.truncation>self.max_truncation:
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            nignoredtp += 1
                        # ignored TP due nieghboring object class
                        elif (self.cls=="car" and gg.obj_type=="van") or (self.cls=="pedestrian" and gg.obj_type=="person_sitting"):
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            nignoredtp += 1

                # correct TP by number of ignored TP due to truncation
                # ignored TP are shown as tracked in visualization
                tmptp -= nignoredtp
                self.n_gt -= (ignoredfn + nignoredtp)

                # false negatives = associated gt bboxes exceding association threshold + non-associated gt bboxes
                tmpfn   += len(g)-len(association_matrix)-ignoredfn
                self.fn += len(g)-len(association_matrix)-ignoredfn
                # false positives = tracker bboxes - associated tracker bboxes
                # mismatches (mme_t) 
                tmpfp   += len(t) - tmptp - nignoredtracker - nignoredtp
                self.fp += len(t) - tmptp - nignoredtracker - nignoredtp

                #jdk

                if include_ignored_gt and not include_ignored_detections:
                    for gg in g:
                        gt_objects[-1][-1].append(gtObject(gg.x1, gg.x2, gg.y1, gg.y2, gg.track_id))

                    for row,col in association_matrix:
                        # apply gating on boxoverlap
                        c = cost_matrix[row][col]
                        if c < max_cost and not t[col].ignored:
                            associated_detections.append(col)
                            det_objects[-1][-1].append(detObject(t[col].x1, t[col].x2, t[col].y1, t[col].y2, g[row].track_id, t[col].score))

                    for det_obj_idx in range(len(t)):
                        assert(t[det_obj_idx].obj_type == self.cls)
                        if(not det_obj_idx in associated_detections and not t[det_obj_idx].ignored):
                            det_objects[-1][-1].append(detObject(t[det_obj_idx].x1, t[det_obj_idx].x2, t[det_obj_idx].y1, t[det_obj_idx].y2, assoc=-1, score=t[det_obj_idx].score))

                elif not include_ignored_gt and not include_ignored_detections:
                    for gg in g:
                        if(not gg.ignored):
                            gt_objects[-1][-1].append(gtObject(gg.x1, gg.x2, gg.y1, gg.y2, gg.track_id))

                    for row,col in association_matrix:
                        # apply gating on boxoverlap
                        c = cost_matrix[row][col]
                        if c < max_cost and not g[row].ignored and not t[col].ignored:
                            associated_detections.append(col)
                            det_objects[-1][-1].append(detObject(t[col].x1, t[col].x2, t[col].y1, t[col].y2, g[row].track_id, t[col].score))

                    for det_obj_idx in range(len(t)):
                        assert(t[det_obj_idx].obj_type == self.cls)
                        if(not det_obj_idx in associated_detections and not t[det_obj_idx].ignored):
                            det_objects[-1][-1].append(detObject(t[det_obj_idx].x1, t[det_obj_idx].x2, t[det_obj_idx].y1, t[det_obj_idx].y2, assoc=-1, score=t[det_obj_idx].score))



                elif include_ignored_gt and include_ignored_detections:
                    for gg in g:
                        gt_objects[-1][-1].append(gtObject(gg.x1, gg.x2, gg.y1, gg.y2, gg.track_id))

                    for row,col in association_matrix:
                        # apply gating on boxoverlap
                        c = cost_matrix[row][col]
                        if c < max_cost:
                            associated_detections.append(col)
                            det_objects[-1][-1].append(detObject(t[col].x1, t[col].x2, t[col].y1, t[col].y2, g[row].track_id, t[col].score))

                    for det_obj_idx in range(len(t)):
                        assert(t[det_obj_idx].obj_type == self.cls)
                        if(not det_obj_idx in associated_detections):
                            det_objects[-1][-1].append(detObject(t[det_obj_idx].x1, t[det_obj_idx].x2, t[det_obj_idx].y1, t[det_obj_idx].y2, assoc=-1, score=t[det_obj_idx].score))

                elif not include_ignored_gt and include_ignored_detections:
                    for gg in g:
                        if(not gg.ignored):
                            gt_objects[-1][-1].append(gtObject(gg.x1, gg.x2, gg.y1, gg.y2, gg.track_id))

                    for row,col in association_matrix:
                        # apply gating on boxoverlap
                        c = cost_matrix[row][col]
                        if c < max_cost and not g[row].ignored:
                            associated_detections.append(col)
                            det_objects[-1][-1].append(detObject(t[col].x1, t[col].x2, t[col].y1, t[col].y2, g[row].track_id, t[col].score))

                    for det_obj_idx in range(len(t)):
                        assert(t[det_obj_idx].obj_type == self.cls)
                        if(not det_obj_idx in associated_detections):
                            det_objects[-1][-1].append(detObject(t[det_obj_idx].x1, t[det_obj_idx].x2, t[det_obj_idx].y1, t[det_obj_idx].y2, assoc=-1, score=t[det_obj_idx].score))
                else:
                    assert(false)

                #jdk



                # append single distance values
                self.distance.append(this_cost)

                # update sequence data
                seqtp += tmptp
                seqfp += tmpfp
                seqfn += tmpfn

                # sanity checks
                if tmptp + tmpfn is not len(g)-ignoredfn-nignoredtp:
                    print "seqidx", seq_idx
                    print "frame ", f
                    print "TP    ", tmptp
                    print "FN    ", tmpfn
                    print "FP    ", tmpfp
                    print "nGT   ", len(g)
                    print "nAss  ", len(association_matrix)
                    print "ign GT", ignoredfn
                    print "ign TP", nignoredtp
                    raise NameError("Something went wrong! nGroundtruth is not TP+FN")
                if tmptp+tmpfp+nignoredtracker+nignoredtp is not len(t):
                    print seq_idx, f, len(t), tmptp, tmpfp
                    print len(association_matrix), association_matrix
                    raise NameError("Something went wrong! nTracker is not TP+FP")

                # check for id switches or fragmentations
                for i,tt in enumerate(this_ids[0]):
                    if tt in last_ids[0]:
                        idx = last_ids[0].index(tt)
                        tid = this_ids[1][i]
                        lid = last_ids[1][idx]
                        if tid != lid and lid != -1 and tid != -1:
                            if g[i].truncation<self.max_truncation:
                                g[i].id_switch = 1
                                ids +=1
                        if tid != lid and lid != -1:
                            if g[i].truncation < self.max_truncation:
                                g[i].fragmentation = 1
                                tmp_frags +=1
                                fr +=1    

                # save current index
                last_ids = this_ids
                # compute MOTP_t
                MODP_t = 0
                if tmptp!=0:
                    MODP_t = tmpc/float(tmptp)
                self.MODP_t.append(MODP_t)


            # remove empty lists for current gt trajectories
            self.gt_trajectories[seq_idx]  = seq_trajectories
            self.ign_trajectories[seq_idx] = seq_ignored


        # compute MT/PT/ML, fragments, idswitches for all groundtruth trajectories
        n_ignored_tr_total = 0
        for seq_idx, (seq_trajectories,seq_ignored) in enumerate(zip(self.gt_trajectories, self.ign_trajectories)):
            if len(seq_trajectories)==0:
                continue
            tmpMT, tmpML, tmpPT, tmpId_switches, tmpFragments = [0]*5
            n_ignored_tr = 0
            for g, ign_g in zip(seq_trajectories.values(), seq_ignored.values()):
                # all frames of this gt trajectory are ignored
                if all(ign_g):
                    n_ignored_tr+=1
                    n_ignored_tr_total+=1
                    continue
                if all([this==-1 for this in g]):
                    tmpML+=1
                    self.ML+=1
                    continue
                # compute tracked frames in trajectory
                last_id = g[0]
                # first detection (necessary to be in gt_trajectories) is always tracked
                tracked = 1 if g[0]>=0 else 0
                lgt = 0 if ign_g[0] else 1
                for f in range(1,len(g)):
                    if ign_g[f]:
                        last_id = -1
                        continue
                    lgt+=1
                    if last_id != g[f] and last_id != -1 and g[f] != -1 and g[f-1] != -1:
                        tmpId_switches   += 1
                        self.id_switches += 1
                    if f < len(g)-1 and g[f-1] != g[f] and last_id != -1  and g[f] != -1 and g[f+1] != -1:
                        tmpFragments   += 1
                        self.fragments += 1
                    if g[f] != -1:
                        tracked += 1
                        last_id = g[f]
                # handle last frame; tracked state is handeled in for loop (g[f]!=-1)
                if len(g)>1 and g[f-1] != g[f] and last_id != -1  and g[f] != -1 and not ign_g[f]:
                    tmpFragments   += 1
                    self.fragments += 1

                # compute MT/PT/ML
                tracking_ratio = tracked/float(len(g))
                if tracking_ratio > 0.8:
                    tmpMT   += 1
                    self.MT += 1
                elif tracking_ratio < 0.2:
                    tmpML   += 1
                    self.ML += 1
                else: # 0.2 <= tracking_ratio <= 0.8
                    tmpPT   += 1
                    self.PT += 1

        if (self.n_gt_trajectories-n_ignored_tr_total)==0:
            self.MT = 0.
            self.PT = 0.
            self.ML = 0.
        else:
            self.MT /= float(self.n_gt_trajectories-n_ignored_tr_total)
            self.PT /= float(self.n_gt_trajectories-n_ignored_tr_total)
            self.ML /= float(self.n_gt_trajectories-n_ignored_tr_total)

        # precision/recall etc.
        if (self.fp+self.tp)==0 or (self.tp+self.fn)==0:
            self.recall = 0.
            self.precision = 0.
        else:
            self.recall = self.tp/float(self.tp+self.fn)
            self.precision = self.tp/float(self.fp+self.tp)
        if (self.recall+self.precision)==0:
            self.F1 = 0.
        else:
            self.F1 = 2.*(self.precision*self.recall)/(self.precision+self.recall)
        if sum(self.n_frames)==0:
            self.FAR = "n/a"
        else:
            self.FAR = self.fp/float(sum(self.n_frames))

        # compute CLEARMOT
        if self.n_gt==0:
            self.MOTA = -float("inf")
            self.MODA = -float("inf")
        else:
            self.MOTA  = 1 - (self.fn + self.fp + self.id_switches)/float(self.n_gt)
            self.MODA  = 1 - (self.fn + self.fp) / float(self.n_gt)
        if self.tp==0:
            self.MOTP  = float("inf")
        else:
            self.MOTP  = self.total_cost / float(self.tp)
        if self.n_gt!=0:
            if self.id_switches==0:
                self.MOTAL = 1 - (self.fn + self.fp + self.id_switches)/float(self.n_gt)
            else:
                self.MOTAL = 1 - (self.fn + self.fp + math.log10(self.id_switches))/float(self.n_gt)
        else:
            self.MOTAL = -float("inf")
        if sum(self.n_frames)==0:
            self.MODP = "n/a"
        else:
            self.MODP = sum(self.MODP_t)/float(sum(self.n_frames))
        return (gt_objects, det_objects)

    def summary(self):
        mail.msg('-'*80)
        mail.msg("jdk's learned parameters".center(20,"#"))
        mail.msg(self.printEntry("clutter count probabilities: ", self.clutter_count_list))
        mail.msg(self.printEntry("p_target_emission: ", self.p_target_emission))
        mail.msg(self.printEntry("ground truth birth count probabilities: ", self.gt_birth_count_probs))
        mail.msg(self.printEntry("detection birth count probabilities: ", self.detection_birth_count_probs))
        mail.msg(self.printEntry("discontinuous_target_count (should be small!!, ignored when computing death probabilities): ", self.discontinuous_target_count))
        mail.msg(self.printEntry("targ_cnt_still_living_but_unassoc_after_n: ", self.targ_cnt_still_living_but_unassoc_after_n))
        mail.msg(self.printEntry("targ_cnt_dead_at_n: ", self.targ_cnt_dead_at_n))
        mail.msg(self.printEntry("total_death_count: ", self.total_death_count))
        mail.msg(self.printEntry("target_death_probabilities: ", self.target_death_probabilities))
        mail.msg(self.printEntry("freq_unassociated_frames_before_target_death: ", self.freq_unassociated_frames_before_target_death))
        mail.msg(self.printEntry("measurement error covariance matrix: ", np.cov(np.asarray(self.meas_errors).T)))
        mail.msg(self.printEntry("measurement error means: ", np.mean(np.asarray(self.meas_errors), 0)))
        mail.msg(self.printEntry("ground truth mean width: ", sum(self.gt_widths)/float(len(self.gt_widths))))
        mail.msg(self.printEntry("ground truth mean height: ", sum(self.gt_heights)/float(len(self.gt_heights))))
        mail.msg(self.printEntry("estimated Q: ", self.Q_estimate))


        mail.msg("tracking evaluation summary".center(80,"="))
        mail.msg(self.printEntry("Multiple Object Tracking Accuracy (MOTA)", self.MOTA))
        mail.msg(self.printEntry("Multiple Object Tracking Precision (MOTP)", self.MOTP))
        mail.msg(self.printEntry("Multiple Object Tracking Accuracy (MOTAL)", self.MOTAL))
        mail.msg(self.printEntry("Multiple Object Detection Accuracy (MODA)", self.MODA))
        mail.msg(self.printEntry("Multiple Object Detection Precision (MODP)", self.MODP))
        mail.msg("")
        mail.msg(self.printEntry("Recall", self.recall))
        mail.msg(self.printEntry("Precision", self.precision))
        mail.msg(self.printEntry("F1", self.F1))
        mail.msg(self.printEntry("False Alarm Rate", self.FAR))
        mail.msg("")
        mail.msg(self.printEntry("Mostly Tracked", self.MT))
        mail.msg(self.printEntry("Partly Tracked", self.PT))
        mail.msg(self.printEntry("Mostly Lost", self.ML))
        mail.msg("")
        mail.msg(self.printEntry("True Positives", self.tp))
        mail.msg(self.printEntry("False Positives", self.fp))
        mail.msg(self.printEntry("Missed Targets", self.fn))
        mail.msg(self.printEntry("ID-switches", self.id_switches))
        mail.msg(self.printEntry("Fragmentations", self.fragments))
        mail.msg("")
        mail.msg(self.printEntry("Ground Truth Objects", self.n_gt))
        mail.msg(self.printEntry("Ground Truth Trajectories", self.n_gt_trajectories))
        mail.msg(self.printEntry("Tracker Objects", self.n_tr))
        mail.msg(self.printEntry("Tracker Trajectories", self.n_tr_trajectories))
        mail.msg("="*80)
        #self.saveSummary()

    def printEntry(self, key, val,width=(43,10)):
        s_out =  key.ljust(width[0])
        if type(val)==int:
            s = "%%%dd" % width[1]
            s_out += s % val
        elif type(val)==float:
            s = "%%%df" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s"%val).rjust(width[1])
        return s_out

    def saveSummary(self):
        filename = os.path.join("./data/object_detections", self.det_method, "3rd_party_metrics.txt")
        open(filename, "w").close()
        dump = open(filename, "a")
        print>>dump, "MOTA,", self.MOTA
        print>>dump, "MOTP,", self.MOTP
        print>>dump, "MOTAL,", self.MOTAL
        print>>dump, "MODA,", self.MODA
        print>>dump, "MODP,", self.MODP
        #print>>dump, ""
        print>>dump, "Recall,", self.recall
        print>>dump, "Precision,", self.precision
        print>>dump, "F1,", self.F1
        print>>dump, "FAR,", self.FAR
        #print>>dump, ""
        print>>dump, "MT,", self.MT
        print>>dump, "PT,", self.PT
        print>>dump, "ML,", self.ML
        #print>>dump, ""
        print>>dump, "TP,", self.tp
        print>>dump, "FP,", self.fp
        print>>dump, "Misses,", self.fn
        print>>dump, "ID-switches,", self.id_switches
        print>>dump, "Fragmentations,", self.fragments
        #print>>dump, ""
        print>>dump, "Ground Truth Objects,", self.n_gt
        print>>dump, "Ground Truth Trajectories,", self.n_gt_trajectories
        print>>dump, "Tracker Objects,", self.n_tr 
        print>>dump, "Tracker Trajectories,", self.n_tr_trajectories
        dump.close()

    def saveToStats(self):
        self.summary()
        filename = os.path.join("./data/object_detections", self.det_method, "stats_%s.txt" % self.cls)
        dump = open(filename, "w+")
        print>>dump, "%.6f " * 21 \
                % (self.MOTA, self.MOTP, self.MOTAL, self.MODA, self.MODP, \
                   self.recall, self.precision, self.F1, self.FAR, \
                   self.MT, self.PT, self.ML, self.tp, self.fp, self.fn, self.id_switches, self.fragments, \
                   self.n_gt, self.n_gt_trajectories, self.n_tr, self.n_tr_trajectories)
        dump.close()
        filename = os.path.join("./data/object_detections", self.det_method, "description.txt")
        dump = open(filename, "w+")
        print>>dump, "MOTA", "MOTP", "MOTAL", "MODA", "MODP", "recall", "precision", "F1", "FAR",
        print>>dump, "MT", "PT", "ML", "tp", "fp", "fn", "id_switches", "fragments",
        print>>dump, "n_gt", "n_gt_trajectories", "n_tr", "n_tr_trajectories"

    def sequenceSummary(self):
        filename = os.path.join("./data/object_detections", self.det_method, self.dataset, "sequences.txt")
        open(filename, "w").close()
        dump = open(filename, "a")

        self.printSep("Sequence Evaluation")
        self.printSep()
        print "seq\t", "\t".join(self.seq_res[0].keys())
        print>>dump, "seq\t", "\t".join(self.seq_res[0].keys())
        for i,s in enumerate(self.seq_res):
            print i,"\t",
            print>>dump, i,"\t",
            for e in s.values():
                if type(e) is int:
                    print "%d" % e, "\t",
                    print>>dump,"%d\t" % e,                                                 
                elif type(e) is float:
                    print "%.3f" % e, "\t",
                    print>>dump, "%.3f\t" % e,
                else:
                    print "%s" % e, "\t",
                    print>>dump, "%s\t" % e,
            print ""
            print>>dump, ""

        self.printSep()
        dump.close()

def evaluate(min_score, det_method,mail,obj_class = "car", include_ignored_gt = False, include_dontcare_in_gt = False, include_ignored_detections = True):
    """
    Output:
    - gt_objects: gt_objects[i][j] is a list of all ground truth objects in the jth frame of the ith video sequence
    - det_objects: det_objects[i][j] is a list of all detected objects in the jth frame of the ith video sequence
    - include_ignored_gt: Boolean, should ignored ground truth objects be included when calculating probabilities? 
    - include_dontcare_in_gt: Boolean, should don't care ground truth objects be included when calculating probabilities? 
    - include_ignored_detections = Boolean, should ignored detections be included when calculating probabilities?
        False doesn't really make sense because when actually running without ground truth information we don't know
        whether or not a detection is ignored, but debugging. (An ignored detection is a detection not associated with
        a ground truth object that would be associated with a don't care ground truth object if they were included.  It 
        can also be a neighobring object type, e.g. "van" instead of "car", but this never seems to occur in the data.
        If this occured, it would make sense to try excluding these detections.)


    """
    # start evaluation and instanciated eval object

    if USE_PICKLED_DATA:
        if not os.path.exists(PICKELD_DATA_DIRECTORY):
            os.makedirs(PICKELD_DATA_DIRECTORY)

        data_filename = PICKELD_DATA_DIRECTORY + "/min_score_%f_det_method_%s_obj_class_%s_include_ignored_gt_%s_include_dontcare_gt_%s_include_ignored_det_%s.pickle" % \
                                                 (min_score, det_method, obj_class, include_ignored_gt, include_dontcare_in_gt, include_ignored_detections)
        lock_filename = data_filename + "_lock"

        if os.path.isfile(data_filename) and (not os.path.isfile(lock_filename)): 
            f = open(data_filename, 'r')
            (gt_objects, det_objects) = pickle.load(f)
            f.close()
            return (gt_objects, det_objects)


    mail.msg("Processing Result for KITTI Tracking Benchmark")
    classes = []
    assert(obj_class == "car" or obj_class == "pedestrian")
    e = trackingEvaluation(min_score, det_method=det_method, mail=mail,cls=obj_class)
    # load tracker data and check provided classes
    e.loadDetections()
    mail.msg("Evaluate Object Class: %s" % obj_class.upper())
    classes.append(obj_class)
    # load groundtruth data for this class
    if not e.loadGroundtruth(include_dontcare_in_gt):
        raise ValueError("Ground truth not found.")
    mail.msg("Loading Groundtruth - Success")
    # sanity checks
    if len(e.groundtruth) is not len(e.tracker):
        mail.msg("The uploaded data does not provide results for every sequence.")
        return False
    mail.msg("Loaded %d Sequences." % len(e.groundtruth))
    mail.msg("Start Evaluation...")
    # create needed directories, evaluate and save stats
    try:
        e.createEvalDir()
    except:
        mail.msg("Feel free to contact us (lenz@kit.edu), if you receive this error message:")
        mail.msg("   Caught exception while creating results.")
#        if e.compute3rdPartyMetrics():
#            e.saveToStats()
#        else:
#            mail.msg("There seem to be no true positives or false positives at all in the submitted data.")

    (gt_objects, det_objects) = e.compute3rdPartyMetrics(include_ignored_gt, include_dontcare_in_gt, include_ignored_detections)

    if USE_PICKLED_DATA and (not os.path.isfile(lock_filename)):
        f_lock = open(lock_filename, 'w')
        f_lock.write("locked\n")
        f_lock.close()

        f = open(data_filename, 'w')
        pickle.dump((gt_objects, det_objects), f)
        f.close()  

        os.remove(lock_filename)

    # finish
    if len(classes)==0:
        mail.msg("The uploaded results could not be evaluated. Check for format errors.")
        return False
    mail.msg("Thank you for participating in our benchmark!")
    return (gt_objects, det_objects)







def get_clutter_probabilities(det_objects):
    """
    Input:
    - det_objects: det_objects[i][j] is a list of all detected objects in the jth frame of the ith video sequence

    Output:
    - clutter_probabilities: clutter_probabilities[i] is (number of frames containing i clutter measurements)/(total number of frames)
    """
    total_frame_count = 0
    #largest number of clutter objects in a single frame
    max_clutter_count = 0
    #clutter_count_dict[5] = 18 means that 18 frames contain 5 clutter measurements
    clutter_count_dict = {}
    for seq_idx in range(len(det_objects)):
        for frame_idx in range(len(det_objects[seq_idx])):
            total_frame_count += 1
            cur_frame_clutter_count = 0
            for det_idx in range(len(det_objects[seq_idx][frame_idx])):
                if det_objects[seq_idx][frame_idx][det_idx].assoc == -1:
                    cur_frame_clutter_count += 1
            if cur_frame_clutter_count > max_clutter_count:
                max_clutter_count = cur_frame_clutter_count

            if cur_frame_clutter_count in clutter_count_dict:
                clutter_count_dict[cur_frame_clutter_count] += 1
            else:
                clutter_count_dict[cur_frame_clutter_count] = 1

    clutter_probabilities = [0 for i in range(max_clutter_count + 1)]
    for clutter_count, frequency in clutter_count_dict.iteritems():
        clutter_probabilities[clutter_count] = float(frequency)/float(total_frame_count)
    return clutter_probabilities



def apply_function_on_intervals(score_cutoffs, function):
    """
    Input:
    - score_cutoffs: a list specifying score intervals to calculate function on

    Output:
    - function_on_intervals: function_on_intervals is a list of length len(score_cutoffs)
        function_on_intervals[i] contains the output of function applied on the interval:
            * [score_cutoffs[i], score_cutoffs[i+1])  for i >= 0 and i < len(score_cutoffs) - 1
            * [score_cutoffs[i], infinity)            for i == len(score_cutoffs) - 1

    """
    
    function_on_intervals = []

    for i in range(len(score_cutoffs) - 1):
        cur_output = function(score_cutoffs[i], score_cutoffs[i+1])
        function_on_intervals.append(cur_output)

    i = len(score_cutoffs) - 1
    cur_output = function(score_cutoffs[i], float("inf"))
    function_on_intervals.append(cur_output)

    return function_on_intervals

def apply_function_on_intervals_2_det(score_cutoffs_det1, score_cutoffs_det2, function):
    """
    Input:
    - score_cutoffs: a list specifying score intervals to calculate function on

    Output:
    - function_on_intervals: function_on_intervals is a list of length len(score_cutoffs)
        function_on_intervals[i] contains the output of function applied on the interval:
            * [score_cutoffs[i], score_cutoffs[i+1])  for i >= 0 and i < len(score_cutoffs) - 1
            * [score_cutoffs[i], infinity)            for i == len(score_cutoffs) - 1

    """
    
    function_on_det1_intervals = []

    for i in range(len(score_cutoffs_det1) - 1):
        cur_output = function(score_cutoffs_det1[i], score_cutoffs_det1[i+1], -float("inf"), float("inf"))
        function_on_det1_intervals.append(cur_output[0])

    i = len(score_cutoffs_det1) - 1
    cur_output = function(score_cutoffs_det1[i], float("inf"), -float("inf"), float("inf"))
    function_on_det1_intervals.append(cur_output[0])

    
    function_on_det2_intervals = []

    for i in range(len(score_cutoffs_det2) - 1):
        cur_output = function(-float("inf"), float("inf"), score_cutoffs_det2[i], score_cutoffs_det2[i+1])
        function_on_det2_intervals.append(cur_output[1])

    i = len(score_cutoffs_det2) - 1
    cur_output = function(-float("inf"), float("inf"), score_cutoffs_det2[i], float("inf"))
    function_on_det2_intervals.append(cur_output[1])

    return (function_on_det1_intervals, function_on_det2_intervals)

class MultiDetections:
    def __init__(self, gt_objects, det_objects1, det_objects2, training_sequences):
        self.gt_objects = gt_objects
        self.det_objects1 = det_objects1
        self.det_objects2 = det_objects2
        self.store_associations_in_gt()

        # A list of sequence indices that will be used for training
        self.training_sequences = training_sequences 
    def store_associations_in_gt(self):
        """
        Store a reference to associated detections in every associated ground truth object
        Can have up to 2 detections associated with a ground truth here, so associations stored in a list
        """
        assert(len(self.gt_objects) == len(self.det_objects1))
        for seq_idx in range(len(self.gt_objects)):
            assert(len(self.gt_objects[seq_idx]) == len(self.det_objects1[seq_idx]))
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                for det_idx in range(len(self.det_objects1[seq_idx][frame_idx])):
                    if self.det_objects1[seq_idx][frame_idx][det_idx].assoc != -1:
                        match_found = False
                        #gt track_id this detection is associated with
                        cur_det_assoc = self.det_objects1[seq_idx][frame_idx][det_idx].assoc 
                        cur_det = self.det_objects1[seq_idx][frame_idx][det_idx]
                        for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                            if self.gt_objects[seq_idx][frame_idx][gt_idx].track_id == cur_det_assoc:
                                #we found the ground truth-detection match
                                assert(match_found == False)
                                match_found = True
                                if self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection:
                                    self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection.append(cur_det)
                                else:
                                    self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection = [cur_det]

                        assert(match_found == True)

        assert(len(self.gt_objects) == len(self.det_objects2))
        for seq_idx in range(len(self.gt_objects)):
            assert(len(self.gt_objects[seq_idx]) == len(self.det_objects2[seq_idx]))
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                for det_idx in range(len(self.det_objects2[seq_idx][frame_idx])):
                    if self.det_objects2[seq_idx][frame_idx][det_idx].assoc != -1:
                        match_found = False
                        #gt track_id this detection is associated with
                        cur_det_assoc = self.det_objects2[seq_idx][frame_idx][det_idx].assoc 
                        cur_det = self.det_objects2[seq_idx][frame_idx][det_idx]
                        for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                            if self.gt_objects[seq_idx][frame_idx][gt_idx].track_id == cur_det_assoc:
                                #we found the ground truth-detection match
                                assert(match_found == False)
                                match_found = True
                                if self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection:
                                    self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection.append(cur_det)
                                else:
                                    self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection = [cur_det]

                        assert(match_found == True)

    def get_birth_probabilities_score_range(self, min_score_det_1, max_score_det_1, min_score_det_2, max_score_det_2,\
                                            allow_target_rebirth = True):
        """
        Input:
        - min_score_det_1: detections must have score >= min_score_det_1 to be considered
        - max_score_det_1: detections must have score < max_score_det_1 to be considered
        - min_score_det_2: detections must have score >= min_score_det_2 to be considered
        - max_score_det_2: detections must have score < max_score_det_2 to be considered
        - allow_target_rebirth: boolean, specifies whether ground truth targets are allowed to die and be reborn.
            In the training data, if ignored ground truth is included, I think there are only two cases where a target
            dies and is reborn.  I'm not sure if this is an error, but it's small enough to not really make a difference
            either way.  If ignored ground truths are not included, the number of ground truth objects that die and are 
            reborn increases, but is still probably small enough to not make much of a difference (would be good to double check!)

        Output:
        - all_birth_probabilities: all_birth_probabilities[j] is 
            (number of frames containing j birth measurements with scores in the range [min_score_det_1, max_score_det_1])
            / (total number of frames)
            where a "birth measurement" is a measurement of a ground truth target that has not been associated
            with a detection (of any score value in this AllData instance) on any previous time instance
        """

        total_frame_count = 0
        #largest number of detection1 births in a single frame
        max_birth_count1 = 0
        #largest number of detection2 births in a single frame
        max_birth_count2 = 0
        #birth_count_dict[5] = 18 means that 18 frames contain 5 birth measurements
        birth_count_dict_det1 = {}
        birth_count_dict_det2 = {}
        assert(len(self.det_objects1) == len(self.det_objects2))
        for seq_idx in self.training_sequences:
            assert(len(self.det_objects1[seq_idx]) == len(self.det_objects2[seq_idx]))

            #contains ids of all ground truth tracks that have been previously associated with a detection
            previously_detected_gt_ids = []
            for frame_idx in range(len(self.det_objects1[seq_idx])):
                if allow_target_rebirth and frame_idx != 0:
                    this_frame_gt_ids = []
                    for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                        this_frame_gt_ids.append(self.gt_objects[seq_idx][frame_idx][gt_idx].track_id)
                    for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx-1])):
                        cur_gt_id = self.gt_objects[seq_idx][frame_idx-1][gt_idx].track_id
                        #removed detected gt objects that have died from previously_detected_gt_ids
                        #to allow for rebirth
                        if not(cur_gt_id in this_frame_gt_ids) and cur_gt_id in previously_detected_gt_ids:
                            previously_detected_gt_ids.remove(cur_gt_id)
                            assert(not cur_gt_id in previously_detected_gt_ids)


                total_frame_count += 1
                cur_frame_birth_count1 = 0
                for det_idx in range(len(self.det_objects1[seq_idx][frame_idx])):
                    if (not self.det_objects1[seq_idx][frame_idx][det_idx].assoc in previously_detected_gt_ids):
                        previously_detected_gt_ids.append(self.det_objects1[seq_idx][frame_idx][det_idx].assoc)
                        if (self.det_objects1[seq_idx][frame_idx][det_idx].score >= min_score_det_1 and \
                            self.det_objects1[seq_idx][frame_idx][det_idx].score < max_score_det_1):
                            cur_frame_birth_count1 += 1
                if cur_frame_birth_count1 > max_birth_count1:
                    max_birth_count1 = cur_frame_birth_count1

                if cur_frame_birth_count1 in birth_count_dict_det1:
                    birth_count_dict_det1[cur_frame_birth_count1] += 1
                else:
                    birth_count_dict_det1[cur_frame_birth_count1] = 1

                cur_frame_birth_count2 = 0
                for det_idx in range(len(self.det_objects2[seq_idx][frame_idx])):
                    if (not self.det_objects2[seq_idx][frame_idx][det_idx].assoc in previously_detected_gt_ids):
                        previously_detected_gt_ids.append(self.det_objects2[seq_idx][frame_idx][det_idx].assoc)
                        if (self.det_objects2[seq_idx][frame_idx][det_idx].score >= min_score_det_2 and \
                            self.det_objects2[seq_idx][frame_idx][det_idx].score < max_score_det_2):
                            cur_frame_birth_count2 += 1
                if cur_frame_birth_count2 > max_birth_count2:
                    max_birth_count2 = cur_frame_birth_count2

                if cur_frame_birth_count2 in birth_count_dict_det2:
                    birth_count_dict_det2[cur_frame_birth_count2] += 1
                else:
                    birth_count_dict_det2[cur_frame_birth_count2] = 1

        all_birth_probabilities_det1 = [0 for i in range(max_birth_count1 + 1)]
        for birth_count, frequency in birth_count_dict_det1.iteritems():
            all_birth_probabilities_det1[birth_count] = float(frequency)/float(total_frame_count)

        all_birth_probabilities_det2 = [0 for i in range(max_birth_count2 + 1)]
        for birth_count, frequency in birth_count_dict_det2.iteritems():
            all_birth_probabilities_det2[birth_count] = float(frequency)/float(total_frame_count)


        return (all_birth_probabilities_det1, all_birth_probabilities_det2)


    def get_death_count(self, time_unassociated, near_border):
        """
        Input:
        - time_unassociated: the number of time instances unassociated before target death
        - near_border: boolean, does the ground truth obect have to be near the border time_unassociated time instances
            after the current time (one time instance before death) or is
            it required to not be near the border at this time?

        Output:
        - count: the total number of targets that die after being unassociated but alive for time_unassociated
            time instances
        """
        count = 0
        for seq_idx in self.training_sequences:
            for frame_idx in range(len(self.gt_objects[seq_idx]) - 1 - time_unassociated):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    cur_gt_id = self.gt_objects[seq_idx][frame_idx][gt_idx].track_id
                    alive_correctly = True
                    near_border_correctly = (self.gt_objects[seq_idx][frame_idx][gt_idx].near_border == near_border)

                    if self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection == None:
                        initially_associated = False
                    else:
                        initially_associated = True
                    associated_correctly = initially_associated
                    for i in range(1, time_unassociated+1):
                        alive = False
                        associated = False
                        for j in range(len(self.gt_objects[seq_idx][frame_idx+i])):
                            if(cur_gt_id == self.gt_objects[seq_idx][frame_idx+i][j].track_id):
                                alive = True
                                if(self.gt_objects[seq_idx][frame_idx+i][j].associated_detection):
                                    associated = True
                                if(i == time_unassociated):
                                    near_border_correctly = (self.gt_objects[seq_idx][frame_idx + time_unassociated][j].near_border == near_border)

                        if(not alive):
                            alive_correctly = False
                        if(associated):
                            associated_correctly = False

                    died_correctly = True
                    for j in range(len(self.gt_objects[seq_idx][frame_idx+1+time_unassociated])):
                        if(cur_gt_id == self.gt_objects[seq_idx][frame_idx+1+time_unassociated][j].track_id):
                            died_correctly = False #target still alive

                    if(alive_correctly and associated_correctly and died_correctly and near_border_correctly and initially_associated):
                        count += 1
        return count


    def get_gt_ids_by_frame(self):
        """
        Output:
        all_gt_ids_by_frame: all_gt_ids_by_frame[i][j] is a list of all ground truth track ids that exist
            in frame j of sequence i
        all_assoc_gt_ids_by_frame: all_assoc_gt_ids_by_frame[i][j] is a list of ground truth track ids that exist
            in frame j of sequence i and are associated with a detection
        """
        all_gt_ids_by_frame = []
        all_assoc_gt_ids_by_frame = []
        for seq_idx in range(len(self.gt_objects)):
            all_gt_ids_by_frame.append([])
            all_assoc_gt_ids_by_frame.append([])
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                all_gt_ids_by_frame[seq_idx].append([])
                all_assoc_gt_ids_by_frame[seq_idx].append([])
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    all_gt_ids_by_frame[seq_idx][frame_idx].append(self.gt_objects[seq_idx][frame_idx][gt_idx].track_id)
                    if self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection:
                        all_assoc_gt_ids_by_frame[seq_idx][frame_idx].append(self.gt_objects[seq_idx][frame_idx][gt_idx].track_id)

        assert(len(all_gt_ids_by_frame) == len(self.gt_objects))
        assert(len(all_assoc_gt_ids_by_frame) == len(self.gt_objects))
        for seq_idx in range(len(self.gt_objects)):
            assert(len(all_gt_ids_by_frame[seq_idx]) == len(self.gt_objects[seq_idx]))
            assert(len(all_assoc_gt_ids_by_frame[seq_idx]) == len(self.gt_objects[seq_idx]))
            for frame_idx in range(len(self.gt_objects[seq_idx]) - 1):
                assert(len(all_gt_ids_by_frame[seq_idx][frame_idx]) == len(self.gt_objects[seq_idx][frame_idx]))

        return (all_gt_ids_by_frame, all_assoc_gt_ids_by_frame)


    def get_death_count1(self, time_unassociated, near_border):
        """
        Input:
        - time_unassociated: the number of time instances unassociated before target death
        - near_border: boolean, does the ground truth obect have to be near the border time_unassociated time instances
            after the current time (one time instance before death) or is
            it required to not be near the border at this time?

        Output:
        - count: the total number of targets that die after being unassociated but alive for time_unassociated
            time instances
        """
        count = 0
        never_associated_gt_count = 0

        gt_track_ids = []
        dead_target_ids = []
        target_count_that_die_multiple_times = 0

        (all_gt_ids_by_frame, all_assoc_gt_ids_by_frame) = self.get_gt_ids_by_frame()

        total_death_count = 0
        total_never_dead_count = 0

        for seq_idx in self.training_sequences:
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    if (not (seq_idx, self.gt_objects[seq_idx][frame_idx][gt_idx].track_id) in gt_track_ids):
                        gt_track_ids.append((seq_idx, self.gt_objects[seq_idx][frame_idx][gt_idx].track_id))
#            print "sequence ", seq_idx, " contains ", len(self.gt_objects[seq_idx][-1]),
#            print " objects alive in the last frame (index ",  len(self.gt_objects[seq_idx]) - 1, ") and ", len(self.gt_objects[seq_idx][-2]),
#            print " objects alive in the 2nd to last frame "
            #debug

            total_never_dead_count += len(self.gt_objects[seq_idx][-1])

            #end debug

            for frame_idx in range(len(self.gt_objects[seq_idx]) - 1):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    cur_gt_id = self.gt_objects[seq_idx][frame_idx][gt_idx].track_id
                    cur_gt_dies_next_step = not (cur_gt_id in all_gt_ids_by_frame[seq_idx][frame_idx + 1])
                    if cur_gt_dies_next_step:
                        if((seq_idx, cur_gt_id) in dead_target_ids):
                            target_count_that_die_multiple_times += 1
                        dead_target_ids.append((seq_idx, cur_gt_id))
                        total_death_count += 1
                        num_unassoc_steps = 0
                        alive = True
                        unassociated = True
                        while alive and unassociated and (frame_idx - num_unassoc_steps >= 0) and num_unassoc_steps <= time_unassociated:
                            alive = cur_gt_id in all_gt_ids_by_frame[seq_idx][frame_idx - num_unassoc_steps]
                            associated = cur_gt_id in all_assoc_gt_ids_by_frame[seq_idx][frame_idx - num_unassoc_steps]
                            if associated:
                                unassociated = False
                            else:
                                num_unassoc_steps += 1

                            if (not alive):
                                never_associated_gt_count += 1
                        if num_unassoc_steps == time_unassociated \
                            and self.gt_objects[seq_idx][frame_idx][gt_idx].near_border == near_border:
                            count += 1
#        print "never associated gt count = ", never_associated_gt_count
#        print "total death count = ", total_death_count
#        print "total number of targets that never die (alive in last frame of a sequence): ", total_never_dead_count
#        print "total number of targets = ", len(gt_track_ids)
#        print "number of targets that die more than once = ", target_count_that_die_multiple_times
        return count



    def get_living_count1(self, time_unassociated, near_border):
        """
        Input:
        - time_unassociated: the number of time instances unassociated before target death
        - near_border: boolean, does the ground truth obect have to be near the border time_unassociated time instances
            after the current time (one time instance before death) or is
            it required to not be near the border at this time?

        Output:
        - count: the total number of targets that are alive and unassociated the time instance after being unassociated for time_unassociated
            previous time instances (get_living_count(2) is the number of targets that are alive and unassociated after
            3 time instances from their last association)
        """
        count = 0
        total_gt_object_count = 0

        (all_gt_ids_by_frame, all_assoc_gt_ids_by_frame) = self.get_gt_ids_by_frame()


        for seq_idx in self.training_sequences:
            total_gt_object_count += len(self.gt_objects[seq_idx][-1])

            for frame_idx in range(len(self.gt_objects[seq_idx]) - 1):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    total_gt_object_count += 1
                    cur_gt_id = self.gt_objects[seq_idx][frame_idx][gt_idx].track_id
                    cur_gt_assoc = (cur_gt_id in all_assoc_gt_ids_by_frame[seq_idx][frame_idx])
                    cur_gt_unassoc_but_living_next_step = not (cur_gt_id in all_assoc_gt_ids_by_frame[seq_idx][frame_idx + 1]) and \
                                                          (cur_gt_id in all_gt_ids_by_frame[seq_idx][frame_idx + 1])
                    if cur_gt_assoc and cur_gt_unassoc_but_living_next_step:
                        num_unassoc_steps = 0
                        unassociated = True
                        alive = True
                        while unassociated and alive and (frame_idx + 2 + num_unassoc_steps < len(self.gt_objects[seq_idx])) and num_unassoc_steps <= time_unassociated:
                            alive = cur_gt_id in all_gt_ids_by_frame[seq_idx][frame_idx + 2 + num_unassoc_steps]
                            associated = cur_gt_id in all_assoc_gt_ids_by_frame[seq_idx][frame_idx + 2 + num_unassoc_steps]
                            if associated:
                                unassociated = False
                            elif alive:
                                num_unassoc_steps += 1

                        if num_unassoc_steps >= time_unassociated \
                            and self.gt_objects[seq_idx][frame_idx][gt_idx].near_border == near_border:
                            count += 1
        print "total gt object count = ", total_gt_object_count
        return count


    def get_living_count(self, time_unassociated, near_border):
        """
        Input:
        - time_unassociated: the number of unassociated time instances
        - near_border: boolean, does the ground truth obect have to be near the border time_unassociated time instances
            after the current time (one time instance before the final time it must be alive and unassociated) or is
            it required to not be near the border at this time?

        Output:
        - count: the total number of targets that are alive and unassociated the time instance after being unassociated for time_unassociated
            previous time instances (get_living_count(2) is the number of targets that are alive and unassociated after
            3 time instances from their last association)
        """
        count = 0
        for seq_idx in self.training_sequences:
            for frame_idx in range(len(self.gt_objects[seq_idx]) - 1 - time_unassociated):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):

                    cur_gt_id = self.gt_objects[seq_idx][frame_idx][gt_idx].track_id
                    alive_correctly = True
                    near_border_correctly = (self.gt_objects[seq_idx][frame_idx][gt_idx].near_border == near_border)
                    if self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection == None:
                        initially_associated = False
                    else:
                        initially_associated = True
                    associated_correctly = initially_associated
                    for i in range(1, time_unassociated + 2):
                        alive = False
                        associated = False
                        for j in range(len(self.gt_objects[seq_idx][frame_idx+i])):
                            if(cur_gt_id == self.gt_objects[seq_idx][frame_idx+i][j].track_id):
                                alive = True
                                if(self.gt_objects[seq_idx][frame_idx+i][j].associated_detection):
                                    associated = True
                                if(i == time_unassociated):
                                    near_border_correctly = (self.gt_objects[seq_idx][frame_idx + time_unassociated][j].near_border == near_border)

                        if(not alive):
                            alive_correctly = False
                        if(associated):
                            associated_correctly = False
                    if(alive_correctly and associated_correctly and near_border_correctly and initially_associated):
                        count += 1
        return count

    def get_death_probs(self, near_border):
        """
        Input:
        - near_border: boolean, death probabilities for ground truth obects near the border on
            their last time instance alive or not near the border?
        """
        death_probs = [-99]
        death_counts = []
        living_counts = []
        print '#'*80
        print "get_death_probs info: "
        for i in range(3):
            death_count = float(self.get_death_count(i, near_border))
            living_count = float(self.get_living_count(i, near_border))
            death_count1 = float(self.get_death_count1(i, near_border))
            living_count1 = float(self.get_living_count1(i, near_border))
            death_counts.append(death_count)
            living_counts.append(living_count)
            if death_count + living_count == 0:
                death_probs.append(1.0)
            else:
                death_probs.append(death_count/(death_count + living_count))

            print "time unassociated = %d:" % i, "death_count =", death_count, ", death_count1=", death_count1, ", living_count=", living_count, ", living_count1=", living_count1
        print '#'*80
        return (death_probs, death_counts, living_counts)

class AllData:
    def __init__(self, gt_objects, det_objects, training_sequences):
        self.gt_objects = gt_objects
        self.det_objects = det_objects
        self.store_associations_in_gt()

        # A list of sequence indices that will be used for training
        self.training_sequences = training_sequences

    def store_associations_in_gt(self):
        """
        Store a reference to associated detections in every associated ground truth object
        """
        assert(len(self.gt_objects) == len(self.det_objects))
        for seq_idx in range(len(self.gt_objects)):
            assert(len(self.gt_objects[seq_idx]) == len(self.det_objects[seq_idx]))
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                for det_idx in range(len(self.det_objects[seq_idx][frame_idx])):
                    if self.det_objects[seq_idx][frame_idx][det_idx].assoc != -1:
                        match_found = False
                        #gt track_id this detection is associated with
                        cur_det_assoc = self.det_objects[seq_idx][frame_idx][det_idx].assoc 
                        cur_det = self.det_objects[seq_idx][frame_idx][det_idx]
                        for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                            if self.gt_objects[seq_idx][frame_idx][gt_idx].track_id == cur_det_assoc:
                                #we found the ground truth-detection match
                                assert(match_found == False)
                                match_found = True
                                self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection = cur_det
                        assert(match_found == True)


    def get_prob_target_emission_by_score_range(self, min_score, max_score, debug=True):
        """
        store_associations_in_gt should be run on self.gt_objects before calling this function
        Return the probability that a ground truth target emits a measurement in the specified score range
        Input:
        - min_score: detections must have score >= min_score to be considered
        - max_score: detections must have score < max_score to be considered

        Output:
        - p_target_emission: probability that a ground truth target emits a measurement in the specified score range
        """
        total_gt_object_count = 0
        total_gt_det_associations = 0
        for seq_idx in self.training_sequences:
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    total_gt_object_count += 1
                    if self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection and \
                        self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection.score >= min_score and \
                        self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection.score < max_score:
                        #this gt_object was associated with a detection in the score range or in other words this
                        #target emitted a measurement on this time instance
                        total_gt_det_associations += 1

        p_target_emission = float(total_gt_det_associations)/float(total_gt_object_count)

        if debug:
            print '-'*10
            print "get_prob_target_emission_by_score_range debug info:"
            print "min_score = ", min_score, " max_score = ", max_score
            print "total_gt_det_associations = ", total_gt_det_associations
            print "total_gt_object_count = ", total_gt_object_count

        return p_target_emission

    def get_clutter_probabilities_score_range(self, min_score, max_score, debug=True):
        """
        Input:
        - min_score: detections must have score >= min_score to be considered
        - max_score: detections must have score < max_score to be considered

        Output:
        - all_clutter_probabilities: all_clutter_probabilities[j] is 
            (number of frames containing j clutter measurements with scores in the range [min_score, max_score])
            / (total number of frames)

        """
        if debug:
            total_frame_count = 0
            total_detection_count = 0
            total_clutter_count = 0
            #largest number of clutter objects in a single frame
            max_clutter_count = 0
            #clutter_count_dict[5] = 18 means that 18 frames contain 5 clutter measurements
            clutter_count_dict = {}
            for seq_idx in self.training_sequences:
                for frame_idx in range(len(self.det_objects[seq_idx])):
                    total_frame_count += 1
                    cur_frame_clutter_count = 0
                    for det_idx in range(len(self.det_objects[seq_idx][frame_idx])):
                        total_detection_count += 1
                        if (self.det_objects[seq_idx][frame_idx][det_idx].assoc == -1 and \
                            self.det_objects[seq_idx][frame_idx][det_idx].score >= min_score and \
                            self.det_objects[seq_idx][frame_idx][det_idx].score < max_score):
                            cur_frame_clutter_count += 1
                            total_clutter_count += 1
                    if cur_frame_clutter_count > max_clutter_count:
                        max_clutter_count = cur_frame_clutter_count

                    if cur_frame_clutter_count in clutter_count_dict:
                        clutter_count_dict[cur_frame_clutter_count] += 1
                    else:
                        clutter_count_dict[cur_frame_clutter_count] = 1

            clutter_probabilities = [0 for i in range(max_clutter_count + 1)]
            for clutter_count, frequency in clutter_count_dict.iteritems():
                clutter_probabilities[clutter_count] = float(frequency)/float(total_frame_count)

            print '-'*10
            print "get_clutter_probabilities_score_range debug info:"
            print "total_frame_count = ", total_frame_count
            print "total_detection_count = ", total_detection_count
            print "total_clutter_count = ", total_clutter_count

            return clutter_probabilities

        else:
            total_frame_count = 0
            #largest number of clutter objects in a single frame
            max_clutter_count = 0
            #clutter_count_dict[5] = 18 means that 18 frames contain 5 clutter measurements
            clutter_count_dict = {}
            for seq_idx in self.training_sequences:
                for frame_idx in range(len(self.det_objects[seq_idx])):
                    total_frame_count += 1
                    cur_frame_clutter_count = 0
                    for det_idx in range(len(self.det_objects[seq_idx][frame_idx])):
                        if (self.det_objects[seq_idx][frame_idx][det_idx].assoc == -1 and \
                            self.det_objects[seq_idx][frame_idx][det_idx].score >= min_score and \
                            self.det_objects[seq_idx][frame_idx][det_idx].score < max_score):
                            cur_frame_clutter_count += 1
                    if cur_frame_clutter_count > max_clutter_count:
                        max_clutter_count = cur_frame_clutter_count

                    if cur_frame_clutter_count in clutter_count_dict:
                        clutter_count_dict[cur_frame_clutter_count] += 1
                    else:
                        clutter_count_dict[cur_frame_clutter_count] = 1

            clutter_probabilities = [0 for i in range(max_clutter_count + 1)]
            for clutter_count, frequency in clutter_count_dict.iteritems():
                clutter_probabilities[clutter_count] = float(frequency)/float(total_frame_count)
            return clutter_probabilities



    def get_clutter_probabilities_score_range_condition_num_meas(self, min_score, max_score):
        """
        Input:
        - min_score: detections must have score >= min_score to be considered
        - max_score: detections must have score < max_score to be considered

        Output:
        - all_clutter_probabilities: all_clutter_probabilities[i][j] is 
            (number of frames containing j clutter measurements and i total measurements with scores in the range [min_score, max_score])
            / (total number of frames containing i measurements with scores in the range [min_score, max_score])

        - frame_count: frame_count[i] = the number of frames containing i measurements with scores in the range [min_score, max_score]
        """
        
        all_clutter_probabilities = {}
        frame_count = {}

        #the maximum number of detections in a single frame with scores in the specified range
        max_detections = 0 

        for seq_idx in range(len(self.det_objects)):
            for frame_idx in range(len(self.det_objects[seq_idx])):
                cur_frame_detections = 0
                for det_idx in range(len(self.det_objects[seq_idx][frame_idx])):
                    if(self.det_objects[seq_idx][frame_idx][det_idx].score >= min_score and \
                       self.det_objects[seq_idx][frame_idx][det_idx].score < max_score):
                        cur_frame_detections += 1
                if cur_frame_detections > max_detections:
                    max_detections = cur_frame_detections

        for number_detections in range(max_detections + 1):
            #number of frames containing number_detections detections in the specified range
            total_frame_count = 0
            #largest number of clutter objects in a single frame
            max_clutter_count = 0
            #clutter_count_dict[5] = 18 means that 18 frames contain 5 clutter measurements
            clutter_count_dict = {}
            for seq_idx in range(len(self.det_objects)):
                for frame_idx in range(len(self.det_objects[seq_idx])):
                    cur_frame_clutter_count = 0
                    cur_frame_detections = 0
                    for det_idx in range(len(self.det_objects[seq_idx][frame_idx])):
                        if(self.det_objects[seq_idx][frame_idx][det_idx].score >= min_score and \
                           self.det_objects[seq_idx][frame_idx][det_idx].score < max_score):
                            cur_frame_detections += 1
                            if self.det_objects[seq_idx][frame_idx][det_idx].assoc == -1:
                                cur_frame_clutter_count += 1
                    if(cur_frame_detections == number_detections):
                        total_frame_count += 1
                        if cur_frame_clutter_count > max_clutter_count:
                            max_clutter_count = cur_frame_clutter_count
                        if cur_frame_clutter_count in clutter_count_dict:
                            clutter_count_dict[cur_frame_clutter_count] += 1
                        else:
                            clutter_count_dict[cur_frame_clutter_count] = 1

    #        cur_clutter_probabilities = [0 for i in range(max_clutter_count + 1)]
            cur_clutter_probabilities = [0 for i in range(number_detections+1)]
            for clutter_count, frequency in clutter_count_dict.iteritems():
                cur_clutter_probabilities[clutter_count] = float(frequency)/float(total_frame_count)

            all_clutter_probabilities[number_detections] = cur_clutter_probabilities
            frame_count[number_detections] = total_frame_count

        return (all_clutter_probabilities, frame_count)

    def get_birth_probabilities_score_range(self, min_score, max_score):
        """
        Input:
        - min_score: detections must have score >= min_score to be considered
        - max_score: detections must have score < max_score to be considered

        Output:
        - all_birth_probabilities: all_birth_probabilities[j] is 
            (number of frames containing j birth measurements with scores in the range [min_score, max_score])
            / (total number of frames)
            where a "birth measurement" is a measurement of a ground truth target that has not been associated
            with a detection (of any score value in this AllData instance) on any previous time instance
        """
        
        total_frame_count = 0
        #largest number of clutter objects in a single frame
        max_birth_count = 0
        #birth_count_dict[5] = 18 means that 18 frames contain 5 birth measurements
        birth_count_dict = {}
        for seq_idx in self.training_sequences:
            #contains ids of all ground truth tracks that have been previously associated with a detection
            previously_detected_gt_ids = []
            for frame_idx in range(len(self.det_objects[seq_idx])):
                total_frame_count += 1
                cur_frame_birth_count = 0
                for det_idx in range(len(self.det_objects[seq_idx][frame_idx])):
                    if (not self.det_objects[seq_idx][frame_idx][det_idx].assoc in previously_detected_gt_ids):
                        previously_detected_gt_ids.append(self.det_objects[seq_idx][frame_idx][det_idx].assoc)
                        if (self.det_objects[seq_idx][frame_idx][det_idx].score >= min_score and \
                            self.det_objects[seq_idx][frame_idx][det_idx].score < max_score):
                            cur_frame_birth_count += 1
                if cur_frame_birth_count > max_birth_count:
                    max_birth_count = cur_frame_birth_count

                if cur_frame_birth_count in birth_count_dict:
                    birth_count_dict[cur_frame_birth_count] += 1
                else:
                    birth_count_dict[cur_frame_birth_count] = 1

        all_birth_probabilities = [0 for i in range(max_birth_count + 1)]
        for birth_count, frequency in birth_count_dict.iteritems():
            all_birth_probabilities[birth_count] = float(frequency)/float(total_frame_count)
        return all_birth_probabilities

    def get_R_score_range(self, min_score, max_score):
        """
        Input:
        - min_score: detections must have score >= min_score to be considered
        - max_score: detections must have score < max_score to be considered

        Output:
        - meas_noise_cov: the measurement noise covariance matrix for measurements (detections)
            in the given score range
        - meas_noise_mean: mean of measurement noise (hopefully close to 0 array) for measurements (detections)
            in the given score range
        """
        meas_errors = []

        for seq_idx in self.training_sequences:
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    if (self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection and \
                        self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection.score >= min_score and \
                        self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection.score < max_score):

                            gt_pos = np.array([self.gt_objects[seq_idx][frame_idx][gt_idx].x, 
                                               self.gt_objects[seq_idx][frame_idx][gt_idx].y])
                            meas_pos = np.array([self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection.x, 
                                                 self.gt_objects[seq_idx][frame_idx][gt_idx].associated_detection.y])
                            meas_errors.append(meas_pos - gt_pos)

        assert(len(meas_errors) != 0), ("There are no associated detections in the score range [%f,%f)" % (min_score, max_score))

        meas_noise_cov = np.cov(np.asarray(meas_errors).T)
        meas_noise_mean = np.mean(np.asarray(meas_errors), 0)
        return (meas_noise_cov, meas_noise_mean)

    def count_measurements(self, min_score, max_score):
        """
        Input:
        - min_score: detections must have score >= min_score to be considered
        - max_score: detections must have score < max_score to be considered

        Output:
        - num_measurements: the number of measurements (detected objects) with scores in this range
        """
        num_measurements = 0

        for seq_idx in self.training_sequences:
            for frame_idx in range(len(self.det_objects[seq_idx])):
                for det_idx in range(len(self.det_objects[seq_idx][frame_idx])):
                    if (self.det_objects[seq_idx][frame_idx][det_idx].score >= min_score and \
                        self.det_objects[seq_idx][frame_idx][det_idx].score < max_score):
                            num_measurements += 1
        return num_measurements







class TargetSet:
    """
    Contains ground truth states for all targets.  Also contains all generated measurements.
    """

    def __init__(self):
        self.living_targets = []
        self.all_targets = [] #alive and dead targets

        self.living_count = 0 #number of living targets
        self.total_count = 0 #number of living targets plus number of dead targets
        self.measurements = [] #generated measurements for a generative TargetSet 

class Measurement:
    #a collection of measurements at a single time instance
    def __init__(self, time = -1):
        #self.val is a list of numpy arrays of measurement x, y locations
        self.val = []
        #list of widths of each bounding box
        self.widths = []
        #list of widths of each bounding box        
        self.heights = []
        #list of scores for each individual measurement
        self.scores = []
        self.time = time


def doctor_clutter_probabilities(all_clutter_probabilities):
    for i in range(len(all_clutter_probabilities)):
        assert(all_clutter_probabilities[i][0] > .00001)
        assert(abs(sum(all_clutter_probabilities[i])-1.0) < .0000000001)
        num_zero_probs = all_clutter_probabilities[i].count(0)

        all_clutter_probabilities[i][0] -= .0000001

        if num_zero_probs != 0:
            for prob_idx in range(len(all_clutter_probabilities[i])):
                if all_clutter_probabilities[i][prob_idx] == 0:
                    all_clutter_probabilities[i][prob_idx] = .0000001/float(20+num_zero_probs)


        # += used to append a list to a list!!
        all_clutter_probabilities[i] += [.0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs), .0000001/float(20+num_zero_probs)]


def get_meas_target_set(training_sequences, score_intervals, det_method="lsvm", obj_class="car", doctor_clutter_probs=True, doctor_birth_probs=True,\
    print_info=False, include_ignored_gt = False, include_dontcare_in_gt = False, include_ignored_detections = True):
    """
    Input:
    - doctor_clutter_probs: if True, replace 0 probabilities with .0000001/float(20+num_zero_probs) and extend
        clutter probability list with 20 values of .0000001/20 and subtract .0000001 from element 0
    - doctor_birth_probs: if True then if any birth probability is 0 subtract .0000001 from element 0
        of its score interval's birth probability list and replacing zero elements with .0000001/(number of
        zero elements in the score interval's birth probability list)
    """
    mail = mailpy.Mail("")

    (gt_objects, det_objects) = evaluate(score_intervals[0], det_method,mail, obj_class="car", include_ignored_gt=include_ignored_gt,\
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)

    measurementTargetSetsBySequence = []

    for seq_idx in range(len(det_objects)):
        cur_seq_meas_target_set = TargetSet()
        for frame_idx in range(len(det_objects[seq_idx])):
            cur_frame_measurements = Measurement()
            cur_frame_measurements.time = frame_idx*.1 #frames are .1 seconds apart
            cur_fram_meas_unsorted = []
            for meas_idx in range(len(det_objects[seq_idx][frame_idx])):
                cur_meas = det_objects[seq_idx][frame_idx][meas_idx]

                meas_pos = np.array([cur_meas.x, cur_meas.y])
                meas_width = cur_meas.x2 - cur_meas.x1
                meas_height = cur_meas.y2 - cur_meas.y1
#                cur_frame_measurements.val.append(meas_pos)
#                cur_frame_measurements.widths.append(meas_width)
#                cur_frame_measurements.heights.append(meas_height)
#                cur_frame_measurements.scores.append(cur_meas.score)

                cur_fram_meas_unsorted.append((cur_meas.score, (meas_pos, meas_width, meas_height)))

            cur_frame_meas_sorted = sorted(cur_fram_meas_unsorted, key=lambda tup: tup[0], reverse = True)
            for i in range(len(cur_frame_meas_sorted)):
                cur_frame_measurements.val.append(cur_frame_meas_sorted[i][1][0])
                cur_frame_measurements.widths.append(cur_frame_meas_sorted[i][1][1])
                cur_frame_measurements.heights.append(cur_frame_meas_sorted[i][1][2])
                cur_frame_measurements.scores.append(cur_frame_meas_sorted[i][0])
            cur_seq_meas_target_set.measurements.append(cur_frame_measurements)


        measurementTargetSetsBySequence.append(cur_seq_meas_target_set)     
############################# now get params ###############################
    all_data = AllData(gt_objects, det_objects, training_sequences)
    target_emission_probs = apply_function_on_intervals(score_intervals, all_data.get_prob_target_emission_by_score_range)
    clutter_probabilities = apply_function_on_intervals(score_intervals, all_data.get_clutter_probabilities_score_range)
    birth_probabilities = apply_function_on_intervals(score_intervals, all_data.get_birth_probabilities_score_range)
    meas_noise_cov_and_mean = apply_function_on_intervals(score_intervals, all_data.get_R_score_range)
    if(doctor_clutter_probs):
        doctor_clutter_probabilities(clutter_probabilities)

    if(doctor_birth_probs):
        for cur_score_int_birth_probs in birth_probabilities:
            assert(abs(sum(cur_score_int_birth_probs)-1.0) < .0000000001)
            num_zero_probs = cur_score_int_birth_probs.count(0)
            if num_zero_probs != 0:
                assert(cur_score_int_birth_probs[0] > .00001)
                cur_score_int_birth_probs[0] -= .0000001
                for prob_idx in range(len(cur_score_int_birth_probs)):
                    if cur_score_int_birth_probs[prob_idx] == 0:
                        cur_score_int_birth_probs[prob_idx] = .0000001/float(num_zero_probs)

    meas_noise_covs = []
    for i in range(len(meas_noise_cov_and_mean)):
        meas_noise_covs.append(meas_noise_cov_and_mean[i][0])

    if print_info:
        print "get_meas_target_set() info:"
        num_measurements = apply_function_on_intervals(score_intervals, all_data.count_measurements)
        for i in range(len(score_intervals)):
            print '-'*10
            print "For detections with scores greater than ", score_intervals[i]
            print "Number of detections = ", num_measurements[i]
            print "Target emission probabilities: ", target_emission_probs[i]
            print "Clutter probabilities", clutter_probabilities[i]
            print "Birth probabilities", birth_probabilities[i]
            print "Measurement noise covariance matrix:"
            print meas_noise_cov_and_mean[i][0]
            print "Measurement noise mean:"
            print meas_noise_cov_and_mean[i][1]

    return (measurementTargetSetsBySequence, target_emission_probs, clutter_probabilities, birth_probabilities, meas_noise_covs)

#    f = open("KITTI_measurements_%s_%s_min_score_%s.pickle" % (self.cls, self.det_method, MIN_SCORE), 'w')
#    pickle.dump(self.measurementTargetSetsBySequence, f)
#    f.close()  


def doctor_birth_probabilities(birth_probabilities):
    """
    If any birth probability is 0 subtract .0000001 from element 0
    of its score interval's birth probability list and replace zero elements with .0000001/(number of
    zero elements in the score interval's birth probability list
    """
    for cur_score_int_birth_probs in birth_probabilities:
        assert(abs(sum(cur_score_int_birth_probs)-1.0) < .0000000001)
        num_zero_probs = cur_score_int_birth_probs.count(0)
        if num_zero_probs != 0:
            assert(cur_score_int_birth_probs[0] > .00001)
            cur_score_int_birth_probs[0] -= .0000001
            for prob_idx in range(len(cur_score_int_birth_probs)):
                if cur_score_int_birth_probs[prob_idx] == 0:
                    cur_score_int_birth_probs[prob_idx] = .0000001/float(num_zero_probs)


def get_meas_target_sets_lsvm_and_regionlets(training_sequences, regionlets_score_intervals, lsvm_score_intervals, \
    obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, include_ignored_gt = False, \
    include_dontcare_in_gt = False, include_ignored_detections = True):
    """
    Input:
    - doctor_clutter_probs: if True, replace 0 probabilities with .0000001/float(20+num_zero_probs) and extend
        clutter probability list with 20 values of .0000001/20 and subtract .0000001 from element 0
    """

    print "HELLO#1"
    (measurementTargetSetsBySequence_regionlets, target_emission_probs_regionlets, clutter_probabilities_regionlets, \
        incorrect_birth_probabilities_regionlets, meas_noise_covs_regionlets) = get_meas_target_set(training_sequences, regionlets_score_intervals, \
        "regionlets", obj_class, doctor_clutter_probs=doctor_clutter_probs, doctor_birth_probs=doctor_birth_probs, include_ignored_gt=include_ignored_gt, \
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#2"

    (measurementTargetSetsBySequence_lsvm, target_emission_probs_lsvm, clutter_probabilities_lsvm, \
        incorrect_birth_probabilities_lsvm, meas_noise_covs_lsvm) = get_meas_target_set(training_sequences, lsvm_score_intervals, \
        "lsvm", obj_class, doctor_clutter_probs=doctor_clutter_probs, doctor_birth_probs=doctor_birth_probs, include_ignored_gt=include_ignored_gt, \
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#3"



    returnTargSets = []
    assert(len(measurementTargetSetsBySequence_lsvm) == len(measurementTargetSetsBySequence_regionlets))
    for seq_idx in range(len(measurementTargetSetsBySequence_lsvm)):
        returnTargSets.append([measurementTargetSetsBySequence_regionlets[seq_idx],\
                               measurementTargetSetsBySequence_lsvm[seq_idx]])
    print "HELLO#4"

    emission_probs = [target_emission_probs_regionlets, target_emission_probs_lsvm]
    clutter_probs = [clutter_probabilities_regionlets, clutter_probabilities_lsvm]
    meas_noise_covs = [meas_noise_covs_regionlets, meas_noise_covs_lsvm]
    print "HELLO#5"

    mail = mailpy.Mail("") #this is silly and could be cleaned up
    (gt_objects, regionlets_det_objects) = evaluate(min_score=regionlets_score_intervals[0], \
        det_method='regionlets', mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#6"

    (gt_objects, lsvm_det_objects) = evaluate(min_score=lsvm_score_intervals[0], \
        det_method='lsvm', mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    multi_detections = MultiDetections(gt_objects, regionlets_det_objects, lsvm_det_objects, training_sequences)
    print "HELLO#7"

    (birth_probabilities_regionlets, birth_probabilities_lsvm) = apply_function_on_intervals_2_det(regionlets_score_intervals, \
        lsvm_score_intervals, multi_detections.get_birth_probabilities_score_range)

    if(doctor_birth_probs):
        doctor_birth_probabilities(birth_probabilities_regionlets)
        doctor_birth_probabilities(birth_probabilities_lsvm)

    birth_probabilities = [birth_probabilities_regionlets, birth_probabilities_lsvm]
    print "HELLO#8"

    (death_probs_near_border, death_counts_near_border, living_counts_near_border) = multi_detections.get_death_probs(near_border = True)
    (death_probs_not_near_border, death_counts_not_near_border, living_counts_not_near_border) = multi_detections.get_death_probs(near_border = False)


    return (returnTargSets, emission_probs, clutter_probs, birth_probabilities, meas_noise_covs, death_probs_near_border, death_probs_not_near_border)




def get_meas_target_sets_regionlets_general_format(training_sequences, regionlets_score_intervals, \
    obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, include_ignored_gt = False, \
    include_dontcare_in_gt = False, include_ignored_detections = True):
    """
    Input:
    - doctor_clutter_probs: if True, replace 0 probabilities with .0000001/float(20+num_zero_probs) and extend
        clutter probability list with 20 values of .0000001/20 and subtract .0000001 from element 0
    """

    print "HELLO#1"
    (measurementTargetSetsBySequence_regionlets, target_emission_probs_regionlets, clutter_probabilities_regionlets, \
        incorrect_birth_probabilities_regionlets, meas_noise_covs_regionlets) = get_meas_target_set(training_sequences, regionlets_score_intervals, \
        "regionlets", obj_class, doctor_clutter_probs=doctor_clutter_probs, doctor_birth_probs=doctor_birth_probs, include_ignored_gt=include_ignored_gt, \
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#2"


    returnTargSets = []
    for seq_idx in range(len(measurementTargetSetsBySequence_regionlets)):
        returnTargSets.append([measurementTargetSetsBySequence_regionlets[seq_idx]])
    print "HELLO#4"

    emission_probs = [target_emission_probs_regionlets]
    clutter_probs = [clutter_probabilities_regionlets]
    meas_noise_covs = [meas_noise_covs_regionlets]
    print "HELLO#5"

    mail = mailpy.Mail("") #this is silly and could be cleaned up
    (gt_objects, regionlets_det_objects) = evaluate(min_score=regionlets_score_intervals[0], \
        det_method='regionlets', mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#6"

########### CLEAN THIS UP BEGIN
#    lsvm_score_intervals = [2] #arbitrary!
#    (gt_objects, lsvm_det_objects) = evaluate(min_score=lsvm_score_intervals[0], \
#        det_method='lsvm', mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
#        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    multi_detections = MultiDetections(gt_objects, regionlets_det_objects, regionlets_det_objects, training_sequences)
    print "HELLO#7"

    (birth_probabilities_regionlets, birth_probabilities_lsvm_nonsense) = apply_function_on_intervals_2_det(regionlets_score_intervals, \
        regionlets_score_intervals, multi_detections.get_birth_probabilities_score_range)

    (death_probs_near_border, death_counts_near_border, living_counts_near_border) = multi_detections.get_death_probs(near_border = True)
    (death_probs_not_near_border, death_counts_not_near_border, living_counts_not_near_border) = multi_detections.get_death_probs(near_border = False)

    if(doctor_birth_probs):
        doctor_birth_probabilities(birth_probabilities_regionlets)
        doctor_birth_probabilities(birth_probabilities_lsvm_nonsense)

########## CLEAN THIS UP END
    birth_probabilities = [birth_probabilities_regionlets]
    print "HELLO#8"

    return (returnTargSets, emission_probs, clutter_probs, birth_probabilities, meas_noise_covs, death_probs_near_border, death_probs_not_near_border)


def get_meas_target_sets_mscnn_and_regionlets(training_sequences, mscnn_score_intervals, regionlets_score_intervals, \
    obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, include_ignored_gt = False, \
    include_dontcare_in_gt = False, include_ignored_detections = True):
    """
    Input:
    - doctor_clutter_probs: if True, replace 0 probabilities with .0000001/float(20+num_zero_probs) and extend
        clutter probability list with 20 values of .0000001/20 and subtract .0000001 from element 0
    """

    print "HELLO#1"
    (measurementTargetSetsBySequence_mscnn, target_emission_probs_mscnn, clutter_probabilities_mscnn, \
        incorrect_birth_probabilities_mscnn, meas_noise_covs_mscnn) = get_meas_target_set(training_sequences, mscnn_score_intervals, \
        "mscnn", obj_class, doctor_clutter_probs=doctor_clutter_probs, doctor_birth_probs=doctor_birth_probs, include_ignored_gt=include_ignored_gt, \
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#2"

    (measurementTargetSetsBySequence_regionlets, target_emission_probs_regionlets, clutter_probabilities_regionlets, \
        incorrect_birth_probabilities_regionlets, meas_noise_covs_regionlets) = get_meas_target_set(training_sequences, regionlets_score_intervals, \
        "regionlets", obj_class, doctor_clutter_probs=doctor_clutter_probs, doctor_birth_probs=doctor_birth_probs, include_ignored_gt=include_ignored_gt, \
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#3"



    returnTargSets = []
    assert(len(measurementTargetSetsBySequence_regionlets) == len(measurementTargetSetsBySequence_mscnn))
    for seq_idx in range(len(measurementTargetSetsBySequence_regionlets)):
        returnTargSets.append([measurementTargetSetsBySequence_mscnn[seq_idx],\
                               measurementTargetSetsBySequence_regionlets[seq_idx]])
    print "HELLO#4"

    emission_probs = [target_emission_probs_mscnn, target_emission_probs_regionlets]
    clutter_probs = [clutter_probabilities_mscnn, clutter_probabilities_regionlets]
    meas_noise_covs = [meas_noise_covs_mscnn, meas_noise_covs_regionlets]
    print "HELLO#5"

    mail = mailpy.Mail("") #this is silly and could be cleaned up
    (gt_objects, mscnn_det_objects) = evaluate(min_score=mscnn_score_intervals[0], \
        det_method='mscnn', mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#6"

    (gt_objects, regionlets_det_objects) = evaluate(min_score=regionlets_score_intervals[0], \
        det_method='regionlets', mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    multi_detections = MultiDetections(gt_objects, mscnn_det_objects, regionlets_det_objects, training_sequences)
    print "HELLO#7"

##############################################################################
    #calculate the joint measurement noise covariance between 2 mscnn and regionlets detections
    meas_errors = []
    multi_det_count = 0
    det_counts = [0,0,0]
    for seq_idx in training_sequences:
        for frame_idx in range(len(multi_detections.gt_objects[seq_idx])):
            for gt_obj in multi_detections.gt_objects[seq_idx][frame_idx]:
#                print type(gt_obj)
#                print gt_obj
#                print type(gt_obj[0])
#                print gt_obj[0]
                if(gt_obj.associated_detection):
                    num_det = len(gt_obj.associated_detection)
                else:    
                    num_det = 0
                assert(num_det in [0, 1, 2]), (num_det, gt_obj.associated_detection)
                det_counts[num_det] += 1
                if(num_det == 2):
                    cur_meas_error = np.array([gt_obj.x - gt_obj.associated_detection[0].x, 
                                            gt_obj.y - gt_obj.associated_detection[0].y,
                                            gt_obj.x - gt_obj.associated_detection[1].x, 
                                            gt_obj.y - gt_obj.associated_detection[1].y])
#                                            gt_obj.x - (gt_obj.associated_detection[1].x + gt_obj.associated_detection[0].x)/2.0, 
#                                            gt_obj.y - (gt_obj.associated_detection[1].y + gt_obj.associated_detection[0].y)/2.0])
                    meas_errors.append(cur_meas_error)
                    multi_det_count += 1
    joint_meas_noise_cov = np.cov(np.asarray(meas_errors).T)


##############################################################################

    (birth_probabilities_mscnn, birth_probabilities_regionlets) = apply_function_on_intervals_2_det(mscnn_score_intervals, \
        regionlets_score_intervals, multi_detections.get_birth_probabilities_score_range)

    if(doctor_birth_probs):
        doctor_birth_probabilities(birth_probabilities_mscnn)
        doctor_birth_probabilities(birth_probabilities_regionlets)

    birth_probabilities = [birth_probabilities_mscnn, birth_probabilities_regionlets]
    print "HELLO#8"

    (death_probs_near_border, death_counts_near_border, living_counts_near_border) = multi_detections.get_death_probs(near_border = True)
    (death_probs_not_near_border, death_counts_not_near_border, living_counts_not_near_border) = multi_detections.get_death_probs(near_border = False)


    return (returnTargSets, emission_probs, clutter_probs, birth_probabilities, meas_noise_covs, death_probs_near_border, death_probs_not_near_border, joint_meas_noise_cov)


def get_meas_target_sets_2sources_general(training_sequences, det1_score_intervals, det2_score_intervals, \
    det1_name, det2_name, obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, \
    include_ignored_gt = False, include_dontcare_in_gt = False, include_ignored_detections = True):
    """
    Input:
    - doctor_clutter_probs: if True, replace 0 probabilities with .0000001/float(20+num_zero_probs) and extend
        clutter probability list with 20 values of .0000001/20 and subtract .0000001 from element 0
    """

    print "HELLO#1"
    (measurementTargetSetsBySequence_det1, target_emission_probs_det1, clutter_probabilities_det1, \
        incorrect_birth_probabilities_det1, meas_noise_covs_det1) = get_meas_target_set(training_sequences, det1_score_intervals, \
        det1_name, obj_class, doctor_clutter_probs=doctor_clutter_probs, doctor_birth_probs=doctor_birth_probs, include_ignored_gt=include_ignored_gt, \
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#2"

    if det2_name:
        assert(det2_score_intervals != None)
        (measurementTargetSetsBySequence_det2, target_emission_probs_det2, clutter_probabilities_det2, \
            incorrect_birth_probabilities_det2, meas_noise_covs_det2) = get_meas_target_set(training_sequences, det2_score_intervals, \
            det2_name, obj_class, doctor_clutter_probs=doctor_clutter_probs, doctor_birth_probs=doctor_birth_probs, include_ignored_gt=include_ignored_gt, \
            include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#3"



    returnTargSets = []
    assert(len(measurementTargetSetsBySequence_det2) == len(measurementTargetSetsBySequence_det1))
    for seq_idx in range(len(measurementTargetSetsBySequence_det2)):
        if det2_name:
            returnTargSets.append([measurementTargetSetsBySequence_det1[seq_idx],\
                                   measurementTargetSetsBySequence_det2[seq_idx]])
        else:
            returnTargSets.append([measurementTargetSetsBySequence_det1[seq_idx]])

    print "HELLO#4"

    if det2_name:
        emission_probs = [target_emission_probs_det1, target_emission_probs_det2]
        clutter_probs = [clutter_probabilities_det1, clutter_probabilities_det2]
        meas_noise_covs = [meas_noise_covs_det1, meas_noise_covs_det2]
    else:
        emission_probs = [target_emission_probs_det1]
        clutter_probs = [clutter_probabilities_det1]
        meas_noise_covs = [meas_noise_covs_det1]

    print "HELLO#5"

    mail = mailpy.Mail("") #this is silly and could be cleaned up
    (gt_objects, det1_objects) = evaluate(min_score=det1_score_intervals[0], \
        det_method=det1_name, mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#6"

    if det2_name:
        (gt_objects, det2_objects) = evaluate(min_score=det2_score_intervals[0], \
            det_method=det2_name, mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
            include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
        multi_detections = MultiDetections(gt_objects, det1_objects, det2_objects, training_sequences)
        print "HELLO#7"

    ##############################################################################
        #calculate the joint measurement noise covariance between det1_name and det2_name detections
        meas_errors = []
        multi_det_count = 0
        det_counts = [0,0,0]
        for seq_idx in training_sequences:
            for frame_idx in range(len(multi_detections.gt_objects[seq_idx])):
                for gt_obj in multi_detections.gt_objects[seq_idx][frame_idx]:
    #                print type(gt_obj)
    #                print gt_obj
    #                print type(gt_obj[0])
    #                print gt_obj[0]
                    if(gt_obj.associated_detection):
                        num_det = len(gt_obj.associated_detection)
                    else:    
                        num_det = 0
                    assert(num_det in [0, 1, 2]), (num_det, gt_obj.associated_detection)
                    det_counts[num_det] += 1
                    if(num_det == 2):
                        cur_meas_error = np.array([gt_obj.x - gt_obj.associated_detection[0].x, 
                                                gt_obj.y - gt_obj.associated_detection[0].y,
                                                gt_obj.x - gt_obj.associated_detection[1].x, 
                                                gt_obj.y - gt_obj.associated_detection[1].y])
    #                                            gt_obj.x - (gt_obj.associated_detection[1].x + gt_obj.associated_detection[0].x)/2.0, 
    #                                            gt_obj.y - (gt_obj.associated_detection[1].y + gt_obj.associated_detection[0].y)/2.0])
                        meas_errors.append(cur_meas_error)
                        multi_det_count += 1
        joint_meas_noise_cov = np.cov(np.asarray(meas_errors).T)


    ##############################################################################

        (birth_probabilities_det1, birth_probabilities_det2) = apply_function_on_intervals_2_det(det1_score_intervals, \
            det2_score_intervals, multi_detections.get_birth_probabilities_score_range)

        if(doctor_birth_probs):
            doctor_birth_probabilities(birth_probabilities_det1)
            doctor_birth_probabilities(birth_probabilities_det2)

        birth_probabilities = [birth_probabilities_det1, birth_probabilities_det2]
        
    print "HELLO#8"

    (death_probs_near_border, death_counts_near_border, living_counts_near_border) = multi_detections.get_death_probs(near_border = True)
    (death_probs_not_near_border, death_counts_not_near_border, living_counts_not_near_border) = multi_detections.get_death_probs(near_border = False)


    return (returnTargSets, emission_probs, clutter_probs, birth_probabilities, meas_noise_covs, death_probs_near_border, death_probs_not_near_border, joint_meas_noise_cov)


def get_meas_target_sets_mscnn_general_format(training_sequences, mscnn_score_intervals, \
    obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, include_ignored_gt = False, \
    include_dontcare_in_gt = False, include_ignored_detections = True):
    """
    Input:
    - doctor_clutter_probs: if True, replace 0 probabilities with .0000001/float(20+num_zero_probs) and extend
        clutter probability list with 20 values of .0000001/20 and subtract .0000001 from element 0
    """

    print "HELLO#1"
    (measurementTargetSetsBySequence_mscnn, target_emission_probs_mscnn, clutter_probabilities_mscnn, \
        incorrect_birth_probabilities_mscnn, meas_noise_covs_mscnn) = get_meas_target_set(training_sequences, mscnn_score_intervals, \
        "mscnn", obj_class, doctor_clutter_probs=doctor_clutter_probs, doctor_birth_probs=doctor_birth_probs, include_ignored_gt=include_ignored_gt, \
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#2"


    returnTargSets = []
    for seq_idx in range(len(measurementTargetSetsBySequence_mscnn)):
        returnTargSets.append([measurementTargetSetsBySequence_mscnn[seq_idx]])
    print "HELLO#4"

    emission_probs = [target_emission_probs_mscnn]
    clutter_probs = [clutter_probabilities_mscnn]
    meas_noise_covs = [meas_noise_covs_mscnn]
    print "HELLO#5"

    mail = mailpy.Mail("") #this is silly and could be cleaned up
    (gt_objects, mscnn_det_objects) = evaluate(min_score=mscnn_score_intervals[0], \
        det_method='mscnn', mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#6"

########### CLEAN THIS UP BEGIN
#    lsvm_score_intervals = [2] #arbitrary!
#    (gt_objects, lsvm_det_objects) = evaluate(min_score=lsvm_score_intervals[0], \
#        det_method='lsvm', mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
#        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    multi_detections = MultiDetections(gt_objects, mscnn_det_objects, mscnn_det_objects, training_sequences)
    print "HELLO#7"

    (birth_probabilities_mscnn, birth_probabilities_lsvm_nonsense) = apply_function_on_intervals_2_det(mscnn_score_intervals, \
        mscnn_score_intervals, multi_detections.get_birth_probabilities_score_range)

    (death_probs_near_border, death_counts_near_border, living_counts_near_border) = multi_detections.get_death_probs(near_border = True)
    (death_probs_not_near_border, death_counts_not_near_border, living_counts_not_near_border) = multi_detections.get_death_probs(near_border = False)

    if(doctor_birth_probs):
        doctor_birth_probabilities(birth_probabilities_mscnn)
        doctor_birth_probabilities(birth_probabilities_lsvm_nonsense)

########## CLEAN THIS UP END
    birth_probabilities = [birth_probabilities_mscnn]
    print "HELLO#8"

    return (returnTargSets, emission_probs, clutter_probs, birth_probabilities, meas_noise_covs, death_probs_near_border, death_probs_not_near_border)


def get_meas_target_sets_1sources_general(training_sequences, det_score_intervals, det_name, \
    obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, include_ignored_gt = False, \
    include_dontcare_in_gt = False, include_ignored_detections = True):
    """
    Input:
    - doctor_clutter_probs: if True, replace 0 probabilities with .0000001/float(20+num_zero_probs) and extend
        clutter probability list with 20 values of .0000001/20 and subtract .0000001 from element 0
    """

    print "HELLO#1"
    (measurementTargetSetsBySequence, target_emission_probs, clutter_probabilities, \
        incorrect_birth_probabilities, meas_noise_covs) = get_meas_target_set(training_sequences, det_score_intervals, \
        det_name, obj_class, doctor_clutter_probs=doctor_clutter_probs, doctor_birth_probs=doctor_birth_probs, include_ignored_gt=include_ignored_gt, \
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#2"


    returnTargSets = []
    for seq_idx in range(len(measurementTargetSetsBySequence)):
        returnTargSets.append([measurementTargetSetsBySequence[seq_idx]])
    print "HELLO#4"

    emission_probs = [target_emission_probs]
    clutter_probs = [clutter_probabilities]
    meas_noise_covs = [meas_noise_covs]
    print "HELLO#5"

    mail = mailpy.Mail("") #this is silly and could be cleaned up
    (gt_objects, det_objects) = evaluate(min_score=det_score_intervals[0], \
        det_method=det_name, mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    print "HELLO#6"

########### CLEAN THIS UP BEGIN
#    lsvm_score_intervals = [2] #arbitrary!
#    (gt_objects, lsvm_det_objects) = evaluate(min_score=lsvm_score_intervals[0], \
#        det_method='lsvm', mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
#        include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
    multi_detections = MultiDetections(gt_objects, det_objects, det_objects, training_sequences)
    print "HELLO#7"

    (birth_probabilities, birth_probabilities_lsvm_nonsense) = apply_function_on_intervals_2_det(det_score_intervals, \
        det_score_intervals, multi_detections.get_birth_probabilities_score_range)

    (death_probs_near_border, death_counts_near_border, living_counts_near_border) = multi_detections.get_death_probs(near_border = True)
    (death_probs_not_near_border, death_counts_not_near_border, living_counts_not_near_border) = multi_detections.get_death_probs(near_border = False)

    if(doctor_birth_probs):
        doctor_birth_probabilities(birth_probabilities)
        doctor_birth_probabilities(birth_probabilities_lsvm_nonsense)

########## CLEAN THIS UP END
    birth_probabilities = [birth_probabilities]
    print "HELLO#8"

    return (returnTargSets, emission_probs, clutter_probs, birth_probabilities, meas_noise_covs, death_probs_near_border, death_probs_not_near_border)



def boxoverlap(a,b,criterion="union"):
    """
        boxoverlap computes intersection over union for bbox a and b in KITTI format.
        If the criterion is 'union', overlap = (a inter b) / a union b).
        If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
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
    elif criterion.lower()=="a":
        o = float(inter) / float(aarea)
    else:
        raise TypeError("Unkown type for criterion")
    return o


class MultiDetections_many:
    def __init__(self, gt_objects, all_det_objects, training_sequences):
        self.gt_objects = gt_objects #list of lists where gt_objects[i][j] is the jth gt_object in sequence i
        #dictionary where all_det_objects['det_name'] contains the detected objects of type 'det_name'
        self.all_det_objects = all_det_objects
        #self.clutter_detections[seq_idx][frame_idx], clutter_groups for the specified sequence and frame
        #clutter_groups, list where each element is a clutter_group
        #clutter_group, dictionary of clutter detections in the group, key='det_name', value=clutter detection
        self.clutter_detections = []
        for seq_idx in range(len(self.gt_objects)):
            seq_clutter_detections = []
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                frame_clutter_groups = []
                seq_clutter_detections.append(frame_clutter_groups)
            self.clutter_detections.append(seq_clutter_detections)


        #self.detection_groups[seq_idx][frame_idx], detection_groups for the specified sequence and frame
        #detection_groups, list where each element is a detection_group
        #detection_group, dictionary of detections in the group, key='det_name', value=detection
        self.detection_groups = []
        for seq_idx in range(len(self.gt_objects)):
            seq_detection_groups = []
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                frame_det_groups = []
                seq_detection_groups.append(frame_det_groups)
            self.detection_groups.append(seq_detection_groups)            

        # A list of sequence indices that will be used for training
        self.training_sequences = training_sequences 

        self.store_associations_in_gt()
        self.associate_all_clutter()

    def get_gt_ids_by_frame(self):
        """
        Output:
        all_gt_ids_by_frame: all_gt_ids_by_frame[i][j] is a list of all ground truth track ids that exist
            in frame j of sequence i
        all_assoc_gt_ids_by_frame: all_assoc_gt_ids_by_frame[i][j] is a list of ground truth track ids that exist
            in frame j of sequence i and are associated with a detection
        """
        all_gt_ids_by_frame = []
        all_assoc_gt_ids_by_frame = []
        for seq_idx in range(len(self.gt_objects)):
            all_gt_ids_by_frame.append([])
            all_assoc_gt_ids_by_frame.append([])
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                all_gt_ids_by_frame[seq_idx].append([])
                all_assoc_gt_ids_by_frame[seq_idx].append([])
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    all_gt_ids_by_frame[seq_idx][frame_idx].append(self.gt_objects[seq_idx][frame_idx][gt_idx].track_id)
                    if len(self.gt_objects[seq_idx][frame_idx][gt_idx].assoc_dets) > 0:
                        all_assoc_gt_ids_by_frame[seq_idx][frame_idx].append(self.gt_objects[seq_idx][frame_idx][gt_idx].track_id)

        assert(len(all_gt_ids_by_frame) == len(self.gt_objects))
        assert(len(all_assoc_gt_ids_by_frame) == len(self.gt_objects))
        for seq_idx in range(len(self.gt_objects)):
            assert(len(all_gt_ids_by_frame[seq_idx]) == len(self.gt_objects[seq_idx]))
            assert(len(all_assoc_gt_ids_by_frame[seq_idx]) == len(self.gt_objects[seq_idx]))
            for frame_idx in range(len(self.gt_objects[seq_idx]) - 1):
                assert(len(all_gt_ids_by_frame[seq_idx][frame_idx]) == len(self.gt_objects[seq_idx][frame_idx]))

        return (all_gt_ids_by_frame, all_assoc_gt_ids_by_frame)

    def group_detections(self, frame_detection_groups, det_name, detections):
        """
        Take a list of detections and try to associate them with detection groups from other measurement sources
        Inputs:
        - frame_detection_groups: a list of detection groups, where each detection group is a dictionary of detections 
            in the group, key='det_name', value=detection
        - det_name: name of the detection source we are currently associating with current detection groups
        - detections: a list of detections from a specific measurement source, sequence, and frame
        - seq_idx: the sequence index
        - frame_idx: the frame index (in the specified sequence)

        Outputs:
        None, but frame_detection_groups will be modified, with the new detections added (passed by reference)
        """

        hm = Munkres()
        max_cost = 1e9

        # use hungarian method to associate, using boxoverlap 0..1 as cost
        # build cost matrix
        cost_matrix = []
        this_ids = [[],[]]


        for cur_detection in detections:
            cost_row = []
            for cur_detection_group in frame_detection_groups:
                min_cost = max_cost
                for grpd_det_name, grouped_detection in cur_detection_group.iteritems():
                    # overlap == 1 is cost ==0
                    c = 1-boxoverlap(cur_detection, grouped_detection)
                    if c < min_cost:
                        min_cost = c
                # gating for boxoverlap
                if min_cost<=.5:
                    cost_row.append(min_cost)
                else:
                    cost_row.append(max_cost)
            cost_matrix.append(cost_row)
        
        if len(detections) is 0:
            cost_matrix=[[]]
        # associate
        association_matrix = hm.compute(cost_matrix)

        PRINT_AFTER = False
        if (len(frame_detection_groups)==4):
            print "group_detections called with det_name = ", det_name
            print "association_matrix:", association_matrix
            print "cost_matrix:", cost_matrix
            print "frame_detection_groups before new association: ", frame_detection_groups
            PRINT_AFTER = True

        associated_detection_indices = []
        check_det_count = 0
        for row,col in association_matrix:
            # apply gating on boxoverlap
            c = cost_matrix[row][col]
            print (row, col, c)

            if c < max_cost:
                associated_detection = detections[row]
                associated_detection_indices.append(row)
                associated_detection_group = frame_detection_groups[col]

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

                print "associated_detection_group before adding:", associated_detection_group
                associated_detection_group[det_name] = associated_detection                
                print "associated_detection_group after adding:", associated_detection_group


        for det_idx in range(len(detections)):
            if not(det_idx in associated_detection_indices):
                frame_detection_groups.append({det_name: detections[det_idx]})
                check_det_count += 1

        if PRINT_AFTER:
            print "frame_detection_groups aftern new association: ", frame_detection_groups


        assert(check_det_count == len(detections))


    def check_detection_groups(self):
        """
        Associate all detections to see what fraction of associations are correct
        """
        detection_count = 0
        correctly_assoc_det_count = 0
        incorrectly_assoc_det_count = 0

        detection_group_count = 0        
        perfect_group_count = 0
        imperfect_group_count = 0


        det_grp_assoc_wClutter = 0
        det_grp_assoc_wGTObject = 0

        num_gt_objs_assoc_mult_grps = 0

        det_group_size_counts = defaultdict(int)
        for det_name, det_objects in self.all_det_objects.iteritems():
            print "associating detections with name:", det_name
            for seq_idx in range(len(det_objects)):
                for frame_idx in range(len(det_objects[seq_idx])):
                    print "det_name before calling = ", det_name
                    self.group_detections(self.detection_groups[seq_idx][frame_idx], det_name, det_objects[seq_idx][frame_idx])

        for seq_idx in range(len(self.detection_groups)):
            for frame_idx in range(len(self.detection_groups[seq_idx])):
                for detection_group in self.detection_groups[seq_idx][frame_idx]:
                    this_frame_assoc_gt_ids = []
                    det_group_size_counts[len(detection_group)] += 1
                    detection_group_count +=1
                    det_grp_assoc = detection_group[detection_group.keys()[0]].assoc
                    correctly_assoc_group = True
                    for det_name, det in detection_group.iteritems():
                        if det.assoc != det_grp_assoc:
                            correctly_assoc_group = False
                    if correctly_assoc_group:
                        perfect_group_count += 1
                        if det_grp_assoc == -1:
                            det_grp_assoc_wClutter += 1
                        else:
                            det_grp_assoc_wGTObject += 1
                            if(det_grp_assoc in this_frame_assoc_gt_ids):
                                num_gt_objs_assoc_mult_grps += 1
                            else:
                                this_frame_assoc_gt_ids.append(det_grp_assoc)
                    else:
                        imperfect_group_count += 1

        print "fraction of perfectly associated groups =", float(perfect_group_count)/detection_group_count
        print "fraction of imperfectly associated groups =", float(imperfect_group_count)/detection_group_count
        print "perfect_group_count =", perfect_group_count
        print "imperfect_group_count =", imperfect_group_count
        print "detection_group_count =", detection_group_count
        print "# detection groups associated with clutter =", det_grp_assoc_wClutter
        print "# detection groups associated with a gt Object =", det_grp_assoc_wGTObject
        print "# of ground truth objects associated with multiple detection groups =", num_gt_objs_assoc_mult_grps
        print det_group_size_counts
#        sleep(5)
        #self.detection_groups[seq_idx][frame_idx], detection_groups for the specified sequence and frame
        #detection_groups, list where each element is a detection_group
        #detection_group, dictionary of detections in the group, key='det_name', value=detection


    def store_associations_in_gt(self):
        """
        Store a reference to associated detections in every associated ground truth object
        """
        for det_name, det_objects in self.all_det_objects.iteritems():
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "det_name:", det_name
            total_associated_det_count = 0
            assert(len(self.gt_objects) == len(det_objects))
            for seq_idx in range(len(self.gt_objects)):
                assert(len(self.gt_objects[seq_idx]) == len(det_objects[seq_idx]))
                for frame_idx in range(len(self.gt_objects[seq_idx])):
                    for cur_det in det_objects[seq_idx][frame_idx]:
                        if cur_det.assoc != -1:
                            match_found = False
                            #gt track_id this detection is associated with
                            cur_det_assoc = cur_det.assoc 
                            for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                                if self.gt_objects[seq_idx][frame_idx][gt_idx].track_id == cur_det_assoc:
                                    #we found the ground truth-detection match
                                    assert(match_found == False)
                                    match_found = True
                                    self.gt_objects[seq_idx][frame_idx][gt_idx].assoc_dets[det_name] = cur_det
                                    total_associated_det_count +=1
                            assert(match_found == True)
            print "total_associated_det_count =", total_associated_det_count

    def associate_clutter(self, clutter_det_name, clutter, seq_idx, frame_idx):
        """
        Take a list of clutter detections and try to associate them with clutter detections from other measurement
        sources in the speficied sequence and frame
        Inputs:
        - clutter: a list of clutter detections from a specific measurement source, sequence, and frame
        - seq_idx: the sequence index
        - frame_idx: the frame index (in the specified sequence)
        """
        clutter_groups = self.clutter_detections[seq_idx][frame_idx]

        hm = Munkres()
        max_cost = 1e9

        # use hungarian method to associate, using boxoverlap 0..1 as cost
        # build cost matrix
        cost_matrix = []
        this_ids = [[],[]]


        for cur_clutter in clutter:
            cost_row = []
            for cur_clutter_group in clutter_groups:
                min_cost = max_cost
                for det_name, grouped_clutter in cur_clutter_group.iteritems():
                    # overlap == 1 is cost ==0
                    c = 1-boxoverlap(cur_clutter, grouped_clutter)
                    if c < min_cost:
                        min_cost = c
                # gating for boxoverlap
                if min_cost<=.5:
                    cost_row.append(min_cost)
                else:
                    cost_row.append(max_cost)
            cost_matrix.append(cost_row)
        
        if len(clutter) is 0:
            cost_matrix=[[]]
        # associate
        association_matrix = hm.compute(cost_matrix)

        associated_clutter_indices = []
        check_clut_count = 0
        for row,col in association_matrix:
            # apply gating on boxoverlap
            c = cost_matrix[row][col]
            if c < max_cost:
                associated_clutter = clutter[row]
                associated_clutter_indices.append(row)
                associated_clutter_group = clutter_groups[col]

                #double check
                check_clut_count += 1
                min_cost = max_cost
                for det_name, grouped_clutter in associated_clutter_group.iteritems():
                    # overlap == 1 is cost ==0
                    check_c = 1-boxoverlap(associated_clutter, grouped_clutter)
                    if check_c < min_cost:
                        min_cost = check_c
                assert(min_cost == c), (min_cost, c)
                #done double check                

                associated_clutter_group[clutter_det_name] = associated_clutter                


        for clut_idx in range(len(clutter)):
            if not(clut_idx in associated_clutter_indices):
                clutter_groups.append({clutter_det_name: clutter[clut_idx]})
                check_clut_count += 1
        assert(check_clut_count == len(clutter))


    def associate_all_clutter(self):
        for det_name, det_objects in self.all_det_objects.iteritems():
            for seq_idx in range(len(det_objects)):
                for frame_idx in range(len(det_objects[seq_idx])):
                    frame_clutter = []
                    for cur_det in det_objects[seq_idx][frame_idx]:
                        if cur_det.assoc == -1:
                            frame_clutter.append(cur_det)
                    self.associate_clutter(det_name, frame_clutter, seq_idx, frame_idx)



    def get_clutter_priors(self):
        """
        Outputs:
        - clutter_grpCountByFrame_priors: dictionary where clutter_grpCountByFrame_priors[n] is the
        prior probability of observing n clutter objects in a frame, calculated as
        (#of frames in training data where n clutter objects were observed)/(#training frames)

        - clutter_group_priors: dictionary where clutter_group_priors[det_set] is the prior probability
        that a clutter object will emit the set of measurements specified by the immutable set det_set.
        Calculated as: (#clutter objects that emitted det_set in training data)/(#clutter objects in training data)
        """

        #self.clutter_detections[seq_idx][frame_idx], clutter_groups for the specified sequence and frame
        #clutter_groups, list where each element is a clutter_group
        #clutter_group, dictionary of clutter detections in the group, key='det_name', value=clutter detection
        
        #the number of underlying clutter objects, where an underlying clutter object creates a group of clutter
        #detections that can be associated from multiple measurement sources
        total_frame_count = 0
        total_clutter_group_count = 0
        #clutter_group_size_count[5] = 123 means that there are 123 underlying clutter objects that have clutter from 5 detection 
        #sources associated together in one group
        clutter_grpCountByFrame_count = defaultdict(int)
        clutter_by_group_count = defaultdict(int)
        for seq_idx in range(len(self.clutter_detections)):
            for frame_idx in range(len(self.clutter_detections[seq_idx])):
                total_frame_count += 1
                group_count = len(self.clutter_detections[seq_idx][frame_idx])
                clutter_grpCountByFrame_count[group_count] += 1
                for clutter_group in self.clutter_detections[seq_idx][frame_idx]:
                    total_clutter_group_count += 1
                    detection_sources_emitted = []
                    for det_name, clutter_det in clutter_group.iteritems():
                        detection_sources_emitted.append(det_name)
                    det_srcs_set = ImmutableSet(detection_sources_emitted)
                    clutter_by_group_count[det_srcs_set] += 1


        clutter_grpCountByFrame_priors = {}
        for group_count, count in clutter_grpCountByFrame_count.iteritems():
            clutter_grpCountByFrame_priors[group_count] = float(count)/total_frame_count

        clutter_group_priors = {}
        for det_set, count in clutter_by_group_count.iteritems():
            clutter_group_priors[det_set] = float(count)/total_clutter_group_count    

        total_prob = 0.0
        for group_count, prob in clutter_grpCountByFrame_priors.iteritems():
            total_prob += prob
        assert(abs(1.0-total_prob) < .0000001), (clutter_grpCountByFrame_count, total_frame_count, total_clutter_group_count, total_prob)

        total_prob = 0.0
        for det_set, prob in clutter_group_priors.iteritems():
            total_prob += prob
        assert(abs(1.0-total_prob) < .0000001), total_prob

        return (clutter_grpCountByFrame_priors, clutter_group_priors)


    def get_target_groupEmission_priors(self):
        """
        Outputs:
        - target_groupEmission_priors: dictionary where target_groupEmission_priors[det_set] is the prior probability
        that a ground truth object will emit the set of measurements specified by the immutable set det_set.
        Calculated as: (#gt objects that emitted det_set in training data)/(#gt objects in training data)
        """
        total_frame_count = 0
        total_gt_object_count = 0
        #clutter_group_size_count[5] = 123 means that there are 123 underlying clutter objects that have clutter from 5 detection 
        #sources associated together in one group
        target_groupEmission_count = defaultdict(int)
        for seq_idx in range(len(self.gt_objects)):
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                for gt_object in self.gt_objects[seq_idx][frame_idx]:
                    total_gt_object_count += 1
                    detection_sources_emitted = []
                    for det_name, det in gt_object.assoc_dets.iteritems():
                        detection_sources_emitted.append(det_name)
                    det_srcs_set = ImmutableSet(detection_sources_emitted)
                    target_groupEmission_count[det_srcs_set] += 1


        target_groupEmission_priors = {}
        for det_set, count in target_groupEmission_count.iteritems():
            target_groupEmission_priors[det_set] = float(count)/total_gt_object_count    

        total_prob = 0.0
        for det_set, prob in target_groupEmission_priors.iteritems():
            total_prob += prob
        assert(abs(1.0-total_prob) < .0000001)

        return target_groupEmission_priors


    def get_birth_priors(self):
        """
        Output:
        - birth_count_priors: dictionary, where birth_count_priors[n] is the prior probability of observing n births in a frame.
            Calculated as:
            (number of frames containing n birth gt objects)/(total number of frames)
            where a "birth gt object" is a gt object with id that did not appear in the previous frame
        """

        total_frame_count = 0
        #birth_count_count[5] = 18 means that 18 frames contain 5 birth measurements
        birth_count_count = defaultdict(int)
        for seq_idx, frames in enumerate(self.gt_objects):
            for frame_idx, gt_objects_in_frame in enumerate(frames):
                total_frame_count += 1
                if frame_idx == 0:
                    birth_count_count[len(gt_objects_in_frame)] += 1
                else:
                    birth_count_this_frame = 0
                    gt_ids_last_frame = []
                    for gt_obj_idx, gt_obj in enumerate(self.gt_objects[seq_idx][frame_idx-1]):
                        gt_ids_last_frame.append(gt_obj.track_id)
                    for gt_obj_idx, gt_obj in enumerate(gt_objects_in_frame):
                        if not (gt_obj.track_id in gt_ids_last_frame):
                            birth_count_this_frame+=1
                    birth_count_count[birth_count_this_frame] += 1

        birth_count_priors = {}
        for birth_count, count in birth_count_count.iteritems():
            birth_count_priors[birth_count] = float(count)/total_frame_count

        total_prob = 0.0
        for birth_count, prob in birth_count_priors.iteritems():
            total_prob += prob
        assert(abs(1.0-total_prob) < .0000001)

        return birth_count_priors



    def get_death_count(self, time_unassociated, near_border):
        """
        Input:
        - time_unassociated: the number of time instances unassociated before target death
        - near_border: boolean, does the ground truth obect have to be near the border time_unassociated time instances
            after the current time (one time instance before death) or is
            it required to not be near the border at this time?

        Output:
        - count: the total number of targets that die after being unassociated but alive for time_unassociated
            time instances
        """
        count = 0
        for seq_idx in self.training_sequences:
            for frame_idx in range(len(self.gt_objects[seq_idx]) - 1 - time_unassociated):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    cur_gt_id = self.gt_objects[seq_idx][frame_idx][gt_idx].track_id
                    alive_correctly = True
                    near_border_correctly = (self.gt_objects[seq_idx][frame_idx][gt_idx].near_border == near_border)

                    if len(self.gt_objects[seq_idx][frame_idx][gt_idx].assoc_dets) == 0:
                        initially_associated = False
                    else:
                        initially_associated = True
                    associated_correctly = initially_associated
                    for i in range(1, time_unassociated+1):
                        alive = False
                        associated = False
                        for j in range(len(self.gt_objects[seq_idx][frame_idx+i])):
                            if(cur_gt_id == self.gt_objects[seq_idx][frame_idx+i][j].track_id):
                                alive = True
                                if(len(self.gt_objects[seq_idx][frame_idx+i][j].assoc_dets) > 0):
                                    associated = True
                                if(i == time_unassociated):
                                    near_border_correctly = (self.gt_objects[seq_idx][frame_idx + time_unassociated][j].near_border == near_border)

                        if(not alive):
                            alive_correctly = False
                        if(associated):
                            associated_correctly = False

                    died_correctly = True
                    for j in range(len(self.gt_objects[seq_idx][frame_idx+1+time_unassociated])):
                        if(cur_gt_id == self.gt_objects[seq_idx][frame_idx+1+time_unassociated][j].track_id):
                            died_correctly = False #target still alive

                    if(alive_correctly and associated_correctly and died_correctly and near_border_correctly and initially_associated):
                        count += 1
        return count

    def get_death_count1(self, time_unassociated, near_border):
        """
        Input:
        - time_unassociated: the number of time instances unassociated before target death
        - near_border: boolean, does the ground truth obect have to be near the border time_unassociated time instances
            after the current time (one time instance before death) or is
            it required to not be near the border at this time?

        Output:
        - count: the total number of targets that die after being unassociated but alive for time_unassociated
            time instances
        """
        count = 0
        never_associated_gt_count = 0

        gt_track_ids = []
        dead_target_ids = []
        target_count_that_die_multiple_times = 0

        (all_gt_ids_by_frame, all_assoc_gt_ids_by_frame) = self.get_gt_ids_by_frame()

        total_death_count = 0
        total_never_dead_count = 0

        for seq_idx in self.training_sequences:
            for frame_idx in range(len(self.gt_objects[seq_idx])):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    if (not (seq_idx, self.gt_objects[seq_idx][frame_idx][gt_idx].track_id) in gt_track_ids):
                        gt_track_ids.append((seq_idx, self.gt_objects[seq_idx][frame_idx][gt_idx].track_id))
#            print "sequence ", seq_idx, " contains ", len(self.gt_objects[seq_idx][-1]),
#            print " objects alive in the last frame (index ",  len(self.gt_objects[seq_idx]) - 1, ") and ", len(self.gt_objects[seq_idx][-2]),
#            print " objects alive in the 2nd to last frame "
            #debug

            total_never_dead_count += len(self.gt_objects[seq_idx][-1])

            #end debug

            for frame_idx in range(len(self.gt_objects[seq_idx]) - 1):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    cur_gt_id = self.gt_objects[seq_idx][frame_idx][gt_idx].track_id
                    cur_gt_dies_next_step = not (cur_gt_id in all_gt_ids_by_frame[seq_idx][frame_idx + 1])
                    if cur_gt_dies_next_step:
                        if((seq_idx, cur_gt_id) in dead_target_ids):
                            target_count_that_die_multiple_times += 1
                        dead_target_ids.append((seq_idx, cur_gt_id))
                        total_death_count += 1
                        num_unassoc_steps = 0
                        alive = True
                        unassociated = True
                        while alive and unassociated and (frame_idx - num_unassoc_steps >= 0) and num_unassoc_steps <= time_unassociated:
                            alive = cur_gt_id in all_gt_ids_by_frame[seq_idx][frame_idx - num_unassoc_steps]
                            associated = cur_gt_id in all_assoc_gt_ids_by_frame[seq_idx][frame_idx - num_unassoc_steps]
                            if associated:
                                unassociated = False
                            else:
                                num_unassoc_steps += 1

                            if (not alive):
                                never_associated_gt_count += 1
                        if num_unassoc_steps == time_unassociated \
                            and self.gt_objects[seq_idx][frame_idx][gt_idx].near_border == near_border:
                            count += 1
#        print "never associated gt count = ", never_associated_gt_count
#        print "total death count = ", total_death_count
#        print "total number of targets that never die (alive in last frame of a sequence): ", total_never_dead_count
#        print "total number of targets = ", len(gt_track_ids)
#        print "number of targets that die more than once = ", target_count_that_die_multiple_times
        return count


    def get_living_count1(self, time_unassociated, near_border):
        """
        Input:
        - time_unassociated: the number of time instances unassociated before target death
        - near_border: boolean, does the ground truth obect have to be near the border time_unassociated time instances
            after the current time (one time instance before death) or is
            it required to not be near the border at this time?

        Output:
        - count: the total number of targets that are alive and unassociated the time instance after being unassociated for time_unassociated
            previous time instances (get_living_count(2) is the number of targets that are alive and unassociated after
            3 time instances from their last association)
        """
        count = 0
        total_gt_object_count = 0

        (all_gt_ids_by_frame, all_assoc_gt_ids_by_frame) = self.get_gt_ids_by_frame()


        for seq_idx in self.training_sequences:
            total_gt_object_count += len(self.gt_objects[seq_idx][-1])

            for frame_idx in range(len(self.gt_objects[seq_idx]) - 1):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):
                    total_gt_object_count += 1
                    cur_gt_id = self.gt_objects[seq_idx][frame_idx][gt_idx].track_id
                    cur_gt_assoc = (cur_gt_id in all_assoc_gt_ids_by_frame[seq_idx][frame_idx])
                    cur_gt_unassoc_but_living_next_step = not (cur_gt_id in all_assoc_gt_ids_by_frame[seq_idx][frame_idx + 1]) and \
                                                          (cur_gt_id in all_gt_ids_by_frame[seq_idx][frame_idx + 1])
                    if cur_gt_assoc and cur_gt_unassoc_but_living_next_step:
                        num_unassoc_steps = 0
                        unassociated = True
                        alive = True
                        while unassociated and alive and (frame_idx + 2 + num_unassoc_steps < len(self.gt_objects[seq_idx])) and num_unassoc_steps <= time_unassociated:
                            alive = cur_gt_id in all_gt_ids_by_frame[seq_idx][frame_idx + 2 + num_unassoc_steps]
                            associated = cur_gt_id in all_assoc_gt_ids_by_frame[seq_idx][frame_idx + 2 + num_unassoc_steps]
                            if associated:
                                unassociated = False
                            elif alive:
                                num_unassoc_steps += 1

                        if num_unassoc_steps >= time_unassociated \
                            and self.gt_objects[seq_idx][frame_idx][gt_idx].near_border == near_border:
                            count += 1
        print "total gt object count = ", total_gt_object_count
        return count


    def get_living_count(self, time_unassociated, near_border):
        """
        Input:
        - time_unassociated: the number of unassociated time instances
        - near_border: boolean, does the ground truth obect have to be near the border time_unassociated time instances
            after the current time (one time instance before the final time it must be alive and unassociated) or is
            it required to not be near the border at this time?

        Output:
        - count: the total number of targets that are alive and unassociated the time instance after being unassociated for time_unassociated
            previous time instances (get_living_count(2) is the number of targets that are alive and unassociated after
            3 time instances from their last association)
        """
        count = 0
        for seq_idx in self.training_sequences:
            for frame_idx in range(len(self.gt_objects[seq_idx]) - 1 - time_unassociated):
                for gt_idx in range(len(self.gt_objects[seq_idx][frame_idx])):

                    cur_gt_id = self.gt_objects[seq_idx][frame_idx][gt_idx].track_id
                    alive_correctly = True
                    near_border_correctly = (self.gt_objects[seq_idx][frame_idx][gt_idx].near_border == near_border)
                    if len(self.gt_objects[seq_idx][frame_idx][gt_idx].assoc_dets) == 0:
                        initially_associated = False
                    else:
                        initially_associated = True
                    associated_correctly = initially_associated
                    for i in range(1, time_unassociated + 2):
                        alive = False
                        associated = False
                        for j in range(len(self.gt_objects[seq_idx][frame_idx+i])):
                            if(cur_gt_id == self.gt_objects[seq_idx][frame_idx+i][j].track_id):
                                alive = True
                                if(len(self.gt_objects[seq_idx][frame_idx+i][j].assoc_dets) > 0):
                                    associated = True
                                if(i == time_unassociated):
                                    near_border_correctly = (self.gt_objects[seq_idx][frame_idx + time_unassociated][j].near_border == near_border)

                        if(not alive):
                            alive_correctly = False
                        if(associated):
                            associated_correctly = False
                    if(alive_correctly and associated_correctly and near_border_correctly and initially_associated):
                        count += 1
        return count        

    def get_death_probs(self, near_border):
        """
        Input:
        - near_border: boolean, death probabilities for ground truth obects near the border on
            their last time instance alive or not near the border?
        """
        death_probs = [-99]
        death_counts = []
        living_counts = []
        print '#'*80
        print "get_death_probs info: "
        for i in range(3):
            death_count = float(self.get_death_count(i, near_border))
            living_count = float(self.get_living_count(i, near_border))
            death_count1 = float(self.get_death_count1(i, near_border))
            living_count1 = float(self.get_living_count1(i, near_border))
            death_counts.append(death_count)
            living_counts.append(living_count)
            if death_count + living_count == 0:
                death_probs.append(1.0)
            else:
                death_probs.append(death_count/(death_count + living_count))

            print "time unassociated = %d:" % i, "death_count =", death_count, ", death_count1=", death_count1, ", living_count=", living_count, ", living_count1=", living_count1
        print '#'*80
        return (death_probs, death_counts, living_counts)




#Moved to returning dictionaries indexed by measurement type rather than lists
#need to change wherever this is being used
#WHEN WORKING delete get_meas_target_sets above
def get_meas_target_sets_general(training_sequences, score_intervals, detection_names, \
    obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, include_ignored_gt = False, \
    include_dontcare_in_gt = False, include_ignored_detections = True):
    """
    Input:
    - score_intervals: dictionary, where score_intervals['det_name'] contains score intervals for the
        detection type specified by the string 'det_name'.  E.g. score_intervals['mscnn'] contains score
        intervals for mscnn detections.
    - detection_names: list, containing names of all detection types to be used.  

    - doctor_clutter_probs: if True, add extend clutter probability list with 20 values of .0000001/20
        and subtract .0000001 from element 0

    Outputs:
    - posAndSize_inv_covariance_blocks: dictionary containing the inverse of the measurement noise covariance matrix, between
    all measurement sources

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
    posAndSize_inv_covariance_blocks[('meas_namei','meas_namej')] where 'meas_namei' is the string representation of the name of
    measurement source i.

    - meas_noise_mean: a dictionary where meas_noise_mean['meas_namei'] = the mean measurement noise for measurement
    source with name 'meas_namei' (position and size)

    - posOnly_covariance_blocks: Same format as posAndSize_inv_covariance_blocks, but posOnly_covariance_blocks[('meas_namei','meas_namej')]
    contains the covariance (NOT inverse) between the two sources and only the covariance of the position (not size)

    """

    #Should have a score interval for each detection type
    assert(len(detection_names) == len(score_intervals))

    #dictionaries for each measurement type, e.g meas_noise_covs['mscnn'] contains
    #meas_noise_covs for mscnn detections
    measurementTargetSetsBySequence = {}
    target_emission_probs = {}
    clutter_probabilities = {}
    meas_noise_covs = {}

    for det_name in detection_names:
        print "getting measurement target set for", det_name, "detections"
        (cur_measurementTargetSetsBySequence, cur_target_emission_probs, cur_clutter_probabilities, \
            junk_birth_probabilities, cur_meas_noise_covs) = get_meas_target_set(training_sequences, score_intervals[det_name], \
            det_name, obj_class, doctor_clutter_probs=doctor_clutter_probs, doctor_birth_probs=doctor_birth_probs, include_ignored_gt=include_ignored_gt, \
            include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)
        measurementTargetSetsBySequence[det_name] = cur_measurementTargetSetsBySequence
        target_emission_probs[det_name] = cur_target_emission_probs
        clutter_probabilities[det_name] = cur_clutter_probabilities
        meas_noise_covs[det_name] = cur_meas_noise_covs


    returnTargSets = []
    #double check measurementTargetSetsBySequence lengths
    seqCount = len(measurementTargetSetsBySequence[detection_names[0]])
    for det_name, det_measurementTargetSetsBySequence in measurementTargetSetsBySequence.iteritems():
        assert(len(det_measurementTargetSetsBySequence) == seqCount)

    for seq_idx in range(seqCount):
        curSeq_returnTargSets = []
        for det_name in detection_names:
            curSeq_returnTargSets.append(measurementTargetSetsBySequence[det_name][seq_idx])
        returnTargSets.append(curSeq_returnTargSets)

    print "Constructed returnTargSets"



    mail = mailpy.Mail("") #this is silly and could be cleaned up
    #dictionary where all_det_objects['det_name'] contains the detected objects of type 'det_name'
    all_det_objects = {}
    for det_name in detection_names:
        (gt_objects, cur_det_objects) = evaluate(min_score=score_intervals[det_name][0], \
            det_method=det_name, mail=mail, obj_class=obj_class, include_ignored_gt=include_ignored_gt,\
            include_dontcare_in_gt=include_dontcare_in_gt, include_ignored_detections=include_ignored_detections)        
        all_det_objects[det_name] = cur_det_objects

    print "Constructed all_det_objects"



    for gt_seq in gt_objects:
        for gt_frame in gt_seq:
            for gt_obj in gt_frame:
                assert(isinstance(gt_obj.assoc_dets, dict))

    all_detections = MultiDetections_many(gt_objects, all_det_objects, training_sequences)

    all_detections.check_detection_groups()
#    sleep(2)

    (clutter_grpCountByFrame_priors, clutter_group_priors) = all_detections.get_clutter_priors()
    target_groupEmission_priors = all_detections.get_target_groupEmission_priors()
    birth_count_priors = all_detections.get_birth_priors()

    (death_probs_near_border, death_counts_near_border, living_counts_near_border) = all_detections.get_death_probs(near_border = True)
    (death_probs_not_near_border, death_counts_not_near_border, living_counts_not_near_border) = all_detections.get_death_probs(near_border = False)

##############################################################################

    (posAndSize_inv_covariance_blocks, posOnly_covariance_blocks, meas_noise_mean_posAndSize) = calc_gaussian_paramaters('ground_truth',\
        all_detections.gt_objects, detection_names)

    (clutter_posAndSize_inv_covariance_blocks, clutter_posOnly_covariance_blocks, clutter_meas_noise_mean_posAndSize) = calc_gaussian_paramaters('clutter',\
        all_detections.gt_objects, detection_names, posAndSize_inv_covariance_blocks, meas_noise_mean_posAndSize, all_detections.clutter_detections)

    print "Ground truth detection covariance:"
    for det_name in detection_names:
        print det_name
        print posOnly_covariance_blocks[(det_name, det_name)]

    print "Clutter detection covariance:"
    for det_name in detection_names:
        print det_name
        print clutter_posOnly_covariance_blocks[(det_name, det_name)]

#    sleep(2.3)

    #FIX ME, move to returning dictionaries with detection name keys instead of lists
    #return (returnTargSets, target_emission_probs, clutter_probabilities, birth_probabilities, meas_noise_covs, death_probs_near_border, death_probs_not_near_border)

    #BIRTH AND CLUTTER PROBS ARE NOT DOCTORED, NEED TO DO LATER, e.g. replace any missing dictionary entry with epsilon when called


    return (returnTargSets, target_groupEmission_priors, clutter_grpCountByFrame_priors, clutter_group_priors, 
            birth_count_priors, death_probs_near_border, death_probs_not_near_border, 
            posAndSize_inv_covariance_blocks, meas_noise_mean_posAndSize, posOnly_covariance_blocks,
            clutter_posAndSize_inv_covariance_blocks, clutter_posOnly_covariance_blocks, clutter_meas_noise_mean_posAndSize)


def calc_gaussian_paramaters(group_type, gt_objects, detection_names, blocked_cov_inv=None, meas_noise_mean=None, clutter_detections=None):
    """
    Inputs:
    - group_type: string, either "ground_truth" or "clutter" calculate Gaussian parameters for detection groups associated
        with either valid ground truth objects or clutter

    - blocked_cov_inv: when calculating gaussian parameters of clutter, we need to come up with a position for the
        underlying clutter object.  We take the blocked_cov_inv calculated from ground truth objects and combine clutter
        measurements assuming the measurement sources have similar characteristics.
    - meas_noise_mean: when calculating gaussian parameters of clutter, we need to come up with a position for the
        underlying clutter object.  We take the meas_noise_mean calculated from ground truth objects and combine clutter
        measurements assuming the measurement sources have similar characteristics.
    - clutter_detections: should be MultiDetections_many.clutter_detections, only supply when group_type == 'clutter'
    Outputs:

    """
    assert(group_type in ['ground_truth', 'clutter'])
    if group_type == 'ground_truth':
        assert(blocked_cov_inv == None and meas_noise_mean == None)
    #dictionary where detection_ids['det_name'] is a list of obj_id's
    #that a detection of type 'det_name' is associated with
    detection_ids = defaultdict(list)
    obj_id = 0
    #detection_errors['det_name'][obj_id] is the error of detection type
    #specified by 'det_name' for the object with id obj_id
    detection_errors = {}
    for det_name in detection_names:
        detection_errors[det_name] = {}

    for seq_idx in range(21):
        for frame_idx in range(len(gt_objects[seq_idx])):
            if group_type == 'ground_truth':
                for gt_obj in gt_objects[seq_idx][frame_idx]:
                    obj_id += 1
                    num_det = len(gt_obj.assoc_dets)
                    for det_name, det in gt_obj.assoc_dets.iteritems():
                        cur_meas_loc_error = np.array([det.x - gt_obj.x, 
                                                       det.y - gt_obj.y,
                                                       det.width - gt_obj.width, 
                                                       det.height - gt_obj.height])                        
                        detection_errors[det_name][obj_id] = cur_meas_loc_error
                        detection_ids[det_name].append(obj_id)
            else: 
                for clutter_grp in clutter_detections[seq_idx][frame_idx]:
                    obj_id += 1
                    num_det = len(clutter_grp)
                    (clutter_loc, clutter_cov) = combine_arbitrary_number_measurements_4d(\
                        blocked_cov_inv, meas_noise_mean, clutter_grp)
                    for det_name, det in clutter_grp.iteritems():
                        cur_meas_loc_error = np.array([det.x - clutter_loc[0], 
                                                       det.y - clutter_loc[1],
                                                       det.width - clutter_loc[2], 
                                                       det.height - clutter_loc[3]])                        
                        detection_errors[det_name][obj_id] = cur_meas_loc_error
                        detection_ids[det_name].append(obj_id)


    def calc_cov_4DetAttributes(det_name1, det_name2, detection_errors, detection_ids):
        joint_meas_errors = []
        for _id in detection_ids[det_name1]:
            if (_id in detection_ids[det_name2]): #detections of both types are associated with this gt_object
                joint_meas_errors.append(np.concatenate((detection_errors[det_name1][_id], detection_errors[det_name2][_id])))
        cov = np.cov(np.asarray(joint_meas_errors).T)
        posAndSize_cov_block_12 = cov[0:4, 4:8]
        posAndSIze_cov_block_21 = cov[4:8, 0:4]

        posOnly_cov_block_12 = cov[0:2, 4:6]
        posOnly_cov_block_21 = cov[4:6, 0:2]

        if(det_name1 == det_name2):
            assert(np.all(abs(posAndSize_cov_block_12-posAndSIze_cov_block_21) < .0000001))
        else:
            assert(np.all(np.transpose(posAndSize_cov_block_12)==posAndSIze_cov_block_21))

        return(posAndSize_cov_block_12, posAndSIze_cov_block_21, posOnly_cov_block_12, posOnly_cov_block_21)

    #posAndSize_covariance_blocks{('det_name1', 'det_name2')} is the block of the complete covariance matrix
    #between detection sources 1 and 2
    posAndSize_covariance_blocks = {}
    posOnly_covariance_blocks = {}

    #meas_noise_mean: a dictionary where meas_noise_mean['meas_namei'] = the mean measurement noise for measurement
    #source with name 'meas_namei'        
    meas_noise_mean_posAndSize = {}
    for det_name1 in detection_names:
        meas_noise_mean_posAndSize[det_name1] = np.mean(np.asarray([v for v in detection_errors[det_name1].values()]).T,1)
        for det_name2 in detection_names:
            if not (det_name1, det_name2) in posAndSize_covariance_blocks:
                (posAndSize_cov_block_12, posAndSIze_cov_block_21, posOnly_cov_block_12, posOnly_cov_block_21) = calc_cov_4DetAttributes(det_name1, det_name2, detection_errors, detection_ids)
                posAndSize_covariance_blocks[(det_name1, det_name2)] = posAndSize_cov_block_12
                posAndSize_covariance_blocks[(det_name2, det_name1)] = posAndSIze_cov_block_21
                posOnly_covariance_blocks[(det_name1, det_name2)] = posOnly_cov_block_12
                posOnly_covariance_blocks[(det_name2, det_name1)] = posOnly_cov_block_21
#                if det_name2 != det_name1:
#                    posAndSize_covariance_blocks[(det_name1, det_name2)] = np.zeros((4,4))
#                    posAndSize_covariance_blocks[(det_name2, det_name1)] = np.zeros((4,4))


    #now assemble the full position and size covariance matrix
    posAndSize_full_cov = np.zeros((0,4*len(detection_names)))
    for det_name1 in detection_names:
        cur_block_of_rows = np.zeros((4,0))
        for det_name2 in detection_names:
            cur_block_of_rows = np.concatenate((cur_block_of_rows,posAndSize_covariance_blocks[(det_name1, det_name2)]),axis=1)
        posAndSize_full_cov = np.concatenate((posAndSize_full_cov,cur_block_of_rows),axis=0)

    posAndSize_full_cov_inv = inv(posAndSize_full_cov)
    #now create dictionary of blocks of the inverse of the covariance matrix
    posAndSize_inv_covariance_blocks = {}
    for (idx1, det_name1) in enumerate(detection_names):
        for (idx2, det_name2) in enumerate(detection_names):
            posAndSize_inv_covariance_blocks[(det_name1, det_name2)] = posAndSize_full_cov_inv[4*idx1:4*(idx1+1), 4*idx2:4*(idx2+1)]


    return (posAndSize_inv_covariance_blocks, posOnly_covariance_blocks, meas_noise_mean_posAndSize)

def combine_arbitrary_number_measurements_4d(blocked_cov_inv, meas_noise_mean, detection_group):
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

    -detection_group: dictionary with key='det_name', value=detection. The group of detections that will be combined, 
    can have an arbitrary number of detections

    """
    meas_count = len(detection_group) #number of associated measurements

#    #dictionary containing all measurements in appropriately formatted numpy arrays
#    reformatted_zs = {}
#    for det_name, det in detection_group.iteritems():
#        cur_z = np.array([det.x - meas_noise_mean[det_name][0], 
#                          det.y - meas_noise_mean[det_name][1],
#                          det.width - meas_noise_mean[det_name][2],
#                          det.height - meas_noise_mean[det_name][3]])
#        reformatted_zs[det_name] = cur_z
#
#    A = 0
#    b = 0
#    for det_name1, det in reformatted_zs.iteritems():
#        for det_name2, ignore_me_det in detection_group.iteritems():
#            A += blocked_cov_inv[(det_name1, det_name2)]
#            b += np.dot(det, blocked_cov_inv[(det_name1, det_name2)])
#
#    combined_meas_mean = np.dot(inv(A), b)
#    combined_covariance = inv(A)
#
#    assert(combined_meas_mean.shape == (4,)), (meas_count, detection_group)
#    return (combined_meas_mean, combined_covariance)

    #dictionary containing all measurements in appropriately formatted numpy arrays
    reformatted_zs = {}
    for det_name, det in detection_group.iteritems():
        cur_z = np.array([det.x - meas_noise_mean[det_name][0], 
                          det.y - meas_noise_mean[det_name][1],
                          det.width - meas_noise_mean[det_name][2],
                          det.height - meas_noise_mean[det_name][3]])
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



#########################################################################
# entry point of evaluation script
# input:
#   - det_method (method used for frame by frame object detection)
#   - user_sha (key of user who submitted the results, optional)
#   - user_sha (email of user who submitted the results, optional)
if __name__ == "__main__":

    sort_dets_on_intervals = True
    if sort_dets_on_intervals:
        score_interval_dict_all_det = {\
            'mscnn' : [float(i)*.1 for i in range(3,10)],              
            'regionlets' : [i for i in range(2, 20)],
            '3dop' : [float(i)*.1 for i in range(2,10)],            
            'mono3d' : [float(i)*.1 for i in range(2,10)],            
            'mv3d' : [float(i)*.1 for i in range(2,10)]}        
#            'regionlets' = [i for i in range(2, 16)]
    else:
        score_interval_dict_all_det = {\
#            'mscnn' = [.5],                                
        'mscnn' : [.3],                                
        'regionlets' : [2],
        '3dop' : [.2],
        'mono3d' : [.2],
        'mv3d' : [.2]}

    #train on all training sequences, except the current sequence we are testing on
    training_sequences = [i for i in range(21)]
    detection_names = ['mscnn', '3dop', 'mono3d', 'mv3d', 'regionlets']

    get_meas_target_sets_general(training_sequences, score_interval_dict_all_det, detection_names, \
    obj_class = "car", doctor_clutter_probs = True, doctor_birth_probs = True, include_ignored_gt = False, \
    include_dontcare_in_gt = False, include_ignored_detections = True)


    sleep(5)












    # check for correct number of arguments. if user_sha and email are not supplied,
    # no notification email is sent (this option is used for auto-updates)
    if len(sys.argv)!=2 or (sys.argv[1] != 'lsvm' and sys.argv[1] != 'regionlets'):
      print "Usage: python eval_tracking.py lsvm"
      print "--OR--"
      print "Usage: python eval_tracking.py regionlets"
      sys.exit(1);

    det_method = sys.argv[1]

    mail = mailpy.Mail("")

###########    score_intervals_lsvm = [i/2.0 for i in range(0, 10)]
###########    score_intervals_regionlets = [i for i in range(2, 20)]
############    score_intervals_lsvm = [0.0]
############    score_intervals_regionlets = [2.0]
#####  score_intervals = [2.0]
#####  get_meas_target_set(training_sequences, score_intervals, det_method = det_method, obj_class = "car", doctor_clutter_probs = True,\
#####                        print_info=True)
#    training_sequences = [i for i in range(21)] #use all sequences for training

    #### Check death probabilities #######
    (gt_objects, lsvm_det_objects) = evaluate(min_score=0.0, det_method='lsvm', mail=mail, obj_class="car")
    (gt_objects, regionlets_det_objects) = evaluate(min_score=2.0, det_method='regionlets', mail=mail, obj_class="car")
#    multi_detections = MultiDetections(gt_objects, regionlets_det_objects, lsvm_det_objects, training_sequences)
    multi_detections = MultiDetections(gt_objects, regionlets_det_objects, regionlets_det_objects, training_sequences)
#    multi_detections = MultiDetections(gt_objects, lsvm_det_objects, lsvm_det_objects, training_sequences)
    (death_probs_near_border, death_counts_near_border, living_counts_near_border) = multi_detections.get_death_probs(near_border = True)
    (death_probs_not_near_border, death_counts_not_near_border, living_counts_not_near_border) = multi_detections.get_death_probs(near_border = False)
    print "death probabilities near border:", death_probs_near_border
    print "death counts near border:", death_counts_near_border
    print "living counts near border:", living_counts_near_border
    print "death probabilities not near border:", death_probs_not_near_border
    print "death counts not near border:", death_counts_not_near_border
    print "living counts not near border:", living_counts_not_near_border

    (all_birth_probabilities_regionlets, all_birth_probabilities_lsvm) = apply_function_on_intervals_2_det(score_intervals_regionlets, \
        score_intervals_lsvm, multi_detections.get_birth_probabilities_score_range)

    print "regionlets birth probabilities: ", all_birth_probabilities_regionlets
    print "lsvm birth probabilities: ", all_birth_probabilities_lsvm


#    score_intervals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
#    score_intervals = [0.0, 5.0, 10.0, 15.0]
#    score_intervals = [i/2.0 for i in range(0, 10)]
#    score_intervals = [i for i in range(2, 20)]
    score_intervals = [2.0]

    #obj_class == "car" or obj_class == "pedestrian"
    (gt_objects, det_objects) = evaluate(score_intervals[0], det_method,mail, obj_class="car")
    all_data = AllData(gt_objects, det_objects, training_sequences)
################
################    print "clutter probabilities, not conditioned on measurement count:"
################    print get_clutter_probabilities(det_objects)
################
################    print len(det_objects)
################    print len(det_objects[0])
################    print len(det_objects[0][0])
################
################    print "clutter probabilities, conditioned on measurement count:"
################    (all_clutter_probabilities, frame_count) = all_data.get_clutter_probabilities_score_range_condition_num_meas(2.0, float("inf"))
################
################    print all_clutter_probabilities
################    print frame_count

##########    print '-'*80
##########    print "Testing detection score intervals"
##########
##########
##########    target_emission_probs = apply_function_on_intervals(score_intervals, all_data.get_prob_target_emission_by_score_range)
##########    clutter_probabilities = apply_function_on_intervals(score_intervals, all_data.get_clutter_probabilities_score_range)
##########    birth_probabilities = apply_function_on_intervals(score_intervals, all_data.get_birth_probabilities_score_range)
##########    num_measurements = apply_function_on_intervals(score_intervals, all_data.count_measurements)
##########    meas_noise_cov_and_mean = apply_function_on_intervals(score_intervals, all_data.get_R_score_range)
##########
##########    for i in range(len(score_intervals)):
##########        print '-'*10
##########        print "For detections with scores greater than ", score_intervals[i]
##########        print "Number of detections = ", num_measurements[i]
##########        print "Target emission probabilities: ", target_emission_probs[i]
##########        print "Clutter probabilities", clutter_probabilities[i]
##########        print "Birth probabilities", birth_probabilities[i]
##########        print "Measurement noise covariance matrix:"
##########        print meas_noise_cov_and_mean[i][0]
##########        print "Measurement noise mean:"
##########        print meas_noise_cov_and_mean[i][1]
##########

         
