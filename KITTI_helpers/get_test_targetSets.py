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
USE_PICKLED_DATA = False
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

    def __init__(self, data_path, cutoff_score, det_method, gt_path=DATA_PATH + "/training_ground_truth", min_overlap=0.5, max_truncation = 0.15, mail=None, cls="car"):
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
        filename_test_mapping = data_path + "/evaluate_tracking.seqmap.test"
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
        self.t_path            = os.path.join(data_path + "/object_detections", self.det_method, "testing/det_02")
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
            print 'filename:', filename
            f              = open(filename, "r") 

            f_data         = [[] for x in xrange(self.n_frames[seq])] # current set has only 1059 entries, sufficient length is checked anyway
            print "self.n_frames[seq]", self.n_frames[seq]
            print "self.n_frames", self.n_frames

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

        print 'loading_groundtruth:', loading_groundtruth
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

    def get_det_objs(self):
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


        #det_objects[i][j] is a list of all detected objects in the jth frame of the ith video sequence
        det_objects = []

        # go through all frames and associate ground truth and tracker results
        # groundtruth and tracker contain lists for every single frame containing lists of KITTI format detections
        fr, ids = 0,0 
        for seq_idx in range(len(self.tracker)):
            x_min = 9999999
            x_max = -9999999
            y_min = 9999999
            y_max = -9999999

            det_objects.append([]) #jdk

            seq_tracker      = self.tracker[seq_idx]

            for f in range(len(seq_tracker)):
                det_objects[-1].append([]) #jdk
                t = seq_tracker[f]
                for det_obj_idx in range(len(t)):
                    assert(t[det_obj_idx].obj_type == self.cls)
                    det_objects[-1][-1].append(detObject(t[det_obj_idx].x1, t[det_obj_idx].x2, t[det_obj_idx].y1, t[det_obj_idx].y2, assoc=-1, score=t[det_obj_idx].score))

        return (det_objects)

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

def get_det_objs1(data_path, min_score, det_method,mail,obj_class = "car"):
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

#    if USE_PICKLED_DATA:
#        if not os.path.exists(PICKELD_DATA_DIRECTORY):
#            os.makedirs(PICKELD_DATA_DIRECTORY)
#
#        data_filename = PICKELD_DATA_DIRECTORY + "/min_score_%f_det_method_%s_obj_class_%s_include_ignored_gt_%s_include_dontcare_gt_%s_include_ignored_det_%s.pickle" % \
#                                                 (min_score, det_method, obj_class, include_ignored_gt, include_dontcare_in_gt, include_ignored_detections)
#        lock_filename = data_filename + "_lock"
#
#        if os.path.isfile(data_filename) and (not os.path.isfile(lock_filename)): 
#            f = open(data_filename, 'r')
#            (gt_objects, det_objects) = pickle.load(f)
#            f.close()
#            return (gt_objects, det_objects)
#

    mail.msg("Processing Result for KITTI Tracking Benchmark")
    classes = []
    assert(obj_class == "car" or obj_class == "pedestrian")
    e = trackingEvaluation(data_path=data_path, cutoff_score=min_score, det_method=det_method, mail=mail,cls=obj_class)
    # load tracker data and check provided classes
    e.loadDetections()
    mail.msg("Evaluate Object Class: %s" % obj_class.upper())
    classes.append(obj_class)
    # load groundtruth data for this class
#    if not e.loadGroundtruth(include_dontcare_in_gt):
#        raise ValueError("Ground truth not found.")
#    mail.msg("Loading Groundtruth - Success")
    # sanity checks
#    if len(e.groundtruth) is not len(e.tracker):
#        mail.msg("The uploaded data does not provide results for every sequence.")
#        return False
#    mail.msg("Loaded %d Sequences." % len(e.groundtruth))
#    mail.msg("Start Evaluation...")
    # create needed directories, evaluate and save stats
#    try:
#        e.createEvalDir()
#    except:
#        mail.msg("Feel free to contact us (lenz@kit.edu), if you receive this error message:")
#        mail.msg("   Caught exception while creating results.")
#        if e.compute3rdPartyMetrics():
#            e.saveToStats()
#        else:
#            mail.msg("There seem to be no true positives or false positives at all in the submitted data.")

    (det_objects) = e.get_det_objs()

#    if USE_PICKLED_DATA and (not os.path.isfile(lock_filename)):
#        f_lock = open(lock_filename, 'w')
#        f_lock.write("locked\n")
#        f_lock.close()
#
#        f = open(data_filename, 'w')
#        pickle.dump((gt_objects, det_objects), f)
#        f.close()  
#
#        os.remove(lock_filename)
#
#    # finish
#    if len(classes)==0:
#        mail.msg("The uploaded results could not be evaluated. Check for format errors.")
#        return False
#    mail.msg("Thank you for participating in our benchmark!")
    return (det_objects)



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


def get_meas_target_set(data_path, score_intervals, det_method="lsvm", obj_class="car"):
    """
    Input:
    - doctor_clutter_probs: if True, replace 0 probabilities with .0000001/float(20+num_zero_probs) and extend
        clutter probability list with 20 values of .0000001/20 and subtract .0000001 from element 0
    - doctor_birth_probs: if True then if any birth probability is 0 subtract .0000001 from element 0
        of its score interval's birth probability list and replacing zero elements with .0000001/(number of
        zero elements in the score interval's birth probability list)
    """
#    if USE_PICKLED_DATA:
#        if not os.path.exists(PICKELD_DATA_DIRECTORY):
#            os.makedirs(PICKELD_DATA_DIRECTORY)
#
#        data_filename = PICKELD_DATA_DIRECTORY + "/meas_targ_set_scores_%s_det_method_%s_obj_class_%s_include_ignored_gt_%s_include_dontcare_gt_%s_include_ignored_det_%s.pickle" % \
#                                                 (str(score_intervals), det_method, obj_class, include_ignored_gt, include_dontcare_in_gt, include_ignored_detections)
#        lock_filename = data_filename + "_lock"
#
#        if os.path.isfile(data_filename) and (not os.path.isfile(lock_filename)): 
#            f = open(data_filename, 'r')
#            (measurementTargetSetsBySequence, target_emission_probs, clutter_probabilities, birth_probabilities, meas_noise_covs) = pickle.load(f)
#            f.close()
#            return (measurementTargetSetsBySequence, target_emission_probs, clutter_probabilities, birth_probabilities, meas_noise_covs)



    mail = mailpy.Mail("")

    print score_intervals

    (det_objects) = get_det_objs1(data_path, score_intervals[0], det_method,mail, obj_class=obj_class)

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


    return (measurementTargetSetsBySequence)


def get_meas_target_sets_test(data_path, score_intervals, detection_names, \
    obj_class = "car"):


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
        (cur_measurementTargetSetsBySequence) = get_meas_target_set(data_path, score_intervals[det_name], \
            det_name, obj_class=obj_class)
        measurementTargetSetsBySequence[det_name] = cur_measurementTargetSetsBySequence


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

    return returnTargSets

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

    returnTargSets = get_meas_target_sets_test(DATA_PATH, score_interval_dict_all_det, detection_names, \
    obj_class = "car")

    print returnTargSets
    print len(returnTargSets)
    for ts in returnTargSets:
        print len(ts)
        for seqSt in ts:
            print len(seqSt)


         
