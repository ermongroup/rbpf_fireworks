#!/usr/bin/env python
# encoding: utf-8
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FWAction, FireTaskBase

import sys,os,copy,math
from munkres import Munkres
from collections import defaultdict
try:
    from ordereddict import OrderedDict # can be installed using pip
except:
    from collections import OrderedDict # only included from python 2.7 on


import numpy as np
import glob

import mailpy


sys.path.insert(0, "../")
from cluster_config import RBPF_HOME_DIRECTORY
#DATA_PATH = "%sKITTI_helpers/data" % RBPF_HOME_DIRECTORY

#########################################################################
# function that does the evaluation
# input:
#   - result_sha (sha key where the results are located
#   - mail (messenger object for output messages sent via email and to cout)
# output:
#   - True if at least one of the sub-benchmarks could be processed successfully
#   - False otherwise
# data:
#   - at this point the submitted files are located in results/<result_sha>/data
#   - the results shall be saved as follows
#     -> summary statistics of the method: results/<result_sha>/stats_task.txt
#        here task refers to the sub-benchmark (e.g., um_lane, uu_road etc.)
#        file contents: numbers for main table, format: %.6f (single space separated)
#        note: only files with successful sub-benchmark evaluation must be created
#     -> detailed results/graphics/plots: results/<result_sha>/subdir
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

        #jdk extras
        self.imprt_weight = None #importance weight of the particle
        #boolean, true if this time instane is the first when this particle has the max importance weight
        self.frst_time_max_imprt_wt = None
        self.target_count = None #the number of targets that are believed to be alive at this time instance



    def __str__(self):
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())
#jdk
def write_to_annotated_results(object, type, filename):
    """
    Inputs:
    - object: type tData, either a ground truth target or detection
    - type: string, possibilities are
        - "tp" (true positive, object is a detection)
        - "fp" (false positive, object is a detection)
        - "fn" (false negative, object is a ground truth target)
        - "id" (id switch, object is a detection)
    - filename: the file to write to
    """
    with open(filename, "a") as myfile:
        #false negative with no associated detection
        if object.imprt_weight == None:
            assert(object.frst_time_max_imprt_wt==None and object.target_count==None)
            myfile.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1 %s\n" % \
                        (object.frame, object.track_id, object.x1, object.y1, object.x2, object.y2,
                         type))

        else:
            myfile.write( "%d %d Car -1 -1 2.57 %d %d %d %d -1 -1 -1 -1000 -1000 -1000 -10 1 %s %f %s %d\n" % \
                        (object.frame, object.track_id, object.x1, object.y1, object.x2, object.y2,
                         type, object.imprt_weight, object.frst_time_max_imprt_wt, object.target_count))
#end jdk
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

    def __init__(self, data_path, det_path, seq_idx_to_eval, corrected_version = True, gt_path = None, min_overlap=0.5, max_truncation = 0.15, cls="car"):
        # get number of sequences and
        # get number of frames per sequence from test mapping
        # (created while extracting the benchmark)
        filename_test_mapping =  data_path + "/evaluate_tracking.seqmap"
        self.n_frames         = {}
        self.sequence_name    = {}
        with open(filename_test_mapping, "r") as fh:
            for i,l in enumerate(fh):
                fields = l.split(" ")
                if int(fields[0]) in seq_idx_to_eval:
                    self.sequence_name[int(fields[0])] = ("%04d" % int(fields[0]))
                    self.n_frames[int(fields[0])] = (int(fields[3]) - int(fields[2])+1)
        fh.close()                                
        self.n_sequences = i+1

        # mail object
        self.mail = None

        # class to evaluate
        self.cls = cls

        # data and parameter
        self.corrected_version = corrected_version
        if gt_path == None: #using KITTI training data
            self.gt_path       = os.path.join(data_path + "/training_ground_truth", "label_02")
        else:
            self.gt_path       = gt_path

#        self.t_sha             = t_sha
        self.t_path            = det_path #file path of folder containing detection files (0000.txt, 0001.txt, etc.)
        self.seq_idx_to_eval   = seq_idx_to_eval
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

        #jdk, create a folder for storing annotated results when evaluating with the old script (correct FP count)
        if (not self.corrected_version) and (not os.path.exists(self.t_path + "annotated_results")):
            os.makedirs(self.t_path + "annotated_results")

#    def createEvalDir(self):
#        """Creates directory to store evaluation results and data for visualization"""
#        self.eval_dir = os.path.join("./results/", self.t_sha, "eval", self.cls)
#        if not os.path.exists(self.eval_dir):
#            print "create directory:", self.eval_dir,
#            os.makedirs(self.eval_dir)
#            print "done"

    def loadGroundtruth(self):
        """Helper function to load ground truth"""
        try:
            self._loadData(self.gt_path, cls=self.cls, loading_groundtruth=True)
        except IOError:
            return False
        return True

    def loadTracker(self):
        """Helper function to load tracker data"""
        try:
            if not self._loadData(self.t_path, cls=self.cls, loading_groundtruth=False):
                return False
        except IOError:
            return False
        return True

    def _loadData(self, root_dir, cls, min_score=-1000, loading_groundtruth=False):
        """
            Generic loader for ground truth and tracking data.
            Use loadGroundtruth() or loadTracker() to load this data.
            Loads detections in KITTI format from textfiles.
        """
        # construct objectDetections object to hold detection data


        print "HI", '!'*80
        print "loading_groundtruth=",loading_groundtruth

        t_data  = tData()
        data    = []
        eval_2d = True
        eval_3d = True

        seq_data           = {}
        n_trajectories     = 0
        n_trajectories_seq = []
        for seq, s_name in self.sequence_name.iteritems():
            i              = 0
            filename       = os.path.join(root_dir, "%s.txt" % s_name)
            f              = open(filename, "r") 
            f_data         = [[] for x in xrange(self.n_frames[seq])] # current set has only 1059 entries, sufficient length is checked anyway
            ids            = []
            n_in_seq       = 0
            id_frame_cache = []
            for line in f:
                # KITTI tracking benchmark data format:
                # (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
                line = line.strip()
                fields            = line.split(" ")
                print "len(fields) =", len(fields)
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
                t_data.track_id     = int(float(fields[1]))     # id
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
                    elif len(fields) == 21:
                        t_data.score  = float(fields[17])     # detection score                        
                        self.imprt_weight = float(fields[18]) #importance weight of the particle
                        #boolean, true if this time instane is the first when this particle has the max importance weight
                        self.frst_time_max_imprt_wt = fields[19].lower()
                        self.target_count = int(fields[20]) #the number of targets that are believed to be alive at this time instance

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
                        print "track ids are not unique for sequence %d: frame %d" % (seq,t_data.frame)
                        print "track id %d occured at least twice for this frame" % t_data.track_id
                        print "Exiting..."
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
            seq_data[seq] = f_data
            f.close()

        if not loading_groundtruth:
            self.tracker=seq_data
            self.n_tr_trajectories=n_trajectories
            self.eval_2d = eval_2d
            self.eval_3d = eval_3d
            self.n_tr_seq = n_trajectories_seq
            if self.n_tr_trajectories==0:
                print 'self.n_tr_trajectories==0'
                return False
        else: 
            # split ground truth and DontCare areas
            self.dcareas     = {}
            self.groundtruth = {}
            for seq_idx, cur_seq_data in seq_data.iteritems():
                seq_gt = cur_seq_data
                s_g, s_dc = [],[]
                for f in range(len(seq_gt)):
                    all_gt = seq_gt[f]
                    g,dc = [],[]
                    for gg in all_gt:
                        if gg.obj_type=="dontcare":
                            dc.append(gg)
                        else:
                            g.append(gg)
                    s_g.append(g)
                    s_dc.append(dc)
                self.dcareas[seq_idx] = s_dc
                self.groundtruth[seq_idx] = s_g
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

    def compute3rdPartyMetrics(self):
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

        # go through all frames and associate ground truth and tracker results
        # groundtruth and tracker contain lists for every single frame containing lists of KITTI format detections
        fr, ids = 0,0 
#        for seq_idx in range(len(self.groundtruth)):
#        for seq_idx in [6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20]:
        for seq_idx in self.seq_idx_to_eval:
            if not self.corrected_version:
                cur_annotated_file = self.t_path + "annotated_results/" + ("%04d.txt" % seq_idx)
                annoted_f = open(cur_annotated_file, 'w')
                annoted_f.close()

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

                jdk_tmptp = 0 #done
                jdk_tmpfp = 0 #done
                jdk_tmpfn = 0 #done
                jdk_ignored_tmptp = 0 #done              
                jdk_ignored_tmpfp = 0 #done
                jdk_ignored_tmpfn = 0 #done


                for row,col in association_matrix:
                    # apply gating on boxoverlap
                    c = cost_matrix[row][col]
                    if c < max_cost:
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
                    #jdk
                    if (not self.corrected_version):
                        if tt.ignored:
                            jdk_ignored_tmpfp += 1
                            write_to_annotated_results(tt, 'ignored_fp', cur_annotated_file)
                        elif (not tt.valid) and (not tt.ignored):
                            jdk_tmpfp += 1
                            write_to_annotated_results(tt, 'fp', cur_annotated_file)

                    #end jdk

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
                        #jdk    
                        if (not self.corrected_version):
                            if gg.ignored:
                                jdk_ignored_tmpfn += 1
                                write_to_annotated_results(gg, 'ignored_fn', cur_annotated_file) 
                            else:
                                jdk_tmpfn += 1
                                write_to_annotated_results(gg, 'fn', cur_annotated_file) 
                        #end jdk

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
                        #jdk    
                        if (not self.corrected_version):
                            if gg.ignored:
                                jdk_ignored_tmptp += 1
                            else:
                                jdk_tmptp += 1

                            if gg.id_switch:
                                write_to_annotated_results(gg, 'id_switch', cur_annotated_file) 
                            elif gg.ignored:
                                write_to_annotated_results(gg, 'ignored_tp', cur_annotated_file) 
                            else:
                                write_to_annotated_results(gg, 'tp', cur_annotated_file)                                 
                        #end jdk
                # correct TP by number of ignored TP due to truncation
                # ignored TP are shown as tracked in visualization
                tmptp -= nignoredtp
                self.n_gt -= (ignoredfn + nignoredtp)

                # false negatives = associated gt bboxes exceding association threshold + non-associated gt bboxes
                tmpfn   += len(g)-len(association_matrix)-ignoredfn
                self.fn += len(g)-len(association_matrix)-ignoredfn

                # false positives = tracker bboxes - associated tracker bboxes
                if self.corrected_version:
                    # mismatches (mme_t)
                    tmpfp   += len(t) - tmptp - nignoredtp
                    self.fp += len(t) - tmptp - nignoredtp
                else:
                    # mismatches (mme_t) 
                    tmpfp   += len(t) - tmptp - nignoredtracker - nignoredtp
                    self.fp += len(t) - tmptp - nignoredtracker - nignoredtp


                #jdk
                if (not self.corrected_version):
                    assert(tmpfn == jdk_tmpfn), (tmpfn, jdk_tmpfn)
                    assert(tmptp == jdk_tmptp), (tmptp, jdk_tmptp)
                    assert(tmpfp == jdk_tmpfp), (tmpfp, jdk_tmpfp)
                    assert(jdk_ignored_tmptp == nignoredtp), (jdk_ignored_tmptp, nignoredtp)
                    assert(jdk_ignored_tmpfp == nignoredtracker), (jdk_ignored_tmpfp, nignoredtracker)
                    assert(jdk_ignored_tmpfn == ignoredfn), (jdk_ignored_tmpfn, ignoredfn)
                    assert(len(g) == jdk_ignored_tmpfn + jdk_ignored_tmptp + jdk_tmptp + jdk_tmpfn)
                    assert(len(t) == jdk_tmptp + jdk_tmpfp + jdk_ignored_tmptp + jdk_ignored_tmpfp)
                #end jdk

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
                if self.corrected_version:
                    if tmptp+tmpfp+nignoredtp is not len(t):
                        print seq_idx, f, len(t), tmptp, tmpfp
                        print len(association_matrix), association_matrix
                        raise NameError("Something went wrong! nTracker is not TP+FP")

                else:
                    if tmptp+tmpfp+nignoredtracker+nignoredtp is not len(t):
                        print seq_idx, f, len(t), tmptp, tmpfp
                        print len(association_matrix), association_matrix
                        raise NameError("Something went wrong! nTracker is not TP+FP")


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
                if self.corrected_version:
                    tracking_ratio = tracked / float(len(g) - sum(ign_g))
                else:
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
        if sum(self.n_frames.values())==0:
            self.FAR = "n/a"
        else:
            self.FAR = self.fp/float(sum(self.n_frames.values()))

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
        if sum(self.n_frames.values())==0:
            self.MODP = "n/a"
        else:
            self.MODP = sum(self.MODP_t)/float(sum(self.n_frames.values()))
        return True

    def print_results(self):
#       print "tracking evaluation summary".center(80,"=")
#       print "Multiple Object Tracking Accuracy (MOTA)", self.MOTA
#       print "Multiple Object Tracking Precision (MOTP)", self.MOTP
#       print "Multiple Object Tracking Accuracy (MOTAL)", self.MOTAL
#       print "Multiple Object Detection Accuracy (MODA)", self.MODA
#       print "Multiple Object Detection Precision (MODP)", self.MODP
#       print ""
#       print "Recall", self.recall
#       print "Precision", self.precision
#       print "F1", self.F1
#       print "False Alarm Rate", self.FAR
#       print ""
#       print "Mostly Tracked", self.MT
#       print "Partly Tracked", self.PT
#       print "Mostly Lost", self.ML
#       print ""
#       print "True Positives", self.tp
#       print "False Positives", self.fp
#       print "Missed Targets", self.fn
#       print "ID-switches", self.id_switches
#       print "Fragmentations", self.fragments
#       print ""
#       print "Ground Truth Objects", self.n_gt
#       print "Ground Truth Trajectories", self.n_gt_trajectories
#       print "Tracker Objects", self.n_tr
#       print "Tracker Trajectories", self.n_tr_trajectories
#       print "="*80


        print "tracking evaluation summary".center(80,"=")
        print self.printEntry("Multiple Object Tracking Accuracy (MOTA)", self.MOTA)
        print self.printEntry("Multiple Object Tracking Precision (MOTP)", self.MOTP)
        print self.printEntry("Multiple Object Tracking Accuracy (MOTAL)", self.MOTAL)
        print self.printEntry("Multiple Object Detection Accuracy (MODA)", self.MODA)
        print self.printEntry("Multiple Object Detection Precision (MODP)", self.MODP)
        print ""
        print self.printEntry("Recall", self.recall)
        print self.printEntry("Precision", self.precision)
        print self.printEntry("F1", self.F1)
        print self.printEntry("False Alarm Rate", self.FAR)
        print ""
        print self.printEntry("Mostly Tracked", self.MT)
        print self.printEntry("Partly Tracked", self.PT)
        print self.printEntry("Mostly Lost", self.ML)
        print ""
        print self.printEntry("True Positives", self.tp)
        print self.printEntry("False Positives", self.fp)
        print self.printEntry("Missed Targets", self.fn)
        print self.printEntry("ID-switches", self.id_switches)
        print self.printEntry("Fragmentations", self.fragments)
        print ""
        print self.printEntry("Ground Truth Objects", self.n_gt)
        print self.printEntry("Ground Truth Trajectories", self.n_gt_trajectories)
        print self.printEntry("Tracker Objects", self.n_tr)
        print self.printEntry("Tracker Trajectories", self.n_tr_trajectories)
        print "="*80

    def get_results_as_array(self):
        results_array = np.array([self.MOTA, self.MOTP, self.MOTAL, self.MODA, self.MODP, \
            self.recall, self.precision, self.F1, self.FAR, self.MT, self.PT, self.ML, \
            self.tp, self.fp, self.fn, self.id_switches, self.fragments, self.n_gt,\
            self.n_gt_trajectories, self.n_tr, self.n_tr_trajectories])
        return results_array

######class evalMetrics:
######    def __init__(self, MOTA, MOTP, MOTAL, MODA, MODP, recall, precision, F1, FAR, MT, PT, ML, tp, fp, fn, id_switches, fragments, n_gt, n_gt_trajectories, n_tr, n_tr_trajectories):
######        self.MOTA = MOTA
######        self.MOTP = MOTP
######        self.MOTAL = MOTAL
######        self.MODA = MODA
######        self.MODP = MODP
######        self.recall = recall
######        self.precision = precision
######        self.F1 = F1
######        self.FAR = FAR
######        self.MT = MT
######        self.PT = PT
######        self.ML = ML
######        self.tp = tp
######        self.fp = fp
######        self.fn = fn
######        self.id_switches = id_switches
######        self.fragments = fragments
######        self.n_gt = n_gt
######        self.n_gt_trajectories = n_gt_trajectories
######        self.n_tr = n_tr
######        self.n_tr_trajectories = n_tr_trajectories
######
######        #(self, self.MOTA, self.MOTP, self.MOTAL, self.MODA, self.MODP, self.recall, self.precision, self.F1, self.FAR, self.MT, self.PT, self.ML, self.tp, self.fp, self.fn, self.id_switches, self.fragments, self.n_gt, self.n_gt_trajectories, self.n_tr, self.n_tr_trajectories)
######

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



#    def saveToStats(self):
#        self.summary()
#        filename = os.path.join("./results", self.t_sha, "stats_%s.txt" % self.cls)
#        dump = open(filename, "w+")
#        print>>dump, "%.6f " * 21 \
#                % (self.MOTA, self.MOTP, self.MOTAL, self.MODA, self.MODP, \
#                   self.recall, self.precision, self.F1, self.FAR, \
#                   self.MT, self.PT, self.ML, self.tp, self.fp, self.fn, self.id_switches, self.fragments, \
#                   self.n_gt, self.n_gt_trajectories, self.n_tr, self.n_tr_trajectories)
#        dump.close()
#        filename = os.path.join("./results", self.t_sha, "description.txt")
#        dump = open(filename, "w+")
#        print>>dump, "MOTA", "MOTP", "MOTAL", "MODA", "MODP", "recall", "precision", "F1", "FAR",
#        print>>dump, "MT", "PT", "ML", "tp", "fp", "fn", "id_switches", "fragments",
#        print>>dump, "n_gt", "n_gt_trajectories", "n_tr", "n_tr_trajectories"

    def sequenceSummary(self):
        filename = os.path.join("./results", self.t_sha, self.dataset, "sequences.txt")
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

def evaluate(data_path, det_path, seq_idx_to_eval, gt_path = None, corrected_version = True, class_to_eval = "car"):
    # start evaluation and instanciated eval object
#    mail.msg("Processing Result for KITTI Tracking Benchmark")
    print "Evaluating class ", class_to_eval

    e = trackingEvaluation(data_path, det_path, seq_idx_to_eval,corrected_version=corrected_version, gt_path=gt_path, cls=class_to_eval)
    # load tracker data and check provided classes
    if not e.loadTracker():
        print "loadTracker failed for class: ", class_to_eval
        print "det_path: ", det_path
        return False
        assert(False)
    # load groundtruth data for this class
    if not e.loadGroundtruth():
        print "Ground truth not found."
        return False
        assert(False)
        
    # sanity checks
    if len(e.groundtruth) is not len(e.tracker):
        return False
    if not e.compute3rdPartyMetrics():
        print("Error evaluating results")
        print "Did not evaluate 3party metrics!"
        return False
        assert(False)

    print("Thank you for participating in our benchmark!")
    return e.get_results_as_array()

def printEntry(key, vals,width=(43,20)):
    s_out =  key.ljust(width[0])

    for val in vals:
        if type(val)==int:
            s = "%%%dd" % width[1]
            s_out += s % val
        elif type(val)==float:
            s = "%%%df" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s"%val).rjust(width[1])
    return s_out

def print_multi_run_metrics(packed_metrics, number_of_runs):
        print ("tracking evaluation summary over %d runs"%number_of_runs).center(80,"=")
        print printEntry("Metric", ["Median", "Mean", "Minimum", "Maximum", "Standard Deviation"])
        print printEntry("Multiple Object Tracking Accuracy (MOTA)", packed_metrics[0])
        print printEntry("Multiple Object Tracking Precision (MOTP)", packed_metrics[1])
        print printEntry("Multiple Object Tracking Accuracy (MOTAL)", packed_metrics[2])
        print printEntry("Multiple Object Detection Accuracy (MODA)", packed_metrics[3])
        print printEntry("Multiple Object Detection Precision (MODP)", packed_metrics[4])
        print ""
        print printEntry("Recall", packed_metrics[5])
        print printEntry("Precision", packed_metrics[6])
        print printEntry("F1", packed_metrics[7])
        print printEntry("False Alarm Rate", packed_metrics[8])
        print ""
        print printEntry("Mostly Tracked", packed_metrics[9])
        print printEntry("Partly Tracked", packed_metrics[10])
        print printEntry("Mostly Lost", packed_metrics[11])
        print ""
        print printEntry("True Positives", packed_metrics[12])
        print printEntry("False Positives", packed_metrics[13])
        print printEntry("Missed Targets", packed_metrics[14])
        print printEntry("ID-switches", packed_metrics[15])
        print printEntry("Fragmentations", packed_metrics[16])
        print ""
        print printEntry("Ground Truth Objects", packed_metrics[17])
        print printEntry("Ground Truth Trajectories", packed_metrics[18])
        print printEntry("Tracker Objects", packed_metrics[19])
        print printEntry("Tracker Trajectories", packed_metrics[20])
        if(len(packed_metrics) == 23):
            print ""
            print "RBPF performance information:"   
            print printEntry("Number of time resampling performed", packed_metrics[21])
            print printEntry("Single run run-time", packed_metrics[22])
     
        print "="*80

def eval_results(data_path, all_run_results, seq_idx_to_eval, SPEC=None, gt_path=None, use_corrected_eval=True, info_by_run=None):
    """
    Inputs:
    - seq_idx_to_eval: a list of sequence indices to evaluate
    - all_run_results: filepath of folder containing folders of results from individual runs (subfolders are where 
        .txt files are located containing results for each sequence to evaluate)
    - info_by_run: a list of length equal to the number of runs, where each element is a list containing
        info from that run (all should be the same length and have the same type of info) if available.
        If not available, this will be None
    - gt_path: string, if true we are evaulating synthetic data that we generated, this is where the ground truth
        is stored.  If false we are evaluating KITTI data, this is None.

    Output:
    - number_of_runs: the number of runs evaluated over
    - metric_medians: a dictionary whose keys are metric names and values are medians 
        of the corresponding metric over the runs
    """
    print "debugging jdk_helpers_evaluate_results.py eval_results:"
    print "all_run_results:"
    print all_run_results
    print "seq_idx_to_eval:"
    print seq_idx_to_eval
    print "info_by_run:"
    print info_by_run

    all_runs_metrics = None

    number_of_runs = 0
    print "all_run_results:", all_run_results
    for cur_run_results in glob.iglob(all_run_results + "/*"): # + operator used for string concatenation!
        if os.path.isdir(cur_run_results):
            all_sequences_completed = True
            for cur_seq_idx in seq_idx_to_eval:
                cur_run_complete_filename = '%s/seq_%d_done.txt' % (cur_run_results, cur_seq_idx)
                if (not os.path.isfile(cur_run_complete_filename)):
                    all_sequences_completed = False
            if all_sequences_completed:
                if SPEC['train_test'] == 'generated_data':
                    cur_run_metrics = evaluate(data_path, cur_run_results + "/", seq_idx_to_eval, gt_path=gt_path, corrected_version=use_corrected_eval, class_to_eval=SPEC['obj_class']) # + operator used for string concatenation!
                else:
                    cur_run_metrics = evaluate(data_path, cur_run_results + "/", seq_idx_to_eval, corrected_version=use_corrected_eval, class_to_eval=SPEC['obj_class']) # + operator used for string concatenation!
                print cur_run_metrics
                print 'hi'
                orig_metrics_len = len(cur_run_metrics)
                #append run info, if given, to evaluation metrics
                if info_by_run:
                    cur_run_metrics.resize(len(cur_run_metrics) + len(info_by_run[0]))
                    for info_idx in range(len(info_by_run[0])):
                        assert(len(info_by_run[0]) == len(info_by_run[number_of_runs]))
                        assert(number_of_runs < len(info_by_run)), number_of_runs
                        cur_run_metrics[orig_metrics_len + info_idx] = info_by_run[number_of_runs][info_idx]
                cur_run_metrics = np.expand_dims(cur_run_metrics, axis=0)
                if all_runs_metrics == None:
                    all_runs_metrics = cur_run_metrics
                else:
                    all_runs_metrics = np.concatenate((all_runs_metrics, cur_run_metrics), axis=0)
                number_of_runs+=1
            else:
                print "Error: all runs not complete"

    if number_of_runs != 0:
        metric_medians = np.median(all_runs_metrics, axis=0)
        metric_means = np.mean(all_runs_metrics, axis=0)
        metric_mins = np.amin(all_runs_metrics, axis=0)
        metric_maxs = np.amax(all_runs_metrics, axis=0)
        metric_std_devs = np.std(all_runs_metrics, axis=0)

        packed_metrics = []
        for metric_idx in range(len(metric_means)):
            cur_metric_stats = [metric_medians[metric_idx], metric_means[metric_idx], metric_mins[metric_idx],\
                                metric_maxs[metric_idx], metric_std_devs[metric_idx]]
            packed_metrics.append(cur_metric_stats)

        print_multi_run_metrics(packed_metrics, number_of_runs)

        metric_medians = {}
        metric_medians['MOTA'] = packed_metrics[0][0]     
        metric_medians['MOTP'] = packed_metrics[1][0]     
        metric_medians['Mostly Tracked'] = packed_metrics[9][0]     
        metric_medians['Mostly Lost'] = packed_metrics[11][0] 
        metric_medians['True Positives'] = packed_metrics[12][0] 
        metric_medians['False Positives'] = packed_metrics[13][0] 
        metric_medians['Missed Targets'] = packed_metrics[14][0] 
        metric_medians['ID-switches'] = packed_metrics[15][0] 
        metric_medians['Fragmentations'] = packed_metrics[16][0] 
        metric_medians['Ground Truth Objects'] = packed_metrics[17][0] 
        metric_medians['Ground Truth Trajectories'] = packed_metrics[18][0] 
        metric_medians['Tracker Objects'] = packed_metrics[19][0] 
        metric_medians['Tracker Trajectories'] = packed_metrics[20][0] 

        print "done evaluating results!"
    return (number_of_runs, metric_medians)

def find_and_eval_results(data_path, directory_to_search, seq_idx_to_eval=[i for i in range(21)], info_by_run=None):
    """
    Inputs:
    - seq_idx_to_eval: a list of sequence indices to evaluate
    - directory_to_search: string, directory name to begin search from (e.g. '/Users/jkuck/rotation3/pykalman')
    """

    for filename in os.listdir(directory_to_search):
        if filename == 'results_by_run':
            print "about to eval: ", directory_to_search

            if (not os.path.isfile(directory_to_search + '/OLD_evaluation_metrics.txt')):
                stdout = sys.stdout
                sys.stdout = open(directory_to_search + '/OLD_evaluation_metrics.txt', 'w')

                eval_results(data_path, directory_to_search + "/results_by_run", seq_idx_to_eval, use_corrected_eval=False) # + operateor used for string concatenation!

                sys.stdout.close()
                sys.stdout = stdout

            if (not os.path.isfile(directory_to_search + '/NEW_evaluation_metrics.txt')):

                stdout = sys.stdout
                sys.stdout = open(directory_to_search + '/NEW_evaluation_metrics.txt', 'w')

                eval_results(data_path, directory_to_search + "/results_by_run", seq_idx_to_eval, use_corrected_eval=True) # + operateor used for string concatenation!

                sys.stdout.close()
                sys.stdout = stdout

            print "done with eval: ", directory_to_search

        elif os.path.isdir(directory_to_search + '/' + filename):
            find_and_eval_results(data_path, directory_to_search + '/' + filename, seq_idx_to_eval, info_by_run)

@explicit_serialize
class RunEval(FireTaskBase):   
 #   _fw_name = "Run RBPF Task"
    def run_task(self, fw_spec):
        print 'fw_spec, atlas:', fw_spec
        results_folder = fw_spec['results_folder']
        use_corrected_eval = fw_spec['use_corrected_eval']
        seq_idx_to_eval = fw_spec['seq_idx_to_eval']
        data_path = fw_spec['data_path']

        stdout = sys.stdout
        if use_corrected_eval:
            sys.stdout = open(results_folder + '/NEW_evaluation_metrics.txt', 'w')
        else:
            sys.stdout = open(results_folder + '/OLD_evaluation_metrics.txt', 'w')
        (number_of_runs, metric_medians) = eval_results(data_path, results_folder + "/results_by_run", SPEC=fw_spec, seq_idx_to_eval=seq_idx_to_eval, 
            gt_path=fw_spec['gt_path'], use_corrected_eval=use_corrected_eval) # + operateor used for string concatenation!
        sys.stdout.close()
        sys.stdout = stdout

        if 'mod_direction' in fw_spec and fw_spec['mod_direction'] != 'orig':
            mod_dir = fw_spec['mod_direction']
            if use_corrected_eval:
                return FWAction(stored_data=metric_medians, mod_spec=[{'_set': {"NEW_eval_metrics_with_%s"%mod_dir: metric_medians}}])
            else:
                return FWAction(stored_data=metric_medians, mod_spec=[{'_set': {"OLD_eval_metrics_with_%s"%mod_dir: metric_medians}}])
        else:

####### Switch to this if running compare_proposals_synth_data.py for easier results viewing in database #########
#            if use_corrected_eval:
#                return FWAction(stored_data=metric_medians, mod_spec=[{'_set': {"NEW_eval_metrics%s%s"%(fw_spec['proposal_distr'], fw_spec['gumbel_scale']) : metric_medians}}])
#            else:
#                return FWAction(stored_data=metric_medians, mod_spec=[{'_set': {"OLD_eval_metrics%s%d"%(fw_spec['proposal_distr'], int(10*fw_spec['gumbel_scale'])) : metric_medians}}])

#            if use_corrected_eval:
#                return FWAction(stored_data=metric_medians, mod_spec=[{'_set': {"NEW_eval_metrics%s%s"%(fw_spec['proposal_distr'], fw_spec['K_NEAREST_TARGETS']) : metric_medians}}])
#            else:
#                return FWAction(stored_data=metric_medians, mod_spec=[{'_set': {"OLD_eval_metrics%s%s"%(fw_spec['proposal_distr'], fw_spec['K_NEAREST_TARGETS']) : metric_medians}}])

            if use_corrected_eval:
                return FWAction(stored_data=metric_medians, mod_spec=[{'_set': {"NEW_eval_metrics" : metric_medians,
                                                                                "number_resamplings": fw_spec['number_resamplings']}}])
            else:
                return FWAction(stored_data=metric_medians, mod_spec=[{'_set': {"OLD_eval_metrics" : metric_medians,
                                                                                "number_resamplings": fw_spec['number_resamplings']}}])


@explicit_serialize
class CoordAscentEval(FireTaskBase):   
 #   _fw_name = "Run RBPF Task"
    def run_task(self, fw_spec):
        results_folder = fw_spec['results_folder']
        use_corrected_eval = fw_spec['use_corrected_eval']
        seq_idx_to_eval = fw_spec['seq_idx_to_eval']
        #did we increase, decrease, or test the original value of a parameter? options:
        #'larger', 'smaller', or 'original'
        delta_name = fw_spec['delta_name']
        #which object are we trying to optimize? options:
        # 'MOTA', 'MOTP', 'Mostly Tracked', or 'Mostly Lost'
        # Although be careful with Mostly Lost, since we want to minimize!
        objective = fw_spec['objective']     
#        run_spec = fw_spec['run_spec'] #rbpf was run with this spec on this past iteration
#        past_MOTA = fw_spec['past_MOTA'] #MOTA from the last iteration
        data_path = fw_spec['data_path']


        stdout = sys.stdout
        if use_corrected_eval:
            sys.stdout = open(results_folder + '/NEW_evaluation_metrics.txt', 'w')
        else:
            sys.stdout = open(results_folder + '/OLD_evaluation_metrics.txt', 'w')
        (number_of_runs, metric_medians) = eval_results(data_path, results_folder + "/results_by_run", seq_idx_to_eval=seq_idx_to_eval, 
            gt_path=fw_spec['gt_path'], use_corrected_eval=use_corrected_eval) # + operateor used for string concatenation!
        sys.stdout.close()
        sys.stdout = stdout

        return FWAction(stored_data=metric_medians, mod_spec=[{'_set': {delta_name: metric_medians[objective]}}])

if __name__ == "__main__":
    if len(sys.argv)!=3:
        print "Supply the directory to evaluate and bool whether to use the corrected evaluation metric"
        sys.exit(1);

    results_dir = sys.argv[1]
    if not sys.argv[2] in ['True', 'False']:
        print "Error, use_corrected_eval must be 'True' or 'False'"

    use_corrected_eval = (sys.argv[2] == 'True')
    data_path = "%sKITTI_helpers/data" % RBPF_HOME_DIRECTORY

    if use_corrected_eval:
        stdout = sys.stdout
        sys.stdout = open(results_dir + '/NEW_evaluation_metrics.txt', 'w')
        eval_results(data_path, results_dir + "/results_by_run", seq_idx_to_eval=[i for i in range(21)], use_corrected_eval=use_corrected_eval) # + operateor used for string concatenation!
        sys.stdout.close()
        sys.stdout = stdout

    else:
        stdout = sys.stdout
        sys.stdout = open(results_dir + '/OLD_evaluation_metrics.txt', 'w')
        eval_results(data_path, results_dir + "/results_by_run", seq_idx_to_eval=[i for i in range(21)], use_corrected_eval=use_corrected_eval) # + operateor used for string concatenation!
        sys.stdout.close()
        sys.stdout = stdout




