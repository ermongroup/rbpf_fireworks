import os

def directories_MOT_to_KITTI(dir_loc, multi_det):
    '''
    Take a MOT directory that contains 'test' and 'train' data
    folders and create a new 'kitti_format' directory that contains
    the same data in KITTI format

    Inputs:
    - dir_loc: string, full path of the MOT directory
    - multi_det: bool, True denotes multiple detection sets (MOT17 format)
                       False denotes one detection set (2DMOT2015, MOT16)

    Outputs:
    - None, but a new directory will be created
    '''
    kitti_dir = '%s/%s' % (dir_loc, 'kitti_format')
    all_det_dir = '%s/%s' % (kitti_dir, 'object_detections')
    gt_dir = '%s/%s' % (kitti_dir, 'training_ground_truth/label_02')

    if not os.path.exists(kitti_dir):
        os.makedirs(kitti_dir)
    if not os.path.exists(all_det_dir):
        os.makedirs(all_det_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)                

    for train_test in ['train', 'test']:
        #process training data
        #save the names of training sequences
        f_seq_names = open('%s/%s' % (kitti_dir, '%sing_seq_names.txt'%train_test), 'w')
        if train_test == 'train':
            f_seqmap = open('%s/%s' % (kitti_dir, 'evaluate_tracking.seqmap'), 'w')
        else:
            f_seqmap = open('%s/%s' % (kitti_dir, 'evaluate_tracking.seqmap.test'), 'w')

        train_seq_idx = 0
        #dictionary of sequence names we have already seen
        #key: seq_name, value: train_seq_str
        seq_names = {}
        for seq_det_name in os.listdir('%s/%s' % (dir_loc, train_test)):
            if seq_det_name[0] != '.': #avoid .DS and .DS_Store
                if multi_det:
                    det_name = seq_det_name.split("-")[-1]
                    seq_name = '-'.join(seq_det_name.split("-")[0:-1])
                else:
                    det_name = 'single_det_src'
                    seq_name = seq_det_name

                cur_det_dir = '%s/%s' % (all_det_dir, det_name)
                training_det_dir = '%s/%s/%s' % (cur_det_dir, 'training', 'det_02')
                test_det_dir = '%s/%s/%s' % (cur_det_dir, 'testing', 'det_02')
                if not os.path.exists(cur_det_dir):
                    os.makedirs(training_det_dir)                
                    os.makedirs(test_det_dir)

                if not seq_name in seq_names:
                    train_seq_str = "%04d" % train_seq_idx
                    train_seq_idx += 1
                    f_seq_names.write("%s %s\n" % (train_seq_str, seq_name))
                    seq_names[seq_name] = train_seq_str

                    img_dir = '%s/%s/%s/img1' % (dir_loc, train_test, seq_det_name)
                    frame_count = len([name for name in os.listdir(img_dir) if 
                                        os.path.isfile(os.path.join(img_dir, name)) and name[-4:] == '.jpg'])
                    f_seqmap.write("%s empty 000000 %d\n" % (train_seq_str, frame_count-1))                    
                    #debugging:
                    print seq_name
                    print seq_names
                    #end debugging
                else:
                    train_seq_str = seq_names[seq_name]

    #            shutil.copy('%s/%s/%s/gt/gt.txt' % (dir_loc, train_test, seq_det_name),
    #                        '%s/%s.txt' % (gt_dir, train_seq_str)
                file_MOT_to_KITTI(mot_file='%s/%s/%s/det/det.txt' % (dir_loc, train_test, seq_det_name),
                    kitti_file='%s/%sing/det_02/%s.txt' % (cur_det_dir, train_test, train_seq_str), file_type='detections')                

                if train_test == 'train':
                    file_MOT_to_KITTI(mot_file='%s/%s/%s/gt/gt.txt' % (dir_loc, train_test, seq_det_name),
                        kitti_file='%s/%s.txt' % (gt_dir, train_seq_str), file_type='ground_truth')
        
        f_seq_names.close()
        f_seqmap.close()


def file_MOT_to_KITTI(mot_file, kitti_file, file_type):
    '''
    Take a text file of object detections or ground truth tracks
    in MOT format and convert them to KITTI format

    Inputs:
    - mot_file: string, denoting the full path to a text file of 
        object detections or ground truth tracks in MOT format

    - kitti_file: string, denoting the full path to a text file
        that we will write containing object detections or ground
        truth tracks in KITTI format

    - file_type: string, either 'ground_truth' or 'detections'
        specifies the file contents

    Outpus:
    - None, but kitti_file will be created or overwritten

    '''
    assert(file_type in ['ground_truth', 'detections'])

    f_mot = open(mot_file, 'r')
    f_kitti = open(kitti_file, 'w')

    for line in f_mot:
        line = line.strip()
        fields = line.split(",")

        if file_type == 'detections':
            assert(len(fields) == 10 or len(fields) == 7), (len(fields), fields, mot_file)

        frame_idx = float(fields[0]) #frame idx
        frame_idx = frame_idx-1 #switch to 0 indexing for KITTI
        track_id = float(fields[1]) #track idx

        bb_left = float(fields[2])
        bb_top = float(fields[3])
        bb_width = float(fields[4])
        bb_height = float(fields[5])

        x1 = bb_left #left  
        y1 = bb_top #top   
        x2 = x1 + bb_width #right 
        y2 = y1 + bb_height #bottom

        if file_type == 'detections':
            det_score = float(fields[6])
            f_kitti.write( "%d %d Pedestrian -1 -1 -1 %f %f %f %f -1 -1 -1 -1 -1 -1 -1 %f\n" % \
                        (frame_idx, track_id, x1, y1, x2, y2, det_score))
        else:
            assert(file_type == 'ground_truth')
#            det_score = float(fields[6])
#            assert(det_score != 0)            
            f_kitti.write( "%d %d Pedestrian -1 -1 -1 %f %f %f %f -1 -1 -1 -1 -1 -1 -1\n" % \
                        (frame_idx, track_id, x1, y1, x2, y2))          

    f_mot.close()
    f_kitti.close()


if __name__ == "__main__":
    directories_MOT_to_KITTI(dir_loc = '/atlas/u/jkuck/MOT17', multi_det = True)
    directories_MOT_to_KITTI(dir_loc = '/atlas/u/jkuck/MOT16', multi_det = False)
    directories_MOT_to_KITTI(dir_loc = '/atlas/u/jkuck/2DMOT2015', multi_det = False)
    sleep(3)


    file_MOT_to_KITTI(mot_file = '/atlas/u/jkuck/MOT17/train/MOT17-02-DPM/det/det.txt',
                 kitti_file = '/atlas/u/jkuck/rbpf_fireworks/MOT_helpers/test_kitti_det.txt',
                 file_type = 'detections')

    file_MOT_to_KITTI(mot_file = '/atlas/u/jkuck/MOT17/train/MOT17-02-DPM/gt/gt.txt',
                 kitti_file = '/atlas/u/jkuck/rbpf_fireworks/MOT_helpers/test_kitti_gt.txt',
                 file_type = 'ground_truth')        