# Rao-Blackwellized Particle Filtering for Multi-Sensor Multi-Target Tracking

Code for testing and comparison of multi-sensor, multi-target tracking methods. The workflow mananger Fireworks (https://materialsproject.github.io/fireworks/) is used to manage running experiments on a cluster.
Our inputs/outputs and evaluation follow the KITTI benchmark (http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

## Fireworks installation

 - make cluster_config.py file
 - make my_qadapter.yaml file (look at fireworks workflow manager website for info)
 
 If the database thinks a firework is still running, but no jobs are running on the cluster, try: 
 
 $ lpad detect_lostruns --time 1 --refresh


## Running on Atlas
start a krbscreen session:

 $ krbscreen #reattach using $ screen -rx\
 $ reauth #important so that jobs can be submitted after logging out, enter password\
 $ export PATH=/opt/rh/python27/root/usr/bin:$PATH \
 $ export LD_LIBRARY_PATH=/opt/rh/python27/root/usr/lib64/:$LD_LIBRARY_PATH \
 $ PACKAGE_DIR=/atlas/u/jkuck/software \
 $ export PATH=$PACKAGE_DIR/anaconda2/bin:$PATH \
 $ export LD_LIBRARY_PATH=$PACKAGE_DIR/anaconda2/local:$LD_LIBRARY_PATH \
 $ source activate anaconda_venv \
 $ cd /atlas/u/jkuck/rbpf_fireworks/ \
 $ python compare_proposals_kitti_data.py
 
 To install anaconda packages run, e.g.:
 $ conda install -c matsci fireworks=1.3.9

May need to run $ kinit -r 30d

 Add the following line to the file ~/.bashrc.user on Atlas:
 
 export PYTHONPATH="/atlas/u/jkuck/rbpf_fireworks:$PYTHONPATH"
 
 
 Weird, but to run commands like "lpad -l my_launchpad.yaml get_fws",
 add the following line to the file ~/.bashrc.user on Atlas:
 
 export PYTHONPATH="${PYTHONPATH}:/atlas/u/jkuck/rbpf_fireworks/KITTI_helpers/"


 To install cvxpy on atlas run:

$ export PATH=/opt/rh/python27/root/usr/bin:$PATH \
$ export LD_LIBRARY_PATH=/opt/rh/python27/root/usr/lib64/:$LD_LIBRARY_PATH \
$ pip install --user numpy \
$ pip install --user cvxpy 

 Install pymatgen:
$ pip install --user pymatgen

## Running on Sherlock
Note, on Sherlock before this script:
$ ml load python/2.7.5\
$ easy_install-2.7 --user pip\
$ export PATH=~/.local/bin:$PATH\
$ pip2.7 install --user fireworks #and others\
$ pip2.7 install --user filterpy\
$ pip2.7 install --user scipy --upgrade\
$ pip2.7 install --user munkres\
$ pip2.7 install --user pymatgen\
$ cd /scratch/users/kuck/rbpf_fireworks/ \
$ python compare_proposals_kitti_data.py


 Add the following line to the file ~/.bashrc on Sherlock:\
 export PYTHONPATH="/scratch/users/kuck/rbpf_fireworks:$PYTHONPATH"
 
 Weird, but to run commands like "lpad -l my_launchpad.yaml get_fws",
 add the following line to the file ~/.bashrc.user on Atlas:\
 export PYTHONPATH="${PYTHONPATH}:/scratch/users/kuck/rbpf_fireworks/KITTI_helpers/"


