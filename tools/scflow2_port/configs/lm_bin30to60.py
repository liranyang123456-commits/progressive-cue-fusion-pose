_base_ = './scflow2_lm.py'

data = dict(test=dict(ref_annots_root='data/initial_poses/ours_bin30-60/lm'))
work_dir = 'work_dirs/ours/lm_bin30to60'
