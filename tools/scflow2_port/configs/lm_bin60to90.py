_base_ = './scflow2_lm.py'

data = dict(test=dict(ref_annots_root='data/initial_poses/ours_bin60-90/lm'))
work_dir = 'work_dirs/ours/lm_bin60to90'
