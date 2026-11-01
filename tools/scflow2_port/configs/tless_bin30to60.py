_base_ = './scflow2_tless.py'

data = dict(test=dict(ref_annots_root='data/initial_poses/ours_bin30-60/tless'))
work_dir = 'work_dirs/ours/tless_bin30to60'
