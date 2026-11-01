dataset_root = 'data/tless'

init_pose_method = 'foundpose'  # foundpose genflow genflow-mh megapose foundpose-fmh foundpose+featref foundpose+fm megapose-best foundationpose sam6d
init_mask_method = 'cnos'     # cnos sam fastsam
ref_mask_dir = f'data/reference_masks/{init_mask_method}_per_method/{init_pose_method}/lmo'
ref_annots_root = 'data/initial_poses/ours_bin15-30/tless'

filter_depth=False

train_point_cloud_sample_num = 1024
test_point_cloud_sample_num = 1024

CLASS_NAMES = ('obj_01', 'obj_02', 'obj_03', 'obj_04', 'obj_05', 'obj_06', 'obj_07', 'obj_08', 'obj_09', 'obj_10', 'obj_11', 'obj_12', 'obj_13', 'obj_14', 'obj_15', 'obj_16', 'obj_17', 'obj_18', 'obj_19', 'obj_20', 'obj_21', 'obj_22', 'obj_23', 'obj_24', 'obj_25', 'obj_26', 'obj_27', 'obj_28', 'obj_29', 'obj_30')
LABEL_MAPPING = {i: i for i in range(1, 31)}
OCCLINEMOD_LABELS = [1, 5, 9, 14, 20, 23, 27, 30]

obj13_classnames = ('ape', 'benchvise', 'cam', 'can', 
                'cat', 'driller', 'duck', 'eggbox', 
                'glue', 'holepuncher', 'iron','lamp', 'phone')
image_scale = 256
normalize_mean = [0., 0., 0., ]
normalize_std = [255., 255., 255.]

symmetry_types = {}

mesh_diameter = [63.515, 66.151, 65.349, 80.726, 108.69, 108.265, 178.615, 217.156, 144.546, 90.211, 76.598, 86.011, 58.126, 71.947, 68.569, 69.188, 112.839, 110.982, 89.069, 98.889, 92.253, 92.253, 142.587, 84.736, 108.801, 108.801, 152.495, 124.778, 134.227, 88.754]

file_client_args = dict(backend='disk')




train_real_pipeline = [
    dict(type='LoadImages', color_type='unchanged', file_client_args=file_client_args),
    dict(type='LoadMasks', file_client_args=file_client_args),
    dict(type='PoseJitter',
        jitter_angle_dis=(0, 15),
        jitter_x_dis=(0, 15),
        jitter_y_dis=(0, 15),
        jitter_z_dis=(0, 50),
        angle_limit=45, 
        translation_limit=200,
        add_limit=1.,
        mesh_dir=dataset_root + '/models_eval',
        mesh_diameter=mesh_diameter,
        jitter_pose_field=['gt_rotations', 'gt_translations'],
        jittered_pose_field=['ref_rotations', 'ref_translations']),
    dict(type='ComputeBbox', mesh_dir=dataset_root + '/models_eval', clip_border=False),
    dict(type='Crop',
        size_range=(1.0, 1.25), 
        crop_bbox_field='ref_bboxes',
        clip_border=False,
        pad_val=128,
    ),
    dict(type='RandomHSV', h_ratio=0.2, s_ratio=0.5, v_ratio=0.5),
    dict(type='RandomNoise', noise_ratio=0.1),
    dict(type='RandomSmooth', max_kernel_size=5.),
    dict(type='Resize', img_scale=image_scale, keep_ratio=True),
    dict(type='Pad', size=(image_scale, image_scale), center=True, pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type='RemapPose', keep_intrinsic=False),
    dict(type='GetPointCloud', depth_sample_num=train_point_cloud_sample_num),
    dict(type='Normalize', mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type='ToTensor', stack_keys=[], ),
    dict(type='Collect', 
        annot_keys=[
            'ref_rotations', 'ref_translations', 'gt_masks',
            'init_add_error', 'init_rot_error', 'init_trans_error',
            'k', 'labels',
            'depths', 'model_list', 'cloud_list',
            ],
        meta_keys=(
            'img_path', 'ori_shape', 'ori_k',
            'img_shape', 'img_norm_cfg', 
            'scale_factor', 'transform_matrix',),
    ),
]


test_pipeline = [
    dict(type='LoadImages', color_type='unchanged', file_client_args=file_client_args),
    dict(type='LoadMasks'),
    dict(type='ComputeBbox', mesh_dir=dataset_root + '/models_eval', clip_border=False, filter_invalid=False),
    dict(type='Crop', 
        size_range=(1.1, 1.1),
        crop_bbox_field='gt_bboxes',   # ref_mask_bbox ref_bboxes gt_bboxes
        clip_border=False,
        pad_val=128),
    dict(type='Resize', img_scale=image_scale, keep_ratio=True),
    dict(type='Pad', size=(image_scale, image_scale), center=True, pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type='RemapPose', keep_intrinsic=False),
    dict(type='GetPointCloud', filter_depth=filter_depth, filter_point_cloud=False,             # filter_depth=True
        minimum_points=32, depth_sample_num=test_point_cloud_sample_num,
        filter_rgb=False
        ),
    dict(type='Normalize', mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type='ToTensor', stack_keys=[], ),
    dict(type='Collect', 
        annot_keys=[
            'ref_rotations', 'ref_translations',
            'labels','k','ori_k','transform_matrix',
            'depths','model_list', 'cloud_list',
            'gt_masks'  # add in 250329 for draw mask
        ],
        meta_keys=(
            'img_path', 'ori_shape', 'img_shape', 'img_norm_cfg', 
            'scale_factor', 'keypoints_3d','geometry_transform_mode',),
    ),
]


data = dict(
    samples_per_gpu=6,
    workers_per_gpu=0,
    test_samples_per_gpu=1,
    test=dict(
        type='RefineDataset',
        # type='RefineTestDataset',
        data_root='data/tless/test',
        ref_annots_root=ref_annots_root,
        image_list='data/tless/image_lists/ours.txt',     # test_tmp test_bop19
        keypoints_json=dataset_root + '/keypoints/bbox.json',
        pipeline=test_pipeline,
        class_names=CLASS_NAMES,
        label_mapping=LABEL_MAPPING,
        target_label=OCCLINEMOD_LABELS,
        load_depth=True,            #-# add depth
        load_mask=True,
        load_point_clouds=True,     #-# add point cloud
        filter_invalid_pose=False,  # --------------
        depth_range=(200, 10000),
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_eval',
        mesh_diameter=mesh_diameter,
        mesh_sample_num=test_point_cloud_sample_num,
    ),
)


# renderer setting
model = dict(
    renderer=dict(
        mesh_dir=dataset_root + '/models',
        image_size=(image_scale, image_scale),
        shader_type='Phong',
        soft_blending=False,
        render_mask=False,
        render_image=True,
        seperate_lights=True,
        faces_per_pixel=1,
        bin_size=-1,
        blur_radius=0.,
        sigma=1e-12,
        gamma=1e-12,
        background_color=(.5, .5, .5),
    ),
)