from haven import haven_utils as hu
import itertools, copy
EXP_GROUPS = {}

EXP_GROUPS['pascal_point_level'] = hu.cartesian_exp_group({
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': [
                                {'name':'pascal'} 
                                ],
                        'dataset_size':{'train':'all', 'val':'all'},
                        # 'dataset_size':{'train':10, 'val':10},
                        'max_epoch': [20],
                        'optimizer': [ "adam"], 
                        'lr': [ 1e-5,],
                        'model': {'name':'semseg', 'loss':'point_level',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':21}
                        })

EXP_GROUPS['pascal_cross_entropy'] = hu.cartesian_exp_group({
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': [
                               {'name':'pascal'} 
                                ],
                        'dataset_size':{'train':'all', 'val':'all'},
                        # 'dataset_size':{'train':10, 'val':10},
                        'max_epoch': [20],
                        'optimizer': [ "adam"], 
                        'lr': [ 1e-5,],
                        'model': {'name':'semseg', 'loss':'cross_entropy',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':21}
                        })

EXP_GROUPS['pascal_consistency_loss'] = hu.cartesian_exp_group({
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': [
                               {'name':'pascal'} 
                                ],
                        'dataset_size':{'train':'all', 'val':'all'},
                        #'dataset_size':{'train':10, 'val':10, 'test':10},
                        'max_epoch': [20],
                        'optimizer': [ "adam"], 
                        'lr': [ 1e-5,],
                        'model': {'name':'semseg', 'loss':'consistency_loss',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':21}
                        })

EXP_GROUPS["pascal"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
       {'name':'pascal'},
    ],
    'dataset_size': [
        #  {'train':10, 'val':10, 'test':10},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [100],
    'optimizer': ["adam"],
    'lr': [1e-5, ],
    'model': [
        {'name': 'semseg', 'loss': 'rot_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},
         
        {'name': 'semseg', 'loss': 'cons_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},

        {'name': 'semseg', 'loss': 'point_level',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},

        {'name': 'semseg', 'loss': 'cross_entropy',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},
    ]
})

EXP_GROUPS["cityscapes"] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
       {'name':'cityscapes'},
    ],
    'dataset_size': [
        #  {'train':10, 'val':10, 'test':10},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [100],
    'optimizer': ["adam"],
    'lr': [1e-5, ],
    'model': [
        {'name': 'semseg', 'loss': 'rot_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},
         
        {'name': 'semseg', 'loss': 'cons_point_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},

        {'name': 'semseg', 'loss': 'point_level',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},

        {'name': 'semseg', 'loss': 'cross_entropy',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 'n_classes': 21},
    ]
})




# ==========================
# the rest
def get_base_config(dataset_list, model_list, dataset_size={'train':'all', 'val':'all', 'test':'all'},
                             max_epoch=10):
        base_config =  {
                        'batch_size': 1,
                        'num_channels':1,
                        'dataset': [
                                {'name':name} for name in dataset_list
                                ],
                        'dataset_size':dataset_size,
                        'max_epoch': [max_epoch],
                        'optimizer': [ "adam"], 
                        'lr': [ 1e-5,],
                        'model': model_list
                        }
        return base_config

pascal_baseline = get_base_config(
                                dataset_list=['pascal'],
                                dataset_size=[{'train':'all', 'val':'all'},],
                                model_list=[{'name':'semseg', 'loss':'point_level',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':21},
                                            {'name':'semseg', 'loss':'cross_entropy',
                                            'base':'fcn8_vgg16',
                                            'n_channels':3, 'n_classes':21}])

EXP_GROUPS['pascal_baseline'] = hu.cartesian_exp_group(pascal_baseline)

pascal_baseline_debug = copy.deepcopy(pascal_baseline)
pascal_baseline_debug['dataset_size'] = [{'train':10, 'val':10},]
EXP_GROUPS['pascal_baseline_debug']  = hu.cartesian_exp_group(pascal_baseline_debug)


EXP_GROUPS["pascal_weakly"] = hu.cartesian_exp_group({
            'batch_size': [1],
            'num_channels': 1,
            'dataset': [
                {'name': 'pascal', 'n_classes': 2},

            ],
            'dataset_size': [
                 {'train':10, 'val':10, 'test':10},
                # {'train': 'all', 'val': 'all'},
            ],
            'max_epoch': [100],
            'optimizer': ["adam"],
            'lr': [1e-4, ],
            'model': [
                   
                
                {'name': 'semseg', 'loss': 'point_loss',
                 'base': 'fcn8_vgg16',
                 'n_channels': 3, 'n_classes': 1},

    ]
})



EXP_GROUPS["cp_weakly"] = hu.cartesian_exp_group({
            'batch_size': [1],
            'num_channels': 1,
            'dataset': [
                {'name': 'cityscapes', 'n_classes': 2},

            ],
            'dataset_size': [
                 {'train':10, 'val':10, 'test':10},
                # {'train': 'all', 'val': 'all'},
            ],
            'max_epoch': [100],
            'optimizer': ["adam"],
            'lr': [1e-4, ],
            'model': [
                   
                
                {'name': 'semseg', 'loss': 'point_loss',
                 'base': 'fcn8_vgg16',
                 'n_channels': 3, 'n_classes': 1},

    ]
})