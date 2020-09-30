from haven import haven_utils as hu
import itertools, copy
EXP_GROUPS = {}


EXP_GROUPS['fish_budget'] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        # {'name': 'JcuFish', 'n_classes': 2, 'n_fish_images':15},
        {'name': 'JcuFish', 'n_classes': 2, 'n_fish_images':10},
        #  {'name': 'JcuFish', 'n_classes': 2, 'n_fish_images':8},
        #   {'name': 'JcuFish', 'n_classes': 2, 'n_fish_images':6},
    ],
    'dataset_size': [
        #  {'train':100, 'val':'all', 'test':'all'},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [1000],
    'optimizer': ["adam"],
    'lr': [1e-5],
    'model':
     [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         } for l in ['cross_entropy', ]]
    ,
})


EXP_GROUPS['fish_hybrid'] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        {'name': 'JcuFish', 'n_classes': 2, 'n_full_images':5},

    ],
    'dataset_size': [
        #  {'train':10, 'val':10, 'test':10},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [1000],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model':
     [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         } for l in ['pseudo_mask', ]] +

     [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'with_affinity':True,
        #  'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]] +
         [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
        #  'with_affinity':True,
        #  'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]] 
         
         +
         [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'with_affinity':True,
         } for l in ['point_level', ]] +
         [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         } for l in ['point_level', ]] 
    ,
})

EXP_GROUPS['fish_weak_supervision'] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        {'name': 'JcuFish', 'n_classes': 2},

    ],
    'dataset_size': [
        #  {'train':10, 'val':10, 'test':10},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [1000],
    'optimizer': ["adam"],
    'lr': [1e-4, 1e-5, 1e-6],
    'model':
     [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         } for l in ['pseudo_mask', ]] +

    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         } for l in ['cross_entropy', ]] +

   [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'with_affinity':True,
         } for l in ['cross_entropy', ]] +
     [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'with_affinity':True,
        #  'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]] +
         [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
        #  'with_affinity':True,
        #  'with_affinity_average':True,
         } for l in ['lcfcn_loss', ]] 
         
         +
         [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'with_affinity':True,
         } for l in ['point_level', ]] +
         [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         } for l in ['point_level', ]] 
    ,
})


EXP_GROUPS['fish_shared'] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        {'name': 'JcuFish', 'n_classes': 2},

    ],
    'dataset_size': [
        #  {'train':10, 'val':10, 'test':10},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [1000],
    'optimizer': ["adam"],
    'lr': [ 1e-6],
    'model':
  
    
    
     [
        {'name': 'semseg',
         'loss': 'lcfcn_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'count_mode':1,
         'with_affinity':True,
         'shared':True,
        #  'with_affinity_average':True,
         } ] +

         [
        {'name': 'semseg',
         'loss': 'lcfcn_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'count_mode':1,
        #  'with_affinity':True,
         'shared':True,
        #  'with_affinity_average':True,
         } ] + 

    # [
    #     {'name': 'semseg',
    #      'loss': 'lcfcn_crossentropy_loss',
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_affinity':True,
    #      'shared':True,
    #     #  'with_affinity_average':True,
    #      } ]  +

    # [
    #     {'name': 'semseg',
    #      'loss': 'lcfcn_crossentropy_crf_loss',
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_affinity':True,
    #      'shared':True,
    #     #  'with_affinity_average':True,
    #      } ]  +
     [
        {'name': 'semseg',
         'loss': 'lcfcn_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
         'with_affinity':True,
         'shared':True,
         'beta':beta
        #  'with_affinity_average':True,
         } for beta in [7]] +

         [
        {'name': 'semseg',
         'loss': 'lcfcn_loss',
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
        #  'with_affinity':True,
         'shared':True,
        #  'with_affinity_average':True,
         }] 
        #  + 
    # [
    #     {'name': 'semseg',
    #      'loss': 'lcfcn_loss',
    #      'base': 'fcn8_vgg16',
    #      'n_channels': 3, 
    #      'n_classes': 2,
    #      'with_affinity':True,
    #      'shared':True,
    #      'logt':logt
    #     #  'with_affinity_average':True,
    #      } for logt in [2,3,5,6]] +  
         
        #  [
        # {'name': 'semseg',
        #  'loss': 'lcfcn_loss',
        #  'base': 'fcn8_vgg16',
        #  'n_channels': 3, 
        #  'n_classes': 2,
        #  'shared':True,
        #  } ] + [
        # {'name': 'semseg',
        #  'loss': 'lcfcn_loss',
        #  'base': 'fcn8_vgg16',
        #  'n_channels': 3, 
        #  'n_classes': 2,
        #  'with_affinity':True,
        #  'shared':True,
        # #  'with_affinity_average':True,
        #  } ]

    

      
        

})


EXP_GROUPS['fish_pseudo'] = hu.cartesian_exp_group({
    'batch_size': [1],
    'num_channels': 1,
    'dataset': [
        {'name': 'JcuFish', 'n_classes': 2},

    ],
    'dataset_size': [
        #  {'train':10, 'val':10, 'test':10},
        {'train': 'all', 'val': 'all'},
    ],
    'max_epoch': [1000],
    'optimizer': ["adam"],
    'lr': [ 1e-5],
    'model':
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
         'n_classes': 2,
          'shared':True,
         'count_mode':1,
         } for l in ['pseudo_mask', ]] +
         
    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
          'shared':True,
         'n_classes': 2,
         } for l in ['pseudo_mask', ]] 
        #  +

        # [
        # {'name': 'semseg',
        #  'loss': l,
        #  'base': 'fcn8_vgg16',
        #  'n_channels': 3, 
        #  'n_classes': 2,
        #  'count_mode':1,
        #  } for l in ['pseudo_mask', ]] +

        # [
        # {'name': 'semseg',
        #  'loss': l,
        #  'base': 'fcn8_vgg16',
        #  'n_channels': 3, 
        #  'n_classes': 2,
        #  } for l in ['pseudo_mask', ]] 

    

      
        

})