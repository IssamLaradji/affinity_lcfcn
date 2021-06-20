from haven import haven_utils as hu
import itertools, copy
EXP_GROUPS = {}


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
    'version':[2, None],
    'model':
  
    
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
      
        

}, remove_none=True)


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
    'version':[2, None],
    'model':

    [
        {'name': 'semseg',
         'loss': l,
         'base': 'fcn8_vgg16',
         'n_channels': 3, 
          'shared':True,
         'n_classes': 2,
         } for l in ['pseudo_mask', ]] 
}, remove_none=True)

EXP_GROUPS['fish_final'] =  EXP_GROUPS['fish_shared'] + EXP_GROUPS['fish_pseudo']