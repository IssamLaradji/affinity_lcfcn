import os


ACCOUNT_ID = os.environ['EAI_ACCOUNT_ID'] 

JOB_CONFIG = {
                                'image': 'registry.console.elementai.com/%s/ssh' % os.environ['EAI_ACCOUNT_ID'] ,
                                'data': [
                                         'eai.colab.public:/mnt/public',
                                         ],
                                'restartable':True,
                                'resources': {
                                    'cpu': 4,
                                    'mem': 8,
                                    'gpu': 1
                                },
                                'interactive': False,
                                'bid': 5000,
                                }