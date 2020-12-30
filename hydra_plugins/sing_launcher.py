from hydra_plugins.hydra_submitit_launcher.submitit_launcher import SlurmLauncher
from typing import Any, Dict, List, Optional, Sequence
import sys

class SingLauncher(SlurmLauncher):
    def __init__(self, **params: Any) -> None:
        sing_img = params['comment'] 
        sys.executable = f'singularity exec --nv {sing_img} python'
        super(SingLauncher, self).__init__(**params)
        if 'setup' not in self.params.keys():
            self.params['setup'] = []
        self.params['setup'].append('module load singularity')
        self.params['gres'] = 'gpu:1'
