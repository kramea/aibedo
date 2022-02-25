from typing import List, Dict, Union, Tuple
from pathlib import Path
from ruamel.yaml import YAML

class Config():
    def __init__(self, yml_path: Path):
        
        if yml_path.exists():
            with yml_path.open('r') as fp:
                yaml = YAML(typ="safe")
                _cfg_dict = yaml.load(fp)
        else:
            raise FileNotFoundError(yml_path)
        
        self.num_variables = len(_cfg_dict)
        self.input_size = _cfg_dict['input_size']
        self.output_dim = _cfg_dict['output_dim']
        self.hidden_size = _cfg_dict['hidden_size']
        
        self.initial_forget_bias = _cfg_dict['initial_forget_bias']
        self.shared_mtslstm = _cfg_dict['shared_mtslstm']
        self.output_dropout = _cfg_dict['output_dropout']
        
        self.transfer_mtslstm_states = _cfg_dict['transfer_mtslstm_states']
        self.use_frequencies = _cfg_dict['use_frequencies']
        self.output_activation = _cfg_dict['output_activation']
        
        self.device = _cfg_dict['device']
        
        self.seq_length = _cfg_dict['seq_length']
        self.predict_last_n = _cfg_dict['predict_last_n']
        self.slice_timestep = _cfg_dict['slice_timestep']
        self.frequency_factors = _cfg_dict['frequency_factors']
        
        self.target_weights = _cfg_dict['target_weights']

        
         
        