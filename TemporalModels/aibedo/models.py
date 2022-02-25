import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple

import logging
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#import xarray

from tqdm import tqdm

from . import Config

LOGGER = logging.getLogger(__name__)


class MTSLSTM(nn.Module):
    
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['lstms', 'transfer_fcs', 'heads']

    def __init__(self, cfg:Config):
        super(MTSLSTM, self).__init__()
        self._cfg = cfg
        self.output_size = cfg.output_dim
        self.lstms = None
        self.transfer_fcs = None
        self.heads = None
        self.dropout = None

        self._slice_timestep = {}
        self._frequency_factors = []

        self._seq_lengths = cfg.seq_length
        self._is_shared_mtslstm = cfg.shared_mtslstm  # default: a distinct LSTM per timescale
        self._transfer_mtslstm_states = cfg.transfer_mtslstm_states  # default: linear transfer layer
        transfer_modes = [None, "None", "identity", "linear"]
        if self._transfer_mtslstm_states["h"] not in transfer_modes \
                or self._transfer_mtslstm_states["c"] not in transfer_modes:
            raise ValueError(f"MTS-LSTM supports state transfer modes {transfer_modes}")

        
        self._frequencies = cfg.use_frequencies
        
        self._input_sizes = cfg.input_size
        self._hidden_size = cfg.hidden_size

        if (self._is_shared_mtslstm
            or self._transfer_mtslstm_states["h"] == "identity"
            or self._transfer_mtslstm_states["c"] == "identity") \
                and any(size != self._hidden_size[self._frequencies[0]] for size in self._hidden_size.values()):
            raise ValueError("All hidden sizes must be equal if shared_mtslstm is used or state transfer=identity.")

        # create layer depending on selected frequencies
        self._init_modules(self._input_sizes)
        self._reset_parameters()

        # frequency factors are needed to determine the time step of information transfer
        #self._init_frequency_factors_and_slice_timesteps()
        self._frequency_factors = cfg.frequency_factors
        self._slice_timestep = cfg.slice_timestep

    def _init_modules(self, input_sizes: Dict[str, int]):
        self.lstms = nn.ModuleDict()
        self.transfer_fcs = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.dropout = nn.Dropout(p=self._cfg.output_dropout)
        for idx, freq in enumerate(self._frequencies):
            freq_input_size = input_sizes[freq]

            if self._is_shared_mtslstm and idx > 0:
                self.lstms[freq] = self.lstms[self._frequencies[idx - 1]]  # same LSTM for all frequencies.
                self.heads[freq] = self.heads[self._frequencies[idx - 1]]  # same head for all frequencies.
            else:
                self.lstms[freq] = nn.LSTM(input_size=freq_input_size, hidden_size=self._hidden_size[freq])
                #self.heads[freq] = get_head(self.cfg, n_in=self._hidden_size[freq], n_out=self.output_size)
                self.heads[freq] = Regression(n_in=self._hidden_size[freq], n_out=self.output_size, activation=self._cfg.output_activation)
                
                

            if idx < len(self._frequencies) - 1:
                for state in ["c", "h"]:
                    if self._transfer_mtslstm_states[state] == "linear":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Linear(self._hidden_size[freq],
                                                                         self._hidden_size[self._frequencies[idx + 1]])
                    elif self._transfer_mtslstm_states[state] == "identity":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Identity()
                    else:
                        pass


                
    

    def _reset_parameters(self):
        if self._cfg.initial_forget_bias is not None:
            for freq in self._frequencies:
                hidden_size = self._hidden_size[freq]
                self.lstms[freq].bias_hh_l0.data[hidden_size:2 * hidden_size] = self._cfg.initial_forget_bias
    
    def _prepare_inputs(self, data: Dict[str, torch.Tensor], freq: str) -> torch.Tensor:
        """Concat all different inputs to the time series input"""
        suffix = f"_{freq}"
        # transpose to [seq_length, batch_size, n_features]
        x_d = data[f'x_d{suffix}'].transpose(0, 1)

        # concat all inputs
        if f'x_s{suffix}' in data and 'x_one_hot' in data:
            x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s, x_one_hot], dim=-1)
        elif f'x_s{suffix}' in data:
            x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s], dim=-1)
        elif 'x_one_hot' in data:
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_one_hot], dim=-1)
        else:
            pass

        if self._is_shared_mtslstm:
            # add frequency one-hot encoding
            idx = self._frequencies.index(freq)
            one_hot_freq = torch.zeros(x_d.shape[0], x_d.shape[1], len(self._frequencies)).to(x_d)
            one_hot_freq[:, :, idx] = 1
            x_d = torch.cat([x_d, one_hot_freq], dim=2)

        return x_d

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the MTS-LSTM model.
        
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Input data for the forward pass. See the documentation overview of all models for details on the dict keys.
        Returns
        -------
        Dict[str, torch.Tensor]
            Model predictions for each target timescale.
        """
        x_d = {freq: self._prepare_inputs(data, freq) for freq in self._frequencies}

        # initial states for lowest frequencies are set to zeros
        batch_size = x_d[self._frequencies[0]].shape[1]
        lowest_freq_hidden_size = self._hidden_size[self._frequencies[0]]
        h_0_transfer = x_d[self._frequencies[0]].new_zeros((1, batch_size, lowest_freq_hidden_size))
        c_0_transfer = torch.zeros_like(h_0_transfer)

        outputs = {}
        for idx, freq in enumerate(self._frequencies):
            if idx < len(self._frequencies) - 1:
                # get predictions and state up to the time step of information transfer
                slice_timestep = self._slice_timestep[freq]
                lstm_output_slice1, (h_n_slice1, c_n_slice1) = self.lstms[freq](x_d[freq][:-slice_timestep],
                                                                                (h_0_transfer, c_0_transfer))

                # project the states through a hidden layer to the dimensions of the next LSTM
                if self._transfer_mtslstm_states["h"] is not None:
                    h_0_transfer = self.transfer_fcs[f"h_{freq}"](h_n_slice1)
                if self._transfer_mtslstm_states["c"] is not None:
                    c_0_transfer = self.transfer_fcs[f"c_{freq}"](c_n_slice1)

                # get predictions of remaining part and concat results
                lstm_output_slice2, _ = self.lstms[freq](x_d[freq][-slice_timestep:], (h_n_slice1, c_n_slice1))
                lstm_output = torch.cat([lstm_output_slice1, lstm_output_slice2], dim=0)

            else:
                # for highest frequency, we can pass the entire sequence at once
                lstm_output, _ = self.lstms[freq](x_d[freq], (h_0_transfer, c_0_transfer))

            head_out = self.heads[freq](self.dropout(lstm_output.transpose(0, 1)))
            outputs.update({f'{key}_{freq}': value for key, value in head_out.items()})

        return outputs
    
    
class Regression(nn.Module):
    
    """Single-layer regression head with different output activations.
    
    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons.
    activation : str, optional
        Output activation function. Can be specified in the config using the `output_activation` argument. Supported
        are {'linear', 'relu', 'softplus'}. If not specified (or an unsupported activation function is specified), will
        default to 'linear' activation.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "linear"):
        super(Regression, self).__init__()

        # TODO: Add multi-layer support
        layers = [nn.Linear(n_in, n_out)]
        if activation != "linear":
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "softplus":
                layers.append(nn.Softplus())
            else:
                LOGGER.warning(f"## WARNING: Ignored output activation {activation} and used 'linear' instead.")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Regression head.
        
        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the model predictions in the 'y_hat' key.
        """
        return {'y_hat': self.net(x)}
    
    
def get_predictions_and_loss(model, data, l_obj):
    predictions = model(data)
    loss = l_obj(predictions, data)
    return predictions, loss.item()

def subset_targets(data, predictions, predict_last_n, freq):
    y_hat_sub = predictions[f'y_hat{freq}'][:, -predict_last_n:, :]
    y_sub = data[f'y{freq}'][:, -predict_last_n:, :]
    return y_hat_sub, y_sub


def evaluate(model, loader, l_obj, frequencies, predict_last_n):
    """Evaluate model"""
    # predict_last_n = conf_obj.predict_last_n
    # if isinstance(predict_last_n, int):
    #     predict_last_n = {frequencies[0]: predict_last_n}  # if predict_last_n is int, there's only one frequency

    preds, obs = {}, {}
    losses = []
    with torch.no_grad():
        for data in loader:

            # for key in data:
            #     data[key] = data[key].to(self.device)
            predictions, loss = get_predictions_and_loss(model, data, l_obj)

            for freq in frequencies:
                freq_key = f'_{freq}'
                y_hat_sub, y_sub = subset_targets(data, predictions, predict_last_n[freq], freq_key)

                if freq not in preds:
                    preds[freq] = y_hat_sub.detach().cpu()
                    obs[freq] = y_sub.cpu()
                else:
                    preds[freq] = torch.cat((preds[freq], y_hat_sub.detach().cpu()), 0)
                    obs[freq] = torch.cat((obs[freq], y_sub.detach().cpu()), 0)

            losses.append(loss)

        for freq in preds.keys():
            preds[freq] = preds[freq].numpy()
            obs[freq] = obs[freq].numpy()

    # set to NaN explicitly if all losses are NaN to avoid RuntimeWarning
    mean_loss = np.nanmean(losses) if len(losses) > 0 and not all(np.isnan(l) for l in losses) else np.nan
    return preds, obs, mean_loss