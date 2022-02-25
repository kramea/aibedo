from typing import List, Dict, Tuple
import torch
from neuralhydrology.training.regularization import BaseRegularization

from . import Config

class BaseLoss(torch.nn.Module):
    """Base loss class.

    All losses extend this class by implementing `_get_loss`.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    prediction_keys : List[str]
        List of keys that will be predicted. During the forward pass, the passed `prediction` dict
        must contain these keys. Note that the keys listed here should be without frequency identifier.
    ground_truth_keys : List[str]
        List of ground truth keys that will be needed to compute the loss. During the forward pass, the
        passed `data` dict must contain these keys. Note that the keys listed here should be without
        frequency identifier.
    additional_data : List[str], optional
        Additional list of keys that will be taken from `data` in the forward pass to compute the loss.
        For instance, this parameter can be used to pass the variances that are needed to compute an NSE.
    output_size_per_target : int, optional
        Number of model outputs (per element in `prediction_keys`) connected to a single target variable, by default 1. 
        For example for regression, one output (last dimension in `y_hat`) maps to one target variable. For mixture 
        models (e.g. GMM and CMAL) the number of outputs per target corresponds to the number of distributions 
        (`n_distributions`).
    """

    def __init__(self,
                 cfg: Config,
                 prediction_keys: List[str],
                 ground_truth_keys: List[str],
                 additional_data: List[str] = None):
        super(BaseLoss, self).__init__()
        self._predict_last_n = cfg.predict_last_n
        self._frequencies = cfg.use_frequencies
        self._output_size_per_target = cfg.output_dim

        self._regularization_terms = []

        # names of ground truth and prediction keys to be unpacked and subset to predict_last_n items.
        self._prediction_keys = prediction_keys
        self._ground_truth_keys = ground_truth_keys

        # subclasses can use this list to register inputs to be unpacked during the forward call
        # and passed as kwargs to _get_loss() without subsetting.
        self._additional_data = []
        if additional_data is not None:
            self._additional_data = additional_data

        # all classes allow per-target weights for multi-target settings. By default, all targets are weighted equally
        if cfg.target_weights is None:
            weights = torch.tensor([1 / cfg.output_dim for _ in range(cfg.output_dim)])
        else:
            if len(cfg.target_weights) == cfg.output_dim:
                weights = torch.tensor(cfg.target_weights)
            else:
                raise ValueError("Number of weights must be equal to the number of target variables")
        self._target_weights = weights

    def forward(self, prediction: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the loss.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Dictionary of predictions for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        data : Dict[str, torch.Tensor]
            Dictionary of ground truth data for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.

        Returns
        -------
        torch.Tensor
            The calculated loss.
        """
        # unpack loss-specific additional arguments
        kwargs = {key: data[key] for key in self._additional_data}

        losses = []
        prediction_sub, ground_truth_sub = {}, {}
        for freq in self._frequencies:
            if self._predict_last_n[freq] == 0:
                continue  # no predictions for this frequency
            freq_suffix = '' if freq == '' else f'_{freq}'

            # apply predict_last_n and mask for all outputs of this frequency at once
            freq_pred, freq_gt = self._subset_in_time(
                {key: prediction[f'{key}{freq_suffix}'] for key in self._prediction_keys},
                {key: data[f'{key}{freq_suffix}'] for key in self._ground_truth_keys}, self._predict_last_n[freq])

            # remember subsets for multi-frequency component
            prediction_sub.update({f'{key}{freq_suffix}': freq_pred[key] for key in freq_pred.keys()})
            ground_truth_sub.update({f'{key}{freq_suffix}': freq_gt[key] for key in freq_gt.keys()})

            for n_target, weight in enumerate(self._target_weights):
                # subset the model outputs and ground truth corresponding to this particular target
                target_pred, target_gt = self._subset_target(freq_pred, freq_gt, n_target)

                # model hook to subset additional data, which might be different for different losses
                kwargs_sub = self._subset_additional_data(kwargs, n_target)

                loss = self._get_loss(target_pred, target_gt, **kwargs_sub)
                losses.append(loss * weight)

        loss = torch.sum(torch.stack(losses))
        for regularization in self._regularization_terms:
            loss = loss + regularization(prediction_sub, ground_truth_sub,
                                         {k: v for k, v in prediction.items() if k not in self._prediction_keys})
        return loss

    @staticmethod
    def _subset_in_time(prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                        predict_last_n: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ground_truth_sub = {key: gt[:, -predict_last_n:, :] for key, gt in ground_truth.items()}
        prediction_sub = {key: pred[:, -predict_last_n:, :] for key, pred in prediction.items()}

        return prediction_sub, ground_truth_sub

    def _subset_target(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                       n_target: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # determine which output neurons correspond to the n_target target variable
        start = n_target * self._output_size_per_target
        end = (n_target + 1) * self._output_size_per_target
        prediction_sub = {key: pred[:, :, start:end] for key, pred in prediction.items()}

        # subset target by slicing to keep shape [bs, seq, 1]
        ground_truth_sub = {key: gt[:, :, n_target:n_target + 1] for key, gt in ground_truth.items()}

        return prediction_sub, ground_truth_sub

    @staticmethod
    def _subset_additional_data(additional_data: Dict[str, torch.Tensor], n_target: int) -> Dict[str, torch.Tensor]:
        # by default, nothing happens
        return additional_data

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        raise NotImplementedError

    def set_regularization_terms(self, regularization_modules: List[BaseRegularization]):
        """Register the passed regularization terms to be added to the loss function.

        Parameters
        ----------
        regularization_modules : List[BaseRegularization]
            List of regularization functions to be added to the loss during `forward`.
        """
        self._regularization_terms = regularization_modules


class MaskedMSELoss(BaseLoss):
    """Mean squared error loss.

    To use this loss in a forward pass, the passed `prediction` dict must contain
    the key ``y_hat``, and the `data` dict must contain ``y``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(MaskedMSELoss, self).__init__(cfg,prediction_keys=['y_hat'], ground_truth_keys=['y'])

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        loss = 0.5 * torch.mean((prediction['y_hat'][mask] - ground_truth['y'][mask])**2)
        return loss


class MaskedRMSELoss(BaseLoss):
    """Root mean squared error loss.

    To use this loss in a forward pass, the passed `prediction` dict must contain
    the key ``y_hat``, and the `data` dict must contain ``y``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(MaskedRMSELoss, self).__init__(cfg,prediction_keys=['y_hat'], ground_truth_keys=['y'])

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        loss = torch.sqrt(0.5 * torch.mean((prediction['y_hat'][mask] - ground_truth['y'][mask])**2))
        return loss