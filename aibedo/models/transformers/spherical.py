from collections import OrderedDict
from typing import Optional, List

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch import Tensor
from einops import rearrange

from aibedo.models.modules.upsampling import Transformer2DUpsampling
from aibedo.models.transformers.backbone import TransformerBackbone
from aibedo.utilities.utils import identity, raise_error_if_invalid_type


class SphericalTransformer(TransformerBackbone):
    def __init__(self,
                 embedder: str = 'simple',
                 n_patches_level: int = 4,
                 pre_embed_dim_factor: float = 2.0,
                 decoder: str = 'upsampling2d',
                 upsampling_mode: str = 'conv',
                 *args, **kwargs):
        """
        Args:

        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def get_embedder_and_decoder(self):
        hdim = self.hparams.hidden_dim
        embed_kwargs = dict(in_channels=self.input_dim, embed_dim=hdim)
        if self.hparams.embedder == 'simple':
            self._num_patches = self.spatial_dim
            embedder = SimpleEmbed1D(**embed_kwargs)
            decoder = identity
        elif self.hparams.embedder == 'patch':
            embedder = PatchEmbedSpherical(**embed_kwargs,
                                           patches_level=self.hparams.n_patches_level,
                                           pre_embed_dim_factor=self.hparams.pre_embed_dim_factor)
            self._num_patches = embedder.n_clusters
            if self.hparams.decoder == 'upsampling2d':
                if self.num_patches == 162:
                    scale_by = (9, 18)
                decoder = Transformer2DUpsampling(hdim, scale_by=scale_by, mode=self.hparams.upsampling_mode)
            else:
                raise NotImplementedError(f'Unknown decoder: {self.hparams.decoder}')

        else:
            raise NotImplementedError(f'Embedder {self.hparams.embedder} not implemented')
        return embedder, decoder

    @property
    def num_patches(self) -> int:
        return self._num_patches


class SimpleEmbed1D(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = rearrange(x, 'b s c -> b c s')
        # Shape of x for 1D: [batch-size, #input-channels, #patches]
        B, C, S = x.shape  # C = input_hidden_dim/ num_channels
        x = self.proj(x)
        # Reshape the tensor to channels-last
        x = rearrange(x, 'b c s -> b s c')  # same as: x.reshape(B, S, C)
        return x


class PatchEmbedSpherical(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 embed_dim: int = 256,
                 patches_level: int = 4,
                 pre_embed_dim_factor: float = 2.0
                 ):
        super().__init__()
        assert 1 <= patches_level <= 6, "patches_level must be in [1, 5]"
        cluster_id_idx = np.load(f'../../../notebooks/partition_labels/l{patches_level - 1}_labels.npy')
        assert 0 in cluster_id_idx, "Index of first cluster must start from 0"
        cluster_ids_unique, clusters_n_members = np.unique(cluster_id_idx, return_counts=True)
        self.n_clusters = len(cluster_ids_unique)
        # cluster_ids_unique is same as [0, 1,.., n_clusters-1]
        # clusters_n_members is the number of pixels in each cluster
        #   E.g.: Let cluster_i, n_members_i = cluster_ids_unique[i], clusters_n_members[i], then
        #       the cluster with the ID cluster_i has n_members_i pixels
        unique_member_counts, self.n_clusters_with_member_count = np.unique(clusters_n_members, return_counts=True)
        # unique_member_counts contains all the unique number of pixels that the clusters have
        # n_clusters_with_member_count is the number of clusters with a given number of pixels
        #   E.g.: Let m, nc = unique_member_counts[0], n_clusters_with_member_count[0], then
        #           there are nc clusters with each of them containing m pixels

        cluster_ids_perm = np.zeros(len(cluster_id_idx), dtype=np.int64)
        # cluster_ids_perm will permute the spherical pixels so that the pixels in each cluster are contiguous
        cur_idx = 0
        self.sizes_of_unique_mem_count_clusters = []
        # self.sizes_of_unique_mem_count_clusters will store the total number of pixels
        #   in each set of clusters with unique pixel count
        for i, (mems_i, n_clusters_with_mems) in enumerate(
                zip(unique_member_counts, self.n_clusters_with_member_count)):
            clusters_with_mems_i_start = cur_idx
            for c_i, mems_c_i in zip(cluster_ids_unique, clusters_n_members):
                if mems_c_i == mems_i:
                    indices_of_cluster_i = (cluster_id_idx == c_i).nonzero()[0]
                    cluster_ids_perm[cur_idx:cur_idx + mems_c_i] = indices_of_cluster_i
                    cur_idx += mems_c_i
            assert mems_i * n_clusters_with_mems == cur_idx - clusters_with_mems_i_start
            self.sizes_of_unique_mem_count_clusters += [mems_i * n_clusters_with_mems]
        assert cur_idx == len(cluster_id_idx)
        self.permute_idx = torch.from_numpy(cluster_ids_perm).long()

        # Now, we can get to the actual embedding layers/weights
        pre_embed_dim = int(embed_dim * pre_embed_dim_factor)
        pre_embedders = list()
        for n_pixels_in_cluster in unique_member_counts:
            # pre_embedders will project flattened version of each cluster to a vector of size pre_embed_dim
            pre_embedders += [nn.Linear(in_channels * n_pixels_in_cluster, pre_embed_dim)]

        self.pre_embedders = nn.ModuleList(pre_embedders)
        self.embedder = nn.Linear(pre_embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Shape of input, x: [batch-size, #pixels, #input-channels] = (B, S, C)
        # First, permute x so that the pixels in each patch are contiguous in the spatial dim (dim=1)
        x = x[:, self.permute_idx, :]
        # Now, split x into the multiple sub-tensors based on the number of pixels in each patch
        x: List[Tensor] = list(torch.split(x, self.sizes_of_unique_mem_count_clusters, dim=1))
        # Continue projecting each patch to the pre-embedding dim:
        for i, (x_sub, n_patches_in_x_sub, sub_embedder) in enumerate(
                zip(x, self.n_clusters_with_member_count, self.pre_embedders)
        ):
            # First, we need to extract the patch dim from x_sub & flatten the pixels in each patch to the channel dim:
            # B = batch-size, p = #patches in sub-tensor, m = #pixel-members in each cluster, C = #input-channels
            x_sub = rearrange(x_sub, 'B (p m) C -> B p (m C)', p=n_patches_in_x_sub)
            # Now we can project the (pixels x in-channels) in each patch to the pre-embedding dim
            x[i] = sub_embedder(x_sub)  # new shape: (B, p, pre_embed_dim)
        # Finally, we can concatenate the pre-projected patches back together into P total patches
        x: Tensor = torch.cat(x, dim=1)  # new shape: (B, P, pre_embed_dim)
        # ... and project them to the embedding dim
        x = self.embedder(x)  # new shape: (B, P, embed_dim)
        return x
