import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import typing as tp
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        #seq_len: int,
        in_channels: tp.Dict[str, int],
        #out_channels: int,
        hidden: tp.Dict[str, int],
        num_channels = 271,
        hid_dim: int = 128,
        dilation: int = 2,
        num_subjects: int = 4,
        subject_emb_dim: int = 4,
        coord_dim: int = 2,
        classify: bool = False,
        depth: int = 4,
        concatenate: bool = False,
        linear_out: bool = False,
        complex_out: bool = False,
        kernel_size: int = 5,
        growth: float = 1.,
        dilation_growth: int = 2,
        dilation_period: tp.Optional[int] = None,
        skip: bool = False,
        post_skip: bool = False,
        scale: tp.Optional[float] = None,
        rewrite: bool = False,
        groups: int = 1,
        glu: int = 0,
        glu_context: int = 0,
        glu_glu: bool = True,
        gelu: bool = False,
        dual_path: int = 0,
        conv_dropout: float = 0.5,
        dropout_input: float = 0.0,
        batch_norm: bool = True,
        relu_leakiness: float = 0.0,
        n_subjects: int = 200,
        subject_dim: int = 64,
        subject_layers: bool = False,
        subject_layers_dim: str = "input",
        subject_layers_id: bool = False,
        embedding_scale: float = 1.0,
        n_fft: tp.Optional[int] = None,
        fft_complex: bool = True,
        spatial_attention: bool = False,
        pos_dim: int = 256,
        dropout: float = 0.,
        dropout_rescale: bool = True,
        initial_linear: int = 0,
        initial_depth: int = 1,
        initial_nonlin: bool = False,
        subsample_meg_channels: int = 0,
    ) -> None:
        super().__init__()

        self.spatial_attention = SpatialAttention(num_channels, 270, coord_dim) # (b, 271, 161) -> (b, 270, 161)
        self.subject_specific_linear_layer = SubjectSpecificLinearLayer(270, 270, num_subjects, subject_emb_dim) # (b, 270, 161) -> (b, 270, 161)
        
        
        sizes = {}
        for name in in_channels:
            sizes[name] = [in_channels[name]]
            sizes[name] += [int(round(hidden[name] * growth ** k)) for k in range(depth)]
        
        params: tp.Dict[str, tp.Any]
        params = dict(kernel=kernel_size, stride=1,
                      leakiness=relu_leakiness, dropout=conv_dropout, dropout_input=dropout_input,
                      batch_norm=batch_norm, dilation_growth=dilation_growth, groups=groups,
                      dilation_period=dilation_period, skip=skip, post_skip=post_skip, scale=scale,
                      rewrite=rewrite, glu=glu, glu_context=glu_context, glu_glu=glu_glu,
                      activation=nn.GELU)
        
        self.encoders = nn.ModuleDict({name: ConvSequence(channels, **params)
                                       for name, channels in sizes.items()})
        
        self.linear = LinearProjection(320, 2048)

        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(2048, 2048, 1),
            nn.AdaptiveAvgPool1d(1),
        )

        self.mlp_projector = nn.Sequential(
            Rearrange("b d 1 -> b d"),
            nn.Linear(2048, 512),
            nn.ReLU(),
        )
        
        self.classify = classify
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # ドロップアウトを追加
            nn.Linear(512, num_classes)
        )
            

    def forward(self, X: torch.Tensor, subject_index: torch.Tensor, pos) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
            subject_index (b): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.spatial_attention(X, pos)
        X = self.subject_specific_linear_layer(X, subject_index)
        X = self.encoders['meg'](X)
        X = self.linear(X)
        X = self.temporal_aggregation(X)
        X = self.mlp_projector(X)
        if self.classify:
            X = self.classifier(X)
        return X


class SpatialAttention(nn.Module):
    def __init__(self, in_dim, out_dim, coord_dim):
        super().__init__()
        self.coord_proj = nn.Linear(coord_dim, out_dim)

    def forward(self, X: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # coords: (b, c, 2)
        coords_proj = self.coord_proj(coords)  # (b, c, out_dim)
        coords_proj = coords_proj.permute(0, 2, 1)  # (b, out_dim, c)
        
        # X: (b, c, t)
        output = torch.matmul(coords_proj, X)  # (b, out_dim, c) x (b, c, t) -> (b, out_dim, t)
        
        return output



class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)


class LinearProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.batchnorm = nn.BatchNorm1d(out_dim)
        self.activation = nn.GELU()

    def forward(self, X):
        X = self.proj(X.transpose(1, 2)).transpose(1, 2)
        X = self.batchnorm(X)
        return self.activation(X)


class SubjectSpecificLinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_subjects, subject_emb_dim):
        super().__init__()
        self.subject_embedding = nn.Embedding(num_subjects, subject_emb_dim)
        self.proj = nn.Linear(in_dim + subject_emb_dim, out_dim)
        self.batchnorm = nn.BatchNorm1d(out_dim)
        self.activation = nn.GELU()

    def forward(self, X, subject_index):
        subject_emb = self.subject_embedding(subject_index).unsqueeze(2)  # (b, emb_dim, 1)
        subject_emb = subject_emb.expand(-1, -1, X.size(2))  # (b, emb_dim, t)
        X = torch.cat([X, subject_emb], dim=1)  # (b, in_dim + emb_dim, t)
        X = self.proj(X.transpose(1, 2)).transpose(1, 2)
        X = self.batchnorm(X)
        return self.activation(X)


class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dilation, kernel_size: int = 3, p_drop: float = 0.1):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, dilation=dilation, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, dilation=dilation, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """
    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class ConvSequence(nn.Module):

    def __init__(self, channels: tp.Sequence[int], kernel: int = 4, dilation_growth: int = 1,
                 dilation_period: tp.Optional[int] = None, stride: int = 2,
                 dropout: float = 0.0, leakiness: float = 0.0, groups: int = 1,
                 decode: bool = False, batch_norm: bool = False, dropout_input: float = 0,
                 skip: bool = False, scale: tp.Optional[float] = None, rewrite: bool = False,
                 activation_on_last: bool = True, post_skip: bool = False, glu: int = 0,
                 glu_context: int = 0, glu_glu: bool = True, activation: tp.Any = None) -> None:
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(Conv(chin, chout, kernel, stride, pad,
                               dilation=dilation, groups=groups if k > 0 else 1))
            dilation *= dilation_growth
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context), act))
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x

