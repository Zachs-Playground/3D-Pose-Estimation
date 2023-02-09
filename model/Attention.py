import copy
from typing import Optional, Any, Union, Callable, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets

from model.Encoding import KeyPointEncoding


def _get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# ===============================================================
#                      Encoder Component
# ===============================================================


class TransformerEncoderLayer(nn.Module):
  __constants__ = ['batch_first', 'norm_first']

  def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                device=None, dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(TransformerEncoderLayer, self).__init__()


    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                        **factory_kwargs)

    self.relu = nn.ReLU(inplace=False)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

    self.norm_first = norm_first
    self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

    x = src
    
    if self.norm_first:
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))

    return x


  # self-attention block
  def _sa_block(self, x: Tensor,
                attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
    y = self.self_attn(x, x, x,
                      attn_mask=attn_mask,
                      key_padding_mask=key_padding_mask,
                      need_weights=False)[0]

    return self.dropout1(x + y)

  # feed forward block
  def _ff_block(self, x: Tensor) -> Tensor:
    y = self.linear1(x)
    y = self.relu(y)
    y = self.dropout(y)
    y = self.linear2(y)
    return self.dropout2(y)


class TransformerEncoder(nn.Module):
  __constants__ = ['norm']

  def __init__(self, encoder_layer, num_layers, d_model, norm=None):
    super(TransformerEncoder, self).__init__()
    self.layers = _get_clones(encoder_layer, num_layers)
    self.num_layers = num_layers
    self.norm = norm

  def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

    output = src

    for mod in self.layers:
        output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

    if self.norm is not None:
        output = self.norm(output)

    return output



# ===============================================================
#                      Decoder Component
# ===============================================================


class TransformerDecoderLayer(nn.Module):

  __constants__ = ['batch_first', 'norm_first']

  def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                device=None, dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(TransformerDecoderLayer, self).__init__()

    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                        **factory_kwargs)

    self.relu = nn.ReLU(inplace=False)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

    self.norm_first = norm_first
    self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

    x = src
    
    if self.norm_first:
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))

    return x

    # self-attention block
  def _sa_block(self, x: Tensor,
                attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
    y = self.self_attn(x, x, x,
                      attn_mask=attn_mask,
                      key_padding_mask=key_padding_mask,
                      need_weights=False)[0]

    return self.dropout1(x + y)

    # feed forward block
  def _ff_block(self, x: Tensor) -> Tensor:
    y = self.linear1(x)
    y = self.relu(y)
    y = self.dropout(y)
    y = self.linear2(y)
    return self.dropout2(y)


class TransformerDecoder(nn.Module):

  __constants__ = ['norm']

  def __init__(self, decoder_layer, num_layers, d_model, norm=None):
    super(TransformerDecoder, self).__init__()
    self.layers = _get_clones(decoder_layer, num_layers)
    self.num_layers = num_layers
    self.norm = norm
    
  def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

    output = src

    for mod in self.layers:
        output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

    if self.norm is not None:
        output = self.norm(output)

    return output



# ===============================================================
#                        Main Component
# ===============================================================


class RayPoseAttention(nn.Module):
  def __init__(self, encoding_dim: List[int], decoding_dim: List[int], 
                n_sequence: int = 5, n_joints: int = 25,  
                d_model: int = 1024, nhead: int = 8, num_encoder_layers: int = 4,
                num_decoder_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                device=None, dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(RayPoseAttention, self).__init__()
    
    self.feature_encoding = KeyPointEncoding(encoding_dim, decoding_dim)
  
    self.n_sequence = n_sequence
    self.n_joints = n_joints
    # self.norm = norm

    # Embedding Layer
    spatial_array = torch.LongTensor([[i for i in range(n_joints)]] * n_sequence)
    # spatial_array = torch.reshape(spatial_array, (n_joints * n_sequence, d_model))
    s_embedding = nn.Embedding(n_joints, d_model)
    self.spatial_embedding = s_embedding(spatial_array)
    # self.spatial_embedding = torch.reshape(self.spatial_embedding, (n_joints * n_sequence, d_model)).cuda()
    self.spatial_embedding = torch.reshape(self.spatial_embedding, (n_joints * n_sequence, d_model))

    frames_array = torch.LongTensor([[i] * n_joints for i in range(n_sequence)])
    # frames_array = torch.reshape(frames_array, (n_joints * n_sequence, d_model))
    t_embedding = nn.Embedding(n_sequence, d_model)
    self.temporal_embedding = t_embedding(frames_array)
    # self.temporal_embedding = torch.reshape(self.temporal_embedding, (n_joints * n_sequence, d_model)).cuda()
    self.temporal_embedding = torch.reshape(self.temporal_embedding, (n_joints * n_sequence, d_model))

    # Downsample dimension Layer
    self.linear_projection = nn.Linear(encoding_dim[-1], d_model)
    
    # Combine frames layer
    self.down_to_one_frame = nn.Linear(n_joints * n_sequence, n_joints)

    self.relu = nn.ReLU(inplace=False)
    self.layernorm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.layernorm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            activation, layer_norm_eps, batch_first, norm_first,
                                            **factory_kwargs)
    
    decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            activation, layer_norm_eps, batch_first, norm_first,
                                            **factory_kwargs)

    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

    self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, 
                                      d_model, encoder_norm)
    
    self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, 
                                      d_model, decoder_norm)

    self.linear_prediction = nn.Linear(d_model, 3)

    self._reset_parameters()

    self.d_model = d_model
    self.nhead = nhead

    self.batch_first = batch_first


  def forward(self, src: Tensor, batch_size: int, src_mask: Optional[Tensor] = None, 
              src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

    # is_batched = src.dim() == 3
    # if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
    #   raise RuntimeError("the batch number of src and tgt must be equal")
    # elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
    #   raise RuntimeError("the batch number of src and tgt must be equal")

    # if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
    #   raise RuntimeError("the feature number of src and tgt must be equal to d_model")

    """"""

    # if src.size(1) != self.n_sequence:
    #   raise RuntimeError("src.size(1) does not equal to the number of frames")
    # if src.size(2) != self.n_joints:
    #   raise RuntimeError("src.size(2) does not equal to the number of joints")
    
    x = src

    # Combine all frames into on matrix
    """"""
    # x = torch.reshape(x, (x.size(0), self.n_sequence * self.n_joints, -1))
    # x = torch.reshape(x, (self.n_sequence * self.n_joints, -1))
    print(x.shape)
    x = self.feature_encoding(x)   # shape => (n_frames, n_joints, feature_dim)
    hold = torch.zeros(batch_size, self.n_sequence * self.n_joints, x.size(-1))
    for b in range(batch_size):
      one_batch = x[b : b + self.n_sequence]
      hold[b] = torch.reshape(one_batch, (-1, x.size(-1)))

    x = hold
    
    # Align input feature size to Attention dimension
    x = self.linear_projection(x)
    x = self.relu(x)

    # Add spatial and temporal embeddings
    x = x + self.spatial_embedding + self.temporal_embedding

    x = self.layernorm1(x)

    x = self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    # Add temporal embedding before commbining information of all frames into one
    x = x + self.temporal_embedding + self.spatial_embedding

    # Add spatial embedding before feeding the information into decoder
    # x += self.spatial_embedding
    x = self.layernorm2(x)


    """"""
    x = torch.transpose(x, 1, 2)  # I need transpose because the donwsample must be column-wise, not row-wise
    # x = torch.transpose(x, 0, 1)
    x = self.down_to_one_frame(x)
    x = self.relu(x)
    """"""
    x = torch.transpose(x, 2, 1)
    # x = torch.transpose(x, 1, 0)
    
    # the decoder is regression
    x = self.decoder(x)
    x = self.linear_prediction(x)
    return x


  @staticmethod
  def generate_square_subsequent_mask(sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


  def _reset_parameters(self):
    r"""Initiate parameters in the transformer model."""

    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)