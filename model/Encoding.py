import torch
import torch.nn as nn
import torchvision

class KeyPointEncoding(nn.Module):
  def __init__(self, encoding_dim, decoding_dim):
    super(KeyPointEncoding, self).__init__()
    assert decoding_dim[-1] == 1
    
    self.mlp_encode = torchvision.ops.MLP(3, encoding_dim, norm_layer=nn.LayerNorm)
    self.mlp_decode = torchvision.ops.MLP(decoding_dim[0], decoding_dim, norm_layer=nn.LayerNorm)
    
    self.init_weights()

  def init_weights(self):
    def weight_init(m):
      if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if getattr(m, 'bias') is not None:
          nn.init.constant_(m.bias, 0)
      # if isinstance(m, nn.ModuleList):
      #   for layer in m:
      #     # linear = layer[0]
      #     nn.init.kaiming_normal_(layer[0].weight)
    self.apply(weight_init)
  
  def forward(self, x):
    assert x.size(-1) == 3
    
    x = self.mlp_encode(x)
    x = torch.permute(x, (0, 1, 3, 2))
    x = self.mlp_decode(x)
    x = torch.squeeze(x)    # shape => (n_frames, n_joints, encoding_dim[-1])
    
    return x
    