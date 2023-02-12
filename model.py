import torch
import torch.nn as nn
import torchvision
import sys

class Pred3DPose(nn.Module):
    def __init__(self, n_dim=3, n_point_cluster=10, n_joint=17):
        super(Pred3DPose, self).__init__()
        init_outchannels = [20, 100]

        self.mlp_init = torchvision.ops.MLP(n_dim, hidden_channels = init_outchannels, norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace = True, bias = True, dropout = 0.1)
        
        self.mlp_cat_points = torchvision.ops.MLP(n_point_cluster, hidden_channels = [8, 1], norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace = True, bias = True, dropout = 0.1)
        
        self.mlp_spatial = torchvision.ops.MLP(n_joint, hidden_channels = [34, n_joint], norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace = True, bias = True, dropout = 0.1)


        self.motion = torchvision.ops.MLP(init_outchannels[-1], hidden_channels = [20, 100], norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace = True, bias = True, dropout = 0.1)
        self.neighbot_frame = torchvision.ops.MLP(init_outchannels[-1], hidden_channels = [20, 100], norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace = True, bias = True, dropout = 0.1)
        self.trajectory = torchvision.ops.MLP(2 * init_outchannels[-1], hidden_channels = [20, 100], norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace = True, bias = True, dropout = 0.1)

        self.combine = torchvision.ops.MLP(3 * init_outchannels[-1], hidden_channels = [100, 100], norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace = True, bias = True, dropout = 0.1)
        
        self.pred = torchvision.ops.MLP(init_outchannels[-1], hidden_channels = [30, 3], norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace = True, bias = True, dropout = 0.1)
                                            

    def forward(self, x):
        x = self.mlp_init(x)                # shape => frames x joints x cluster points x feature size
        x = torch.transpose(x, -1, -2)      # shape => frames x joints x feature size x cluster points
        x = self.mlp_cat_points(x)          # shape => frames x joints x feature size x 1
        x = torch.squeeze(x, -1)            # shape => frames x joints x feature size
        x = torch.transpose(x, -1, -2)      # shape => frames x feature size x joints
        x = self.mlp_spatial(x)             # shape => frames x feature size x joints
        x = torch.transpose(x, -1, -2)      # shape => frames x joints x feature size

        out = torch.zeros(12, 17, 100)
        for i in range(2, 12):
            motion_forward = x[i - 1] - x[i - 2]
            motion_forward = self.motion(motion_forward)
            previous_frame = self.neighbot_frame(x[i - 1])
            previous_frame = torch.cat((previous_frame, motion_forward), 1)
            previous_frame = self.trajectory(previous_frame)

            motion_backward = x[i + 2] - x[i + 1]
            motion_forward = self.motion(motion_backward)
            next_frame = self.neighbot_frame(x[i + 1])
            next_frame = torch.cat((next_frame, motion_forward), 1)
            next_frame = self.trajectory(next_frame)
            
            concat = torch.cat((previous_frame, x[i], next_frame), 1)
            concat = self.combine(concat)
            out[i - 2] = concat
        
        out = self.pred(out)
        return out


a = torch.rand(16,17,10,3)
m = Pred3DPose()
y = m(a)
print(y.shape)