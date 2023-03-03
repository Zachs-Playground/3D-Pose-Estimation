import torch
import torch.nn as nn
import torchvision

class Pred3DPose(nn.Module):
    def __init__(self, n_dim=4, n_point_cluster=8, n_joint=13):
        super(Pred3DPose, self).__init__()
        
        self.n_joint = n_joint
        # self.n_features = 100

        # init_channels = [20, 60, self.n_features]
        # cat_channels = [16, 32, 1]
        # spatial_channels = [n_joint * 2, n_joint * 3, n_joint]
        # temporal_channels = [init_channels[-1] * 2, init_channels[-1] * 3, init_channels[-1]]
        # pred_channels = [init_channels[-1] * 3, init_channels[-1], init_channels[-1] // 2, 3]

        self.n_features = 200

        init_channels = [20, 60, 100, self.n_features]
        cat_channels = [16, 32, 8, 1]
        spatial_channels = [n_joint * 2, n_joint * 3, n_joint * 2, n_joint]
        temporal_channels = [init_channels[-1] * 2, init_channels[-1] * 3, init_channels[-1] * 2, init_channels[-1]]
        pred_channels = [init_channels[-1] * 3, init_channels[-1] * 2, init_channels[-1], init_channels[-1] // 2, 3]

        self.mlp_init = torchvision.ops.MLP(n_dim, hidden_channels = init_channels, norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace=False, bias = True, dropout = 0.1)
        
        self.mlp_cat_points = torchvision.ops.MLP(n_point_cluster, hidden_channels = cat_channels, norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace=False, bias = True, dropout = 0.1)
        
        self.mlp_spatial = torchvision.ops.MLP(n_joint, hidden_channels = spatial_channels, norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace=False, bias = True, dropout = 0.1)

        self.motion = torchvision.ops.MLP(init_channels[-1], hidden_channels = temporal_channels, norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace=False, bias = True, dropout = 0.1)
        
        self.process_features = torchvision.ops.MLP(init_channels[-1], hidden_channels = temporal_channels, norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace=False, bias = True, dropout = 0.1)
        
        self.trajectory = torchvision.ops.MLP(2 * init_channels[-1], hidden_channels = temporal_channels, norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace=False, bias = True, dropout = 0.1)
        
        self.pred = torchvision.ops.MLP(3 * init_channels[-1], hidden_channels = pred_channels, norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace=False, bias = True, dropout = 0.1)
                                            
    def _get_motion(self, a, b):
        for joint_idx in range(a.size(0)):
            if torch.sum(a[joint_idx]).item() == 0.0 or torch.sum(b[joint_idx]).item() == 0.0:
                a[joint_idx] = torch.zeros_like(a[joint_idx])
                b[joint_idx] = torch.zeros_like(b[joint_idx])
        return b - a
    
    def forward(self, x, span):
        n_frames = x.size(0) - span * 2
        x = self.mlp_init(x)                # shape => frames x joints x cluster points x feature size
        x = torch.transpose(x, -1, -2)      # shape => frames x joints x feature size x cluster points
        x = self.mlp_cat_points(x)          # shape => frames x joints x feature size x 1
        x = torch.squeeze(x, -1)            # shape => frames x joints x feature size
        x = torch.transpose(x, -1, -2)      # shape => frames x feature size x joints
        x = self.mlp_spatial(x)             # shape => frames x feature size x joints
        x = torch.transpose(x, -1, -2)      # shape => frames x joints x feature size
        x = self.process_features(x)
        
        target_frames = x[span : n_frames + span]
        previous_t1_frames = x[span - 1 : n_frames + span - 1]
        next_t1_frames = x[span + 1 : n_frames + span + 1]
        forward_motion = torch.ones_like(target_frames)
        backward_motion = torch.ones_like(target_frames)
        out = torch.zeros(n_frames, self.n_joint, self.n_features, dtype=torch.float32)
        for i in range(span, n_frames + span):
            forward_flow = self._get_motion(x[i - 2], x[i - 1])
            forward_motion[i - span] = forward_flow
            backward_flow = self._get_motion(x[i + 1], x[i + 2])
            backward_motion[i - span] = backward_flow
        
        forward_motion = self.motion(forward_motion)
        backward_motion = self.motion(backward_motion)

        forward_pred = torch.cat((forward_motion, previous_t1_frames), 2)
        backward_pred = torch.cat((backward_motion, next_t1_frames), 2)
        
        forward_pred = self.trajectory(forward_pred)
        backward_pred = self.trajectory(backward_pred)

        out = torch.cat((forward_pred, target_frames, backward_pred), 2)
        out = self.pred(out)
        return out