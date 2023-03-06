import torch
import torch.nn as nn
import torchvision

class Pred3DPose(nn.Module):
    def __init__(self, n_dim=4, n_point_cluster=8, n_joint=13, sequence=5):
        super(Pred3DPose, self).__init__()
        
        self.n_joint = n_joint
        self.span = sequence // 2
        # self.n_features = 100

        # init_channels = [20, 60, self.n_features]
        # cat_channels = [16, 32, 1]
        # spatial_channels = [n_joint * 2, n_joint * 3, n_joint]
        # temporal_channels = [init_channels[-1] * 2, init_channels[-1] * 3, init_channels[-1]]
        # spatial_channels_2 = [(sequence - self.span) * n_joint * 2, (sequence - self.span) * n_joint * 3, (sequence - self.span) * n_joint]
        # pred_channels = [init_channels[-1] * 3, init_channels[-1], init_channels[-1] // 2, 3]

        # self.n_features = 200
        # init_channels = [20, 60, 100, self.n_features]
        # cat_channels = [16, 32, 8, 1]
        # spatial_channels = [n_joint * 2, n_joint * 4, n_joint, n_joint]
        # temporal_channels = [init_channels[-1] * 2, init_channels[-1] * 3, init_channels[-1], init_channels[-1]]
        # spatial_channels_2 = [(sequence - self.span) * n_joint * 2, (sequence - self.span) * n_joint * 3, (sequence - self.span) * n_joint, (sequence - self.span) * n_joint]
        # pred_channels = [init_channels[-1] * 3, init_channels[-1], init_channels[-1], init_channels[-1] // 2, 3]

        self.n_features = 50
        init_channels = [20, self.n_features]
        cat_channels = [16, 1]
        spatial_channels = [n_joint * 2, n_joint]
        temporal_channels = [init_channels[-1] * 2, init_channels[-1]]
        spatial_channels_2 = [(sequence - self.span) * n_joint * 2, (sequence - self.span) * n_joint]
        pred_channels = [init_channels[-1] * 2, init_channels[-1] // 2, 3]


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
        
        self.mlp_spatial_2 = torchvision.ops.MLP((sequence - self.span) * n_joint, hidden_channels = spatial_channels_2, norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace=False, bias = True, dropout = 0.1)
        
        self.pred = torchvision.ops.MLP(3 * init_channels[-1], hidden_channels = pred_channels, norm_layer = nn.LayerNorm, 
                                            activation_layer = nn.ReLU, inplace=False, bias = True, dropout = 0.1)
        
        self.init_weights()


    def init_weights(self):
        def weight_init(layer):
            # if getattr(m, 'bias') is not None:
            #     print("bias")
            #     nn.init.constant_(m.bias, 0)
            if isinstance(layer, torchvision.ops.MLP):
                for linear in layer:
                    if linear == nn.Linear:
                        nn.init.kaiming_normal_(linear.weight)
                        # nn.init.constant_(linear.bias, 0)
        self.apply(weight_init)
                                            
    def _get_motion(self, a, b):
        for joint_idx in range(a.size(0)):
            if torch.sum(a[joint_idx]).item() == 0.0 or torch.sum(b[joint_idx]).item() == 0.0:
                a[joint_idx] = torch.zeros_like(a[joint_idx])
                b[joint_idx] = torch.zeros_like(b[joint_idx])
        return b - a
    
    def forward(self, x):
        n_frames = x.size(0) - self.span * 2
        x = self.mlp_init(x)                # shape => frames x joints x cluster points x feature size
        x = torch.transpose(x, -1, -2)      # shape => frames x joints x feature size x cluster points
        x = self.mlp_cat_points(x)          # shape => frames x joints x feature size x 1
        x = torch.squeeze(x, -1)            # shape => frames x joints x feature size
        x = torch.transpose(x, -1, -2)      # shape => frames x feature size x joints
        x = self.mlp_spatial(x)             # shape => frames x feature size x joints
        x = torch.transpose(x, -1, -2)      # shape => frames x joints x feature size
        x = self.process_features(x)

        target_frames = x[self.span : n_frames + self.span]
        previous_t1_frames = x[self.span - 1 : n_frames + self.span - 1]
        next_t1_frames = x[self.span + 1 : n_frames + self.span + 1]
        forward_motion = torch.ones_like(target_frames)
        backward_motion = torch.ones_like(target_frames)
        out = torch.zeros(n_frames, self.n_joint, self.n_features, dtype=torch.float32)
        for i in range(self.span, n_frames + self.span):
            forward_flow = self._get_motion(x[i - 2], x[i - 1])
            forward_motion[i - self.span] = forward_flow
            backward_flow = self._get_motion(x[i + 1], x[i + 2])
            backward_motion[i - self.span] = backward_flow

        forward_motion = self.motion(forward_motion)
        backward_motion = self.motion(backward_motion)

        forward_pred = torch.cat((forward_motion, previous_t1_frames), 2)
        backward_pred = torch.cat((backward_motion, next_t1_frames), 2)
        
        forward_pred = self.trajectory(forward_pred)
        backward_pred = self.trajectory(backward_pred)

        out = torch.cat((forward_pred, target_frames, backward_pred), 1)
        out = torch.transpose(out, -2,-1)
        out = self.mlp_spatial_2(out)
        out = torch.transpose(out, -2,-1)
        out = torch.cat((out[:, 0:13,:], out[:, 13:26,:], out[:, 26:39,:]), 2)
        out = self.pred(out)
        return out