import numpy as np
import torch
import cv2

def prepare_inputs(annots_2d, cam_params):
  inputs = torch.zeros(annots_2d.size(1), annots_2d.size(2), 2 * annots_2d.size(0), 4)   # shape => (n_cams, n_frames, n_joints, 2 * n_cams, projection)

  for cam_idx in range(annots_2d.size(0)):
    one_vid_all_frames = annots_2d[cam_idx]
      
    xs = one_vid_all_frames[:, :, 0]
    xs = torch.unsqueeze(xs, dim=2)
    inputs[:, :, cam_idx * 2 + 0, :] = xs * cam_params[cam_idx][2, :] - cam_params[cam_idx][0, :]

    ys = one_vid_all_frames[:, :, 1]
    ys = torch.unsqueeze(ys, dim=2)
    inputs[:, :, cam_idx * 2 + 1, :] = xs * cam_params[cam_idx][2, :] - cam_params[cam_idx][0, :]
      
  return inputs


def calculate_loss(pred_2d, gt_2d, loss_func):
  for cam_idx, cam in enumerate(gt_2d):
    for frame_idx, frame in enumerate(cam):
      for joint_idx, joint in enumerate(frame):
        # check if the joint is detected or not, and check the confidence.
        if joint[0] == 0.0 or joint[1] == 0.0:
          gt_2d[cam_idx][frame_idx][joint_idx] = torch.zeros(2, dtype=torch.float32)
          pred_2d[cam_idx][frame_idx][joint_idx] = torch.zeros(2, dtype=torch.float32)
  
  loss = loss_func(pred_2d, gt_2d)
  return loss


# def project_3d_to_2d(pred_3d_torch, cam_params_list):

#   pred_3d = pred_3d_torch.detach().numpy()
#   cam_list = list(cam_params_list.keys())
#   pred_2d = torch.zeros(len(cam_list), pred_3d.shape[0], pred_3d.shape[1], 2)

#   for idx, cam_name in enumerate(cam_list):
#     R, T, K = cam_params_list[cam_name]
#     R = np.array(R)[:,:3]
#     T = np.array(T)
#     K = np.array(K)
    
#     for frame_idx in range(pred_3d.shape[0]):
#       imagePoints = cv2.projectPoints(pred_3d[frame_idx], R, T, K, distCoeffs=None)
#       imagePoints = torch.tensor(imagePoints[0])
#       imagePoints = torch.squeeze(imagePoints)
#       pred_2d[idx][frame_idx] = imagePoints
#   return pred_2d



# def project_3d_to_2d(pred_3d, cam_params):
#   n_cams = cam_params.size(0)
#   all_preds_2d = torch.zeros(n_cams, pred_3d.size(0), pred_3d.size(1), 2, dtype=torch.float32)
  
#   w = torch.ones(pred_3d.size(0), pred_3d.size(1), 1, dtype=torch.float32)
#   pred_3d = torch.cat((pred_3d, w), axis=2)
#   pred_3d = torch.unsqueeze(pred_3d, axis=-1)
#   for idx in range(n_cams):
#     proj_xyz = cam_params[idx] @ pred_3d
#     proj_xyz = torch.squeeze(proj_xyz, dim=3)
#     xy = proj_xyz[:,:,:2] / torch.unsqueeze(proj_xyz[:,:,2], dim=-1)  # divide z-value to get x and y in image coords
#     all_preds_2d[idx] = xy

#   return all_preds_2d


# def projectPoints(X, K, R, t, Kd):
#   """ Projects points X (3xN) using camera intrinsics K (3x3),
#   extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
  
#   Roughly, x = K*(R*X + t) + distortion
  
#   See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
#   or cv2.projectPoints
#   """
#   for idx in range(4):
#     x = R * X + t
#   x[0:2,:] = x[0:2,:]/x[2,:]
  
#   # r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
#   # x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
#   # x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

#   x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
#   x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
  
#   return x


# def align_3d_scales(pred_3d, label_3d):
#   pred_3d_hip_points = pred_3d[:, 7:9, :]
#   pred_3d_mid_point = torch.mean(pred_3d_hip_points, dim=0)
#   pred_3d_mid_point = torch.mean(pred_3d_mid_point, dim=0)

#   lebel_3d_hip_points = label_3d[:, 7:9, :]
#   lebel_3d_mid_point = torch.mean(lebel_3d_hip_points, dim=0)
#   lebel_3d_mid_point = torch.mean(lebel_3d_mid_point, dim=0)

#   scale = lebel_3d_mid_point / pred_3d_mid_point
  
#   # label_3d[:,:,0] = label_3d[:,:,0] * scale[0]
#   # label_3d[:,:,1] = label_3d[:,:,1] * scale[1]
#   # label_3d[:,:,2] = label_3d[:,:,2] * scale[2]

#   return scale


# def calculate_loss(pred_2d, pred_3d, gt_2d, gt_3d, loss_func):
#   for cam_idx, cam in enumerate(gt_2d):
#     for frame_idx, frame in enumerate(cam):
#       for joint_idx, joint in enumerate(frame):
#         # check if the joint is detected or not, and check the confidence.
#         if joint[0] == 0.0 or joint[1] == 0.0:
#           gt_2d[cam_idx][frame_idx][joint_idx] = torch.zeros(2, dtype=torch.float32)
#           pred_2d[cam_idx][frame_idx][joint_idx] = torch.zeros(2, dtype=torch.float32)
  
#   loss_2d = loss_func(pred_2d, gt_2d)
#   loss_3d = loss_func(pred_3d, gt_3d)
#   total_loss = (loss_2d + 1000.0 * loss_3d) / 2.0
#   print("loss 2d: ", loss_2d)
#   print("loss 3d: ", loss_3d)
#   return total_loss