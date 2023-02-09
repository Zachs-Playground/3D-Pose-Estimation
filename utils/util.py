import numpy as np
import torch

# -----------------------------------------------------------------------
# functions for ray generation 
# -----------------------------------------------------------------------

def _get_sizes(annots):
  cam_list = list(annots.keys())
  n_cams = len(cam_list)
  cam_idx = cam_list[0]
  n_frames = len(annots[cam_idx])
  n_joints = len(annots[cam_idx][0])
  return n_cams, n_frames, n_joints

def _c2w(u, v, intri, extri):
  ox, oy, fx, fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
  x = (u - ox) / fx
  y = (v - oy) / fy
  w_coord = np.matmul(extri.T, np.array([x, y, 1.]))
  w_coord = w_coord / w_coord[-1]
  return w_coord[:3]

def _get_rays(w_coord, r_mat, t_mat):
  ray_origin = np.matmul(r_mat.T, -t_mat)
  ray_origin = np.squeeze(ray_origin)
  ray_direction = w_coord - ray_origin
  return ray_origin, ray_direction

def generate_rays(cams_params, annots):
  n_cams, n_frames, n_joints = _get_sizes(annots)
  rays = [[[] for i in range(n_joints)] for j in range(n_frames)]
  cam_list = list(cams_params.keys())
  
  for frame_idx in range(n_frames):
    for joint_idx in range(n_joints):
      for cam_idx, cam_key in enumerate(cam_list):
        cam_param = cams_params[cam_key]
        R, T, K = list(cam_param.keys())
        r_mat = cam_param[R]
        t_mat = cam_param[T]
        extri = np.concatenate((r_mat, t_mat), -1)
        intri = cam_param[K]
        key_joint = annots[cam_key][frame_idx][joint_idx]
        x, y, c = key_joint
        if c >= 0.75:
          w_coord = _c2w(x, y, intri, extri)
          ray_o, ray_d = _get_rays(w_coord, r_mat, t_mat)
          rays[frame_idx][joint_idx].append([ray_o, ray_d])

  return rays


# -----------------------------------------------------------------------
# functions for finding closest points between rays
# -----------------------------------------------------------------------

def _calculate_mid_point(ray_1, ray_2):
  ray_1_origin, ray_1_direction = ray_1
  ray_2_origin, ray_2_direction = ray_2
  
  # compute unit vectors of directions of lines A and B
  unit_vector_ray_1 = (ray_1_direction - ray_1_origin) / np.linalg.norm(ray_1_direction - ray_1_origin)
  unit_vector_ray_2 = (ray_2_direction - ray_2_origin) / np.linalg.norm(ray_2_direction - ray_2_origin)

  # find unit direction vector for line C, which is perpendicular to lines A and B
  vector_c = np.cross(unit_vector_ray_2, unit_vector_ray_1)
  vector_c /= np.linalg.norm(vector_c)

  # solve the system
  RHS = ray_2_origin - ray_1_origin
  LHS = np.array([unit_vector_ray_1, -unit_vector_ray_2, vector_c]).T

  return np.linalg.solve(LHS, RHS)


def get_closest_point(rays, n_cams):
  n_rays = sum([n for n in range(n_cams)])
  points = np.zeros((len(rays), len(rays[0]), n_rays, 3), dtype=np.float32)
  for frame_idx, one_frame_rays in enumerate(rays):
    for joint_idx, one_joint_rays in enumerate(one_frame_rays):
      actual_n_rays = len(one_joint_rays)
      one_joint_rays = np.array(one_joint_rays)
      count = 0
      if actual_n_rays > 1:
        for cur_idx in range(actual_n_rays):
          for nxt_idx in range(cur_idx+1, actual_n_rays):
            point = _calculate_mid_point(one_joint_rays[cur_idx], one_joint_rays[nxt_idx])
            points[frame_idx][joint_idx][count] = point
            count += 1
  
  return points


# -----------------------------------------------------------------------
# util functions
# -----------------------------------------------------------------------

def project_3d_to_2d(pred_3d, cam_params):
  cam_names = list(cam_params.keys())
  all_preds_2d = torch.zeros(len(cam_names), pred_3d.size(0), pred_3d.size(1), 2)
  
  w = torch.ones(pred_3d.size(0), pred_3d.size(1), 1)
  pred_3d = torch.cat((pred_3d, w), axis=2)
  pred_3d = torch.unsqueeze(pred_3d, axis=-1)
  for idx, cam in enumerate(cam_names):
    R, T, intri = list(cam_params[cam].values())
    extri = torch.cat((torch.tensor(R, dtype=torch.float32),torch.tensor(T, dtype=torch.float32)), axis=1)
    pred_2d = torch.tensor(intri, dtype=torch.float32) @ extri @ pred_3d
    pred_2d = torch.squeeze(pred_2d)
    xy = pred_2d[:,:,:2] / torch.unsqueeze(pred_2d[:,:,2], dim=-1)  # divide z-value to get x and y in image coords
    all_preds_2d[idx] = xy
  return all_preds_2d


def get_gt_annots(all_annots, start, end):
  keys = list(all_annots.keys())
  annots = []
  for key in keys:
    annots.append(all_annots[key][start:end])
  return torch.tensor(annots)


def calculate_loss(pred_2d, gt, loss_func):
  for cam_idx, cam in enumerate(gt):
    for frame_idx, frame in enumerate(cam):
      for joint_idx, joint in enumerate(frame):
        if joint[2] < 0.75:
          gt[cam_idx][frame_idx][joint_idx] = torch.zeros(3)
          pred_2d[cam_idx][frame_idx][joint_idx] = torch.zeros(2)
  gt = gt[:,:,:,:2]
  loss = loss_func(pred_2d, gt)
  return loss




  # def _get_rays(self, w_coord, r_mat, t_mat, n_linespace):
  #   ray_origin = np.matmul(r_mat.T, -t_mat)
  #   ray_origin = np.squeeze(ray_origin)
  #   ray_direction = w_coord - ray_origin
  #   t_vals = np.linspace(0., 2., num=(n_linespace - 1))
  #   points = [(ray_origin + ray_direction * t_val) for t_val in t_vals]
  #   points = np.array(points, dtype=np.float32)
  #   return points
  
  # def __call__(self, n_linespace=15):
  #   n_cams, n_frames, n_joints = self._get_sizes()
  #   all_points = [[[] for i in range(n_joints)] for j in range(n_frames)]
  #   cam_list = list(self.cams_params.keys())
    
  #   for frame_idx in range(n_frames):
  #     for joint_idx in range(n_joints):
  #       for cam_idx, cam_key in enumerate(cam_list):
  #         cam_param = self.cams_params[cam_key]
  #         R, T, K = list(cam_param.keys())
  #         r_mat = cam_param[R]
  #         t_mat = cam_param[T]
  #         extri = np.concatenate((r_mat, t_mat), -1)
  #         intri = cam_param[K]
  #         key_joint = self.annots[cam_key][frame_idx][joint_idx]
  #         x, y, c = key_joint
  #         if c >= 0.75:
  #           w_coord = self._c2w(x, y, intri, extri)
  #           point = self._get_rays(w_coord, r_mat, t_mat, n_linespace)
  #         all_points[frame_idx][joint_idx].append(point)

  #   return all_points
