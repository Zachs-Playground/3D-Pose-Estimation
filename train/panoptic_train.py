""" Train Panoptic Dataset """

import math
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from util import prepare_inputs, get_mpjpe
from model import Pred3DPose

cuda = False
load_pretrained = True
device = torch.device("cuda" if torch.cuda.is_available() and cuda == True else "cpu")
print("The coda is running on: ", device)
epoch = 200
batch_size = 31
frame_sequence = 5
if frame_sequence % 2 == 0:
  raise RuntimeError("frame_sequence must be an odd number.")

span = frame_sequence // 2


# ---------------------------------------------------------
# data section
# ---------------------------------------------------------
main_folder = ["./panoptic-toolbox/171204_pose1/", "./panoptic-toolbox/171204_pose2/"]


model = Pred3DPose().to(device)
model.train()
# criterion = nn.MSELoss()
criterion = nn.L1Loss()
l2_loss_ref = nn.MSELoss()

if load_pretrained:
  checkpoint = torch.load("./checkpoint_panoptic_L1/epoch6.pt")
  model.load_state_dict(checkpoint['model_state_dict'])
  print("Loaded Checkpoint")

optimizer = optim.Adam(model.parameters(), lr=0.001)
if load_pretrained:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

vid_sets = [[10, 11, 18, 21], [12, 16, 17, 21], [1, 7, 17, 23]]
cam_sets = [["cam_10", "cam_11", "cam_18", "cam_21"], ["cam_12", "cam_16", "cam_17", "cam_21"], ["cam_1", "cam_7", "cam_17", "cam_23"]]

for i in range(epoch):

  avg_loss_per_epoch = 0.0
  l2_loss_per_epoch = 0.0
  total_mpjpe = 0.0
  count = 0
  sub_count = 0

  for pose_idx, pose_folder in enumerate(main_folder):
    with open(pose_folder + "hd_cam.json", "r") as camf:
      cam_params = json.loads(camf.read())    #  4cams 

    with open(pose_folder + "adjusted3d.json", "r") as annot3df:
      labels_3d = json.loads(annot3df.read())   # frame, joints, xyz
    
    with open(pose_folder + "adjusted2d.json", "r") as annot2df:
      annot_2d = json.loads(annot2df.read())   # video, frame, joints, xy

    for cam_set_idx, cam_set in enumerate(cam_sets):
      
      cam_param_set = []
      landmark_2d = []

      for cam_name in cam_set:
        cam_param_set.append(cam_params[cam_name])
      
      for vid_idx in vid_sets[cam_set_idx]:
        landmark_2d.append(annot_2d[vid_idx])

      cam_param_set = torch.tensor(cam_param_set, dtype=torch.float32)
      labels_3d = np.array(labels_3d)
      labels_3d = torch.tensor(labels_3d, dtype=torch.float32)
      landmark_2d = np.array(landmark_2d)
      landmark_2d = torch.tensor(landmark_2d, dtype=torch.float32)
      
      # not sure about the cam_params_torch
      inputs = prepare_inputs(landmark_2d, cam_param_set)
         
      for idx in range(span, inputs.size(0) - span, batch_size):
        optimizer.zero_grad()

        small_batch_inputs = inputs[idx - span : idx + batch_size + span].to(device)

        if (small_batch_inputs.size(0) - span) < 1 + span:
          continue
        else:
          pred_3d = model(small_batch_inputs)
          actual_n_frames = pred_3d.size(0)
          
          batch_gt_3d = labels_3d[idx : idx + actual_n_frames].to(device)

          loss = criterion(pred_3d, batch_gt_3d)
          avg_loss_per_epoch += loss
          count += 1

          l2_loss = l2_loss_ref(pred_3d, batch_gt_3d)
          l2_loss_per_epoch += l2_loss

          mpjpe, sub_cnt = get_mpjpe(pred_3d, batch_gt_3d)
          total_mpjpe += mpjpe
          sub_count += sub_cnt
          
          loss.backward()
          optimizer.step()


    # print(f"average loss for epoch: {i}, people: {person_idx}, loss: {avg_loss_per_vid / count}")
  print(f"average loss for epoch: {i}, loss: {avg_loss_per_epoch / count}", f"  L2 Loss: {l2_loss_per_epoch / count}", f"  MPJPE: {total_mpjpe / sub_count}")
    
  torch.save({
      'epoch': i,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
    }, f"./checkpoint_panoptic_L1/epoch{i}.pt")
