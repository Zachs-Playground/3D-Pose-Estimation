""" Train """

import math
import json

import torch
import torch.nn as nn
import torch.optim as optim


from util import prepare_inputs
from model import Pred3DPose

cuda = True
load_pretrained = False
device = torch.device("cuda" if torch.cuda.is_available() and cuda == True else "cpu")
print("The coda is running on: ", device)
epoch = 200
batch_size = 31
frame_sequence = 5
if frame_sequence % 2 == 0:
  raise RuntimeError("frame_sequence must be an odd number.")

span = frame_sequence // 2


cam_path = "./data/processed_data/cam.json"    
annot_3d_path = "./data/processed_data/annot_3d.json"   
annot_2d_path = "./data/processed_data/annot_2d.json"   


# ---------------------------------------------------------
# data section
# ---------------------------------------------------------
with open(cam_path, "r") as camf:
  cam_params = json.loads(camf.read())
  cam_params_torch = torch.tensor(cam_params[0], dtype=torch.float32)   #  4cams
  cam_params_torch = torch.squeeze(cam_params_torch, dim=1)
  cam_params_dict = cam_params[1]

with open(annot_3d_path, "r") as annot3df:
  labels_3d = json.loads(annot3df.read())   # person, video, frame, joints, xyz
  test_labels_3d = labels_3d[-1]
  labels_3d = labels_3d[:-1]

with open(annot_2d_path, "r") as annot2df:
  labels_2d = json.loads(annot2df.read())   # person, camera, video, frame, joints, xy
  test_labels_2d = labels_2d[-1]
  labels_2d = labels_2d[:-1]


model = Pred3DPose().to(device)
model.train()
# criterion = nn.MSELoss()
criterion = nn.L1Loss()

if load_pretrained:
  checkpoint = torch.load("./checkpoint/epoch119.pt")
  model.load_state_dict(checkpoint['model_state_dict'])
  print("Loaded Checkpoint")

optimizer = optim.Adam(model.parameters(), lr=0.001)
if load_pretrained:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


for i in range(epoch):
  avg_loss_per_epoch = 0.0
  count = 0
  for person_idx, person in enumerate(labels_2d):
    
    for vid_idx, vid in enumerate(person[0]):
      mini_len =  math.inf
      inputs = []
      for cam_idx in range(len(person)):
        frames_in_one_video = person[cam_idx][vid_idx]
        mini_len = min(len(frames_in_one_video), mini_len)
        inputs.append(frames_in_one_video)

      for cam_input_idx, one_cam_input in enumerate(inputs):
        if len(one_cam_input) > mini_len:
          inputs[cam_input_idx] = one_cam_input[ : mini_len]
      
      gt_2d = torch.tensor(inputs, dtype=torch.float32)
      # gt_2d = torch.squeeze(gt_2d, dim=1)
      inputs = prepare_inputs(gt_2d, cam_params_torch)
      
      for idx in range(span, inputs.size(0) - span, batch_size):
        optimizer.zero_grad()

        small_batch_inputs = inputs[idx - span : idx + batch_size + span].to(device)

        if (small_batch_inputs.size(0) - span) < 1 + span:
          continue
        else:
          pred_3d = model(small_batch_inputs)
          # pred_2d = project_3d_to_2d(pred_3d, cam_params_dict)
          actual_n_frames = pred_3d.size(0)
          # batch_gt_2d = gt_2d[:, idx : idx + actual_n_frames].to(device)
          
          batch_gt_3d = labels_3d[person_idx][vid_idx][idx : idx + actual_n_frames]
          batch_gt_3d = torch.tensor(batch_gt_3d, dtype=torch.float32).to(device)
          batch_gt_3d *= 100.0
          # scale = align_3d_scales(pred_3d, batch_gt_3d)
          # pred_3d *= scale
          
          # loss = calculate_loss(pred_2d, pred_3d, batch_gt_2d, batch_gt_3d, loss_func=criterion)
          # loss = calculate_loss(pred_2d, batch_gt_2d, loss_func=criterion)

          loss = criterion(pred_3d, batch_gt_3d)
          avg_loss_per_epoch += loss
          count += 1
          
          loss.backward()
          optimizer.step()

    # print(f"average loss for epoch: {i}, people: {person_idx}, loss: {avg_loss_per_vid / count}")
  print(f"average loss for epoch: {i}, loss: {avg_loss_per_epoch / count}")
  
  torch.save({
      'epoch': i,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
    }, f"./checkpoint_L1/epoch{i}.pt")
