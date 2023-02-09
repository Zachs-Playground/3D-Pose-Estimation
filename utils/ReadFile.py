import sys
import json
import cv2
import numpy as np

class ReadFile:
  def __init__(self, camera_folder_path, annot_folder_path):
    self.camera_folder_path = camera_folder_path
    self.annot_folder_path = annot_folder_path
  
  def _read_camera_file(self, camera_file_path, camera_idx):
    fs = cv2.FileStorage(camera_file_path, cv2.FILE_STORAGE_READ)
    r_mat = fs.getNode(camera_idx)
    return np.array(r_mat.mat())
  
  def get_camera_params(self, dataset="ZJM", 
                        extri_file="extri.yml",
                        intri_file="intri.yml",
                        cam_list=[["Rot_01", "T_01", "K_01"],
                                 ["Rot_06", "T_06", "K_06"],
                                 ["Rot_12", "T_12", "K_12"],
                                 ["Rot_18", "T_18", "K_18"]]):
    cams_params = dict()
    extri_camera_file_path = self.camera_folder_path + "/" + extri_file
    intri_camera_file_path = self.camera_folder_path + "/" + intri_file
    if dataset == "ZJM":
      for idx, item in enumerate(cam_list):
        cam_name = f"cam_{idx + 1}"
        cams_params[cam_name] = {
          item[0]: self._read_camera_file(extri_camera_file_path, item[0]),
          item[1]: self._read_camera_file(extri_camera_file_path, item[1]),
          item[2]: self._read_camera_file(intri_camera_file_path, item[2])
        }
      return cams_params
    else:
      print("")
      print(f"{dataset} is not implemented in this class")
      print("")
      sys.exit(0)
  
  def get_annots(self, file_name):
    file_path = self.annot_folder_path + "/" + file_name
    with open(file_path, "r") as f:
      data = json.loads(f.read())
    return data