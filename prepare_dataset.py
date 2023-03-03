# https://colab.research.google.com/drive/1uCuA6We9T5r0WljspEHWPHXCT_2bMKUy#scrollTo=BAivyQ_xOtFp


import json
import glob

import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import numpy as np
mp_pose = mp.solutions.pose

skip_n_frames = 15

def combine_cam_params(R,T,c,f):
    for i in range(3):
        R[i].append(T[0][i])
    extri = np.array(R)

    intri = np.eye(3)
    intri[0][0] = c[0]
    intri[0][2] = f[0]
    intri[1][1] = c[0]
    intri[1][2] = c[1]

    mat = intri @ extri
    return mat.tolist()

def get_intri(c, f):
    intri = np.eye(3)
    intri[0][0] = c[0]
    intri[0][2] = f[0]
    intri[1][1] = c[0]
    intri[1][2] = c[1]

    return intri.tolist()

def align_key_joint_idx(mediapipe_2d_landmark):
    mediapipe_joint = [0,11,12,13,14,15,16,23,24,25,26,27,28]
    joints = []
    for idx in mediapipe_joint:
        joints.append(mediapipe_2d_landmark[idx][0:2])
    return joints


cam_folder = "./data/Fitness3D/train/s03/camera_parameters/*"
cam_param_list = [[],[],[],[]]      # cam_1, cam2, cam3, cam4
cam_param_dict = {}

for cam_idx, cam_sub_folder in enumerate(glob.glob(cam_folder)):
    cam_json_file = cam_sub_folder + "/band_pull_apart.json"
    print(cam_json_file)
    cam_name = "cam_" + str(cam_idx)
    with open(cam_json_file, "r") as cjf:
        data = json.loads(cjf.read())
        R = data["extrinsics"]["R"]
        T = data["extrinsics"]["T"]
        c = data["intrinsics_wo_distortion"]["c"]
        f = data["intrinsics_wo_distortion"]["f"]
        param = combine_cam_params(R,T,c,f)
        cam_param_list[cam_idx].append(param)
        intri = get_intri(c, f)
        cam_param_dict[cam_name] = [R,T,intri]
with open("./data/processed_data/cam.json", "w") as cwf:
    json.dump([cam_param_list, cam_param_dict], cwf)


human36_joint = [9,11,14,12,15,13,16,1,4,2,5,3,6]


data_dir = "./data/Fitness3D/train/*"
annot_3d = [[[] for _ in range(47)] for i in range(8)]      # person, video, frame
annot_2d = [[[[] for _ in range(47)] for j in range(4)] for i in range(8)]    # person, camera, video, frame
empty = [[0, 0] for _ in range(13)]
for data_folder_idx, data_folder in enumerate(glob.glob(data_dir)):
    anno_folder = data_folder + "/joints3d_25/*.json"
    video_folder = data_folder + "/videos/*"
    
    for annot_file_idx, anno_file in enumerate(glob.glob(anno_folder)):
        with open(anno_file, "r") as arf:
            annot_data = json.loads(arf.read())
            annot_3d_one_video = annot_data["joints3d_25"]
            hold_frame_annot = []
            for frame_idx, annot_in_one_frame in enumerate(annot_3d_one_video):
                if frame_idx % skip_n_frames == 0:
                    align_to_mediapipe_joints = []
                    for joint_idx in human36_joint:
                        align_to_mediapipe_joints.append(annot_in_one_frame[joint_idx])
                    annot_3d[data_folder_idx][annot_file_idx].append(align_to_mediapipe_joints)

        print(f"Finished 3D annotation files: {annot_file_idx}")
    
    for vid_subfolder_idx, vid_subfolder in enumerate(glob.glob(video_folder)):
        vid_path = vid_subfolder + "/*.mp4"
        for vid_file_idx, vid_file in enumerate(glob.glob(vid_path)):
            with mp_pose.Pose(min_detection_confidence=0.7) as pose:
                cap = cv2.VideoCapture(vid_file)
                if (cap.isOpened()== False):
                    print("Error opening video file")
                
                ret, frame = cap.read()
                count = 0
                while(ret):
                    if count % skip_n_frames == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                    
                        results = pose.process(frame)
                        xyz = results.pose_landmarks
                        if (xyz != None):
                            mediapipe_2d_landmark = []
                            for joint in xyz.landmark:
                                x,y = _normalized_to_pixel_coordinates(joint.x,joint.y,frame.shape[1],frame.shape[0])
                                mediapipe_2d_landmark.append([x,y])
                            realigned_2d_landmark = align_key_joint_idx(mediapipe_2d_landmark)
                            annot_2d[data_folder_idx][vid_subfolder_idx][vid_file_idx].append(realigned_2d_landmark)
                        else:
                            annot_2d[data_folder_idx][vid_subfolder_idx][vid_file_idx].append(empty)
                    ret, frame = cap.read()
                    count += 1

                cap.release()
                cv2.destroyAllWindows()
            print(f"Finished 2D annotation files: {data_folder_idx}-{vid_subfolder_idx}-{vid_file_idx}")

with open("./data/processed_data/annot_3d.json", "w") as a3wf:
    json.dump(annot_3d, a3wf)

with open("./data/processed_data/annot_2d.json", "w") as a2wf:
    json.dump(annot_2d, a2wf)

