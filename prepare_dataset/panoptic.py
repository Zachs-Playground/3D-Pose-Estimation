import json
import glob
import math

import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import numpy as np
mp_pose = mp.solutions.pose

skip_n_frames = 15


# -------------------------------------------------------------------------
# Camera Preparation
# -------------------------------------------------------------------------

print("Preparing for camera parameters")

def combine_cam_params(R,T,K):
    for i in range(3):
        R[i].append(T[i][0])
    extri = np.array(R)

    mat = np.array(K) @ extri
    return mat.tolist()


def align_key_joint_idx(mediapipe_2d_landmark):
    mediapipe_joint = [0,11,12,13,14,15,16,23,24,25,26,27,28]
    joints = []
    for idx in mediapipe_joint:
        joints.append(mediapipe_2d_landmark[idx][0:2])
    return joints


main_folder = ["./panoptic-toolbox/171204_pose1/", "./panoptic-toolbox/171204_pose2/", "./panoptic-toolbox/171204_pose3/"]
cam_param_dict = {}


for pose_idx, pose_folder in enumerate(main_folder):
    cam_path = pose_folder + "camera_calibration.json"
    
    with open(cam_path, "r") as cjf:
        data = json.loads(cjf.read())
        data = data["cameras"]

    cam_count = 0  
    for cams in data:
        if cams["type"] == "hd":
            R = cams["R"]
            T = cams["t"]
            K = cams["K"]
            param = combine_cam_params(R,T,K)
            cam_name = "cam_" + str(cam_count)
            cam_param_dict[cam_name] = param
            cam_count += 1

    with open(pose_folder + "hd_cam.json", "w") as cwf:
        json.dump(cam_param_dict, cwf)



# -------------------------------------------------------------------------
# 3D Annotation Preparation
# -------------------------------------------------------------------------

print("Preparing for 3D annotations")

main_folder = ["./panoptic-toolbox/171204_pose1/", "./panoptic-toolbox/171204_pose2/", "./panoptic-toolbox/171204_pose3/"]
targeted_joints = [1,3,4,5,6,7,8,9,10,11,12,13,14]
for pose_idx, pose_folder in enumerate(main_folder):
    annot_folder = pose_folder + "annot3d/*.json"
    joints_in_one_vid = []
    for file_idx, annot_file in enumerate(glob.glob(annot_folder)):
        joints = []
        with open(annot_file, "r") as ajf:
            data = json.loads(ajf.read())
            if len(data["bodies"]) > 0:
                body = data["bodies"][0]["joints19"]                
                for i in range(0, len(body), 4):
                    xyzc = body[i:i+4]
                    if int(i / 4) in targeted_joints:
                        if xyzc[3] > 0.5:
                            xyzc = xyzc[0:3]
                            joints.append(xyzc)
                        else:
                            joints.append([0.0, 0.0, 0.0])
        if file_idx % 15 == 0:
            joints_in_one_vid.append(joints)
    print(len(joints_in_one_vid))
    with open(pose_folder + "annot3d.json", "w") as awf:
        json.dump(joints_in_one_vid, awf)



# -------------------------------------------------------------------------
# 2D Annotation Preparation
# -------------------------------------------------------------------------

print("Preparing for 2D annotations")

def align_key_joint_idx(landmarks):
    target_joints = [0,11,13,15,23,25,27,12,14,16,24,26,28]
    joints = []
    for i in target_joints:
        joints.append(landmarks[i])
    return joints

def normalized_to_pixel_coordinates(normalized_x,normalized_y, image_width, image_height):
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


main_folder = ["./panoptic-toolbox/171204_pose1/", "./panoptic-toolbox/171204_pose2/", "./panoptic-toolbox/171204_pose3/"]
targeted_joints = [1,3,4,5,6,7,8,9,10,11,12,13,14]
empty = [[0, 0] for _ in range(13)]

for pose_idx, pose_folder in enumerate(main_folder):
    vid_folder = pose_folder + "hdVideos/*.mp4"
    
    all_vids = []
    for vid_idx, vid in enumerate(glob.glob(vid_folder)):
        print("vid_idx: ", vid_idx)
        joints_in_one_vid = []
        with mp_pose.Pose(min_detection_confidence=0.7) as pose:
            cap = cv2.VideoCapture(vid)
            if (cap.isOpened()== False):
                print("Error opening video file")
            
            ret, frame = cap.read()
            count = 0
            while(ret):
                if count % 15 == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                    
                    results = pose.process(frame)
                    xyz = results.pose_landmarks
                    if (xyz != None):
                        mediapipe_2d_landmark = []
                        for joint in xyz.landmark:
                            x,y = normalized_to_pixel_coordinates(joint.x,joint.y,frame.shape[1],frame.shape[0])
                            mediapipe_2d_landmark.append([x,y])
                        realigned_2d_landmark = align_key_joint_idx(mediapipe_2d_landmark)
                        joints_in_one_vid.append(realigned_2d_landmark)
                    else:
                        joints_in_one_vid.append(empty)
                ret, frame = cap.read()
                count += 1

            cap.release()
            cv2.destroyAllWindows()
        all_vids.append(joints_in_one_vid)

    print(len(all_vids), len(all_vids[0]), len(all_vids[0][0]))
    with open(pose_folder + "annot2d.json", "w") as awf:
        json.dump(all_vids, awf)



# -------------------------------------------------------------------------
# Some 3D annot files has no information, so remove these empty ones
#  and remove the according ones for the 2D files
# -------------------------------------------------------------------------

print("Removing empty files")

main_folder = ["./panoptic-toolbox/171204_pose1/", "./panoptic-toolbox/171204_pose2/", "./panoptic-toolbox/171204_pose3/"]

for path in main_folder:
    json3d = path + "annot3d.json"
    json2d = path + "annot2d.json"

    with open(json3d, "r") as j3f:
        data3d = json.loads(j3f.read())
    
    with open(json2d, "r") as j2f:
        data2d = json.loads(j2f.read())
    

    print("3d length: ", len(data3d))
    print("2d length: ", len(data2d), len(data2d[0]))

    new_json3d = []
    new_json2d = [[] for _ in range(len(data2d))]

    for i in range(len(data3d)):
        if len(data3d[i]) != 0:
            new_json3d.append(data3d[i])
            for j in range(len(data2d)):
                if len(data2d[j]) >= len(data3d):
                    new_json2d[j].append(data2d[j][i])
                else:
                    print("#pose: ", path, ", #cam: ", j)
                    new_json2d[j].append([])

    with open(path + "adjusted3d.json", "w") as w3f:
        json.dump(new_json3d, w3f)     
    
    with open(path + "adjusted2d.json", "w") as w2f:
        json.dump(new_json2d, w2f)

print("Data preparatino is done")