# 3D Pose Estimation
 
This project uses multi-view 2D cameras to predict human key joints in a 3D space. The procedure steps are:
<br />
1. pick five keys frames t-2, t-1, t, t+1, t+2 (15 frames between each picked frames)
2. use triangulation technique to generate rays through the key joints
3. use MLP to increase the dimentionality of the joints
4. use MLP to extract spatial and temporal features to make the prediction

### Model Architecture
<br />
<img src="./imgs/model.jpg" />
