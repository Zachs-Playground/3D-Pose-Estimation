""" Use KMeans To Find N Closest Points At A Joint Location """

import numpy as np
from sklearn.cluster import KMeans

class FindPoints:
  def __init__(self, all_points):
    self.all_points = all_points
  
  def _get_centroid(self, points, method):
    if method == "mean":
      centroid = np.mean(points, axis=0)
      return centroid
    elif method == "kmeans":
      Kmean = KMeans(n_clusters=1)
      Kmean.fit(points)
      centroid = Kmean.cluster_centers_
      return centroid[0]
    else:
      raise RuntimeError(f"The method, {method}, is not implemented. Please use 'mean' or 'kmeans'") 
  
  def _find_closest_points(self, centroid_pos, points_pos, select_n_points):
    dist = [np.sum(np.absolute(centroid_pos - point_pos)) for point_pos in points_pos]
    dist_sort = np.argsort(dist)            # sort the list and return indices
    dist_sort = dist_sort[:select_n_points]       # selected the first n indices since it is in ascending order
    selected_points = [points_pos[idx] for idx in dist_sort]
    return np.array(selected_points)
  
  def __call__(self, select_n_points=15, method="kmeans"):
    result = np.zeros((len(self.all_points), len(self.all_points[0]), select_n_points, 3), dtype=np.float32)
    for frame_idx, one_frame_points in enumerate(self.all_points):
      for joint_idx, one_joint_points in enumerate(one_frame_points):
        points = np.reshape(np.stack(one_joint_points), (-1, 3))
        centroid = self._get_centroid(points, method)
        selected_points = self._find_closest_points(centroid, points, select_n_points)
        shape = selected_points.shape
        # print("frame: ", frame_idx, " joint: ", joint_idx)
        result[frame_idx][joint_idx][:shape[0]] = selected_points

    return result