"""Construct T_C_T: meters; optical camera X right, Y down, Z forward."""
import cv2
import numpy as np


def target_to_camera(pose):
    transform = np.eye(4)
    transform[:3,:3] = cv2.Rodrigues(np.asarray(pose['rvec'],dtype=float))[0]
    transform[:3,3] = pose['tvec_m']
    return transform
