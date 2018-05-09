import numpy as np
from pyquaternion import Quaternion

def add_err(gt_pose, est_pose, model):
    def transform_points(points_3d, mat):
        rot = np.matmul(mat[:3, :3], points_3d.transpose())
        return rot.transpose() + mat[:3, 3]

    v_A = transform_points(model.vertices, gt_pose)
    v_B = transform_points(model.vertices, est_pose)

    return np.mean(np.linalg.norm(v_A - v_B, axis=1))


def rot_error(gt_pose, est_pose):
    def matrix2quaternion(m):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])

    gt_quat = Quaternion(matrix2quaternion(gt_pose[:3, :3]))
    est_quat = Quaternion(matrix2quaternion(est_pose[:3, :3]))

    return np.abs((gt_quat * est_quat.inverse).degrees)


def trans_error(gt_pose, est_pose):
    trans_err_norm = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    trans_err_single = np.abs(gt_pose[:3, 3] - est_pose[:3, 3])

    return trans_err_norm, trans_err_single

def iou(gt_box, est_box):
    xA = max(gt_box[0], est_box[0])
    yA = max(gt_box[1], est_box[1])
    xB = min(gt_box[2], est_box[2])
    yB = min(gt_box[3], est_box[3])

    if xB <= xA or yB <= yA:
        return 0.

    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    boxBArea = (est_box[2] - est_box[0]) * (est_box[3] - est_box[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    return interArea / float(boxAArea + boxBArea - interArea)