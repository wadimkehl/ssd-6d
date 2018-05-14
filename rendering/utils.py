


import math
import numpy as np
import cv2
from tqdm import tqdm
from scipy.linalg import expm, norm

from rendering.renderer import Renderer


def draw_detections_2D(image, detections):
    """Draws detections onto resized image with name and confidence

        Parameters
        ----------
        image: Numpy array, normalized to [0-1]
        detections: A list of detections for this image, coming from SSD.detect() in the form
            [l, t, r, b, name, confidence, .....]

    """
    out = np.copy(image)
    for det in detections:
        lt = (int(det[0] * image.shape[1]), int(det[1] * image.shape[0]))
        rb = (int(det[2] * image.shape[1]), int(det[3] * image.shape[0]))
        text = '{}: {:.2f}'.format(det[4], det[5])
        cv2.rectangle(out, lt, rb, (0., 1., 0.), 2)
        cv2.putText(out, text, lt, 0, 0.8, (0., 1., 0.), 2)
    return out


def draw_detections_3D(image, detections, cam, model_map):
    """Draws 6D detections onto resized image

        Parameters
        ----------
        image: Numpy array, normalized to [0-1]
        detections: A list of detections for this image, coming from SSD.detect() in the form
            [l, t, r, b, name, confidence, 6D_pose0, ..., 6D_poseN]
        cam: Intrinsics for rendering
        model_map: Mapping of model name to Model3D instance {'obj': model3D}

    """
    if not detections:
        return np.copy(image)

    ren = Renderer((image.shape[1], image.shape[0]), cam)
    ren.clear()
    out = np.copy(image)
    for det in detections:
        model = model_map[det[4]]
        for pose in det[6:]:
            ren.draw_model(model, pose)
            ren.draw_boundingbox(model, pose)
    col, dep = ren.finish()

    # Copy the rendering over into the scene
    mask = np.dstack((dep, dep, dep)) > 0
    out[mask] = col[mask]
    return out


def compute_rotation_from_vertex(vertex):
    """Compute rotation matrix from viewpoint vertex """
    up = [0, 0, 1]
    if vertex[0] == 0 and vertex[1] == 0 and vertex[2] != 0:
        up = [-1, 0, 0]
    rot = np.zeros((3, 3))
    rot[:, 2] = -vertex / norm(vertex)  # View direction towards origin
    rot[:, 0] = np.cross(rot[:, 2], up)
    rot[:, 0] /= norm(rot[:, 0])
    rot[:, 1] = np.cross(rot[:, 0], -rot[:, 2])
    return rot.T


def create_pose(vertex, scale=0, angle_deg=0):
    """Compute rotation matrix from viewpoint vertex and inplane rotation """
    rot = compute_rotation_from_vertex(vertex)
    transform = np.eye(4)
    rodriguez = np.asarray([0, 0, 1]) * (angle_deg * math.pi / 180.0)
    angle_axis = expm(np.cross(np.eye(3), rodriguez))
    transform[0:3, 0:3] = np.matmul(angle_axis, rot)
    transform[0:3, 3] = [0, 0, scale]
    return transform


def precompute_projections(views, inplanes, cam, model3D):
    """Precomputes the projection information needed for 6D pose construction

    # Arguments
        views: List of 3D viewpoint positions
        inplanes: List of inplane angles in degrees
        cam: Intrinsics to use for translation estimation
        model3D: Model3D instance

    # Returns
        data: a 3D list with precomputed entities with shape
            (views, inplanes, (4x4 pose matrix, 3) )
    """
    w, h = 640, 480
    ren = Renderer((w, h), cam)
    data = []
    if model3D.vertices is None:
        return data

    for v in tqdm(range(len(views))):
        data.append([])
        for i in inplanes:
            pose = create_pose(views[v], angle_deg=i)
            pose[:3, 3] = [0, 0, 0.5]  # zr = 0.5

            # Render object and extract tight 2D bbox and projected 2D centroid
            ren.clear()
            ren.draw_model(model3D, pose)
            box = np.argwhere(ren.finish()[1])  # Deduct bbox from depth rendering
            box = [box.min(0)[1], box.min(0)[0], box.max(0)[1] + 1, box.max(0)[0] + 1]
            centroid = pose[:3, 3]
            centroid_x = cam[0, 2] + centroid[0] * cam[0, 0] / centroid[2]
            centroid_y = cam[1, 2] + centroid[1] * cam[1, 1] / centroid[2]

            # Compute 2D centroid position in normalized, box-local reference frame
            box_w, box_h = (box[2] - box[0]), (box[3] - box[1])
            norm_centroid_x = (centroid_x - box[0]) / box_w
            norm_centroid_y = (centroid_y - box[1]) / box_h

            # Compute normalized diagonal box length
            lr = np.sqrt((box_w / w) ** 2 + (box_h / h) ** 2)
            data[-1].append((pose, [norm_centroid_x, norm_centroid_y, lr]))
    return data


def build_6D_poses(detections, model_map, cam, img_size=(640, 480)):
    """Processes the detections to build full 6D poses

    # Arguments
        detections: List of predictions for every image. Each prediction is:
                [xmin, ymin, xmax, ymax, label, confidence,
                (view0, inplane0), ..., (viewN, inplaneM)]
        model_map: Mapping of model name to Model3D instance {'obj': model3D}
        cam: Intrinsics to use for backprojection

    # Returns
        new_detections: List of list of 6D predictions for every picture.
                Each prediction has the form:
                [xmin, ymin, xmax, ymax, label, confidence,
                (pose00), ..., (poseNM)] where poseXX is a 4x4 matrix

    """

    new_detections = []
    for image_dets in detections:
        new_image_dets = []
        for det in image_dets:
            new_det = det[:6]  # Copy over 2D bbox, label and confidence
            box_w, box_h = det[2] - det[0], det[3] - det[1]
            ls = np.sqrt(box_w ** 2 + box_h ** 2)

            projected = model_map[det[4]].projections
            for v, i in det[6:]:  # Process each View/Inplane pair

                if not projected:  # No pre-projections available for this model, skip...
                    new_det.append(np.eye(4))
                    continue

                pose = projected[v][i][0]
                norm_centroid_x, norm_centroid_y, lr = projected[v][i][1]
                pose[2, 3] = 0.5 * lr / ls  # Compute depth from projective ratio

                # Compute the new 2D centroid in pixel space
                new_centroid_x = (det[0] + norm_centroid_x * box_w) * img_size[0]
                new_centroid_y = (det[1] + norm_centroid_y * box_h) * img_size[1]

                # Backproject into 3D metric space
                pose[0, 3] = pose[2, 3] * (new_centroid_x - cam[0, 2]) / cam[0, 0]
                pose[1, 3] = pose[2, 3] * (new_centroid_y - cam[1, 2]) / cam[1, 1]
                new_det.append(pose)
            new_image_dets.append(new_det)
        new_detections.append(new_image_dets)

    return new_detections


def verify_6D_poses(detections, model_map, cam, image):
    """For one image, select for each detection the best pose from the 6D pool

    # Arguments
        detections: List of predictions for one image. Each prediction is:
                [xmin, ymin, xmax, ymax, label, confidence,
                (pose00), ..., (poseNM)] where poseXX is a 4x4 matrix
        model_map: Mapping of model name to Model3D instance {'obj': model3D}
        cam: Intrinsics to use for backprojection
        image: The scene color image

    # Returns
        filtered: List of predictions for one image.
                Each prediction has the form:
                [xmin, ymin, xmax, ymax, label, confidence, pose] where pose is a 4x4 matrix

    """

    def compute_grads_and_mags(color):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        grads = np.dstack((cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5),
                           cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)))
        mags = np.sqrt(np.sum(grads**2, axis=2)) + 0.001  # To avoid div/0
        grads /= np.dstack((mags, mags))
        mask = mags < 5
        mags[mask] = 0
        grads[np.dstack((mask, mask))] = 0
        return grads, mags

    scene_grads, scene_mags = compute_grads_and_mags(image)
    scene_grads = np.reshape(scene_grads, (-1, 2))

    ren = Renderer((image.shape[1], image.shape[0]), cam)
    filtered = []
    for det in detections:
        model = model_map[det[4]]
        scores = []
        for pose in det[6:]:
            ren.clear()
            ren.draw_model(model, pose)
            ren_grads, ren_mags = compute_grads_and_mags(ren.finish()[0])
            ren_grads = np.reshape(ren_grads, (-1, 2))
            dot = np.sum(np.abs(ren_grads[:, 0]*scene_grads[:, 0] + ren_grads[:, 1]*scene_grads[:, 1]))
            sum = np.sum(ren_mags>0)
            scores.append(dot / sum)
        new_det = det[:6]
        new_det.append(det[6 + np.argmax(np.asarray(scores))])  # Put best pose first
        filtered.append(new_det)

    return filtered
