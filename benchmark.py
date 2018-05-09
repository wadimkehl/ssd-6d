"""
Simple script to run a forward pass with SSD-6D on a SIXD dataset with a trained model.
Usage:
    run.py [options]
    run.py (-h | --help)

Options:
    -d, --dataset=<string>   Path to SIXD dataset [default: /Users/kehl/Desktop/sixd/hinterstoisser]
    -s, --sequence=<int>     Number of the sequence [default: 1]
    -n, --network=<string>   Path to trained network [default: /Users/kehl/Dropbox/iccv-models/hinterstoisser_obj_01.pb]
    -t, --threshold=<float>  Threshold for the detection confidence [default: 0.5]
    -v, --views=<int>        Views to parse for 6D pose pooling [default: 3]
    -i, --inplanes=<int>     In-plane rotations to parse for 6D pose pooling [default: 3]
    -h --help                Show this message and exit

"""

import cv2
from docopt import docopt
import numpy as np
import tensorflow as tf

from ssd.ssd_utils import load_frozen_graph, NMSUtility, process_detection_output
from rendering.utils import precompute_projections, build_6D_poses, verify_6D_poses
from rendering.utils import draw_detections_2D, draw_detections_3D
from utils.sixd import load_sixd
from utils.metrics import *

args = docopt(__doc__)
sixd_base = args["--dataset"]
sequence = int(args["--sequence"])
network = args["--network"]
threshold = float(args["--threshold"])
views_to_parse = int(args["--views"])
inplanes_to_parse = int(args["--inplanes"])

# run evaluation on all frames
nr_frames = -1

# Build detection and NMS networks
load_frozen_graph(network)
nms = NMSUtility(max_output_size=100, iou_threshold=0.45)

adds = []
trans_errors_norm = []
trans_errors_single = []
rot_errors = []

with tf.Session() as sess:
    # Read out constant information
    models = sess.run(sess.graph.get_tensor_by_name('models:0'))
    models = [m.decode('utf-8') for m in models]  # Strings are byte-encoded
    views = sess.run(sess.graph.get_tensor_by_name('views:0'))
    inplanes = sess.run(sess.graph.get_tensor_by_name('inplanes:0'))
    priors = sess.run(sess.graph.get_tensor_by_name('priors:0'))
    variances = sess.run(sess.graph.get_tensor_by_name('variances:0'))
    priors = np.concatenate((priors, variances), axis=1)

    # Get tensor handles
    tensor_in = sess.graph.get_tensor_by_name('input:0')
    tensor_loc = sess.graph.get_tensor_by_name('locations:0')
    tensor_cla = sess.graph.get_tensor_by_name('class_probs:0')
    tensor_view = sess.graph.get_tensor_by_name('view_probs:0')
    tensor_inpl = sess.graph.get_tensor_by_name('inplane_probs:0')

    if len(models) == 1:  # If single-object network
        models = ['obj_{:02d}'.format(sequence)]  # Overwrite model name

    bench = load_sixd(sixd_base, nr_frames=nr_frames, seq=sequence)

    input_shape = (1, 299, 299, 3)
    print('Models:', models)
    print('Views:', len(views))
    print('Inplanes:', len(inplanes))
    print('Priors:', priors.shape)

    print('Precomputing projections for each used model...')
    model_map = bench.models  # Mapping from name to model3D instance
    for model_name in models:
        m = model_map[model_name]
        m.projections = precompute_projections(views, inplanes, bench.cam, m)

    # Process each frame separately
    for f in bench.frames:
        image = cv2.resize(f.color, (input_shape[2], input_shape[1]))
        image = image[np.newaxis, :]  # Bring image into 4D batch shape

        # Get the raw network output
        run = [tensor_loc, tensor_cla, tensor_view, tensor_inpl]
        encoded_boxes, cla_probs, view_probs, inpl_probs = sess.run(run, {tensor_in: image})

        # Extend rank because of buggy TF 1.0 softmax
        cla_probs = cla_probs[np.newaxis, :]
        view_probs = view_probs[np.newaxis, :]
        inpl_probs = inpl_probs[np.newaxis, :]

        # Read out the detections in proper format for us
        dets_2d = process_detection_output(sess, priors, nms, models,
                                           encoded_boxes, cla_probs, view_probs, inpl_probs,
                                           threshold, views_to_parse, inplanes_to_parse)

        # Convert the 2D detections with their view/inplane IDs into 6D poses
        dets_6d = build_6D_poses(dets_2d, model_map, bench.cam)[0]

        # (NOT INCLUDED HERE) Run pose refinement for each pose in pool

        # Pick for each detection the best pose from the 6D pose pool
        final = verify_6D_poses(dets_6d, model_map, bench.cam, f.color)

        cv2.imshow('Final poses', draw_detections_3D(f.color, final, bench.cam, model_map))
        cv2.waitKey(10)

        for gt_obj, gt_pose, gt_bbox in f.gt:
            # to bring both bboxes to the same format
            gt_bbox = [gt_bbox[0] / 640., gt_bbox[1] / 480.,
                       (gt_bbox[0] + gt_bbox[2]) / 640., (gt_bbox[1] + gt_bbox[3]) / 480.]
            gt_obj = 'obj_{:02d}'.format(gt_obj)

            for fin in final:
                est_bbox = fin[:4]
                est_pose = fin[6]
                est_obj = fin[4]

                if gt_obj == est_obj:
                    if iou(gt_bbox, est_bbox) >= 0.5:
                        model = model_map[est_obj]
                        adds.append(add_err(gt_pose, est_pose, model) < (0.1 * model.diameter))
                        trans_errors = trans_error(gt_pose, est_pose)
                        trans_errors_norm.append(trans_errors[0])
                        trans_errors_single.append(trans_errors[1])
                        rot_errors.append(rot_error(gt_pose, est_pose))

mean_add = np.mean(adds)
mean_trans_error_norm = np.mean(trans_errors_norm)
mean_trans_error_single = np.mean(trans_errors_single, axis=0)
mean_rot_error = np.mean(rot_errors)


print("Raw Unrefined Results:")
print("\tMean ADD: {:.3f}".format(mean_add))
print("\tMean Trans Error Norm: {:.3f}".format(mean_trans_error_norm))
print("\tMean Trans Errors: X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(mean_trans_error_single[0],
                                                                    mean_trans_error_single[1],
                                                                    mean_trans_error_single[2]))
print("\tMean Rotation Error: {:.3f}".format(mean_rot_error))
