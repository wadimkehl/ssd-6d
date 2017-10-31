"""Some utils for SSD."""

import numpy as np
import tensorflow as tf


def load_frozen_graph(filename):
    """ Loads the provided network as the new default graph """
    tf.reset_default_graph()
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


class NMSUtility():
    """Utility class to do non-max suppression for SSD6D

    """
    def __init__(self, max_output_size=100, iou_threshold=0.45):

        # Structures for non-max suppression
        self.nms_boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.nms_scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(self.nms_boxes, self.nms_scores,
                                                max_output_size=max_output_size,
                                                iou_threshold=iou_threshold)

    def run(self, session, boxes, confs):
        feed_dict = {self.nms_boxes: boxes, self.nms_scores: confs}
        idx = session.run(self.nms, feed_dict=feed_dict)
        return idx


def decode_boxes(locations, net_priors):
    """Convert bboxes from local predictions to shifted priors.

    # Arguments
        locations: Numpy array of predicted locations.
        net_priors: Numpy array of prior boxes from network.

    # Return
        decode_bbox: Shifted priors.

    """

    priors = net_priors[:, 0:4]
    vars = net_priors[:, 4:8]

    # Compute normalized prior positions in image
    prior_w = priors[:, 2] - priors[:, 0]
    prior_h = priors[:, 3] - priors[:, 1]
    prior_center_x = 0.5 * (priors[:, 2] + priors[:, 0])
    prior_center_y = 0.5 * (priors[:, 3] + priors[:, 1])

    # Use predicted locations as factors for decoding
    decoded_center_x = prior_center_x + locations[:, 0] * prior_w * vars[:, 0]
    decoded_center_y = prior_center_y + locations[:, 1] * prior_h * vars[:, 1]
    decoded_w = prior_w * np.exp(locations[:, 2] * vars[:, 2])
    decoded_h = prior_h * np.exp(locations[:, 3] * vars[:, 3])

    # Compute l,t,r,b representation
    xmin = decoded_center_x - 0.5 * decoded_w
    ymin = decoded_center_y - 0.5 * decoded_h
    xmax = decoded_center_x + 0.5 * decoded_w
    ymax = decoded_center_y + 0.5 * decoded_h

    decoded = np.concatenate(
        (xmin[:, None], ymin[:, None], xmax[:, None], ymax[:, None]), axis=-1)
    return np.clip(decoded, 0, 1)


def process_detection_output(session, priors, nms, models,
                             encoded_boxes, cla_probs, view_probs, inpl_probs,
                             confidence_threshold, views_to_parse, inplanes_to_parse):
    """Processes the tensors coming from the prediction layers into detections.

    # Arguments
        session: The tf.Session
        priors: The SSD priors in form (num_priors, 8)
        nms: An instance of NMSUtility
        models: List of model names to map label ID to proper names
        encoded_boxes: Encoded location regression
        cla_probs: Class probabilities of each prior
        view_probs: View ID probabilities of each prior
        inpl_probs: Inplane ID probabilities of each prior
        confidence_threshold: All detections below score will be ignored
        views_to_parse: How many views to parse per detection
        inplanes_to_parse: How many inplanes to parse per detection

    # Returns
        results: List of list of predictions for every picture.
                Each prediction has the form:
                [xmin, ymin, xmax, ymax, label, confidence,
                (view0, inplane0), ..., (viewN, inplaneM)]

    """
    results = []
    for i in range(encoded_boxes.shape[0]):
        decoded = decode_boxes(encoded_boxes[i], priors)
        detections = []

        for c in range(1, cla_probs.shape[2]):  # skip background (class 0)
            confs = cla_probs[i, :, c]
            mask = confs > confidence_threshold
            if not np.any(confs[mask]):  # No detections for that class
                continue

            # Subselect all predictions for that class that survived threshold
            boxes = decoded[mask]
            confs = confs[mask]
            views = view_probs[i, mask]
            inplanes = inpl_probs[i, mask]

            # TODO: tf.nms expects them in (y1, x1, y2, x2) but should be fine
            for idx in nms.run(session, boxes, confs):

                # Get 2D bbox and label with confidence
                pred = boxes[idx].tolist()
                pred.extend([models[c-1], confs[idx]])  # Extend with model name and confidence

                # Build 6D hypothesis pool
                views_sorted = np.argsort(views[idx])[::-1]
                inplanes_sorted = np.argsort(inplanes[idx])[::-1]

                views_parsed = views_sorted[:views_to_parse]
                inplanes_parsed = inplanes_sorted[:inplanes_to_parse]

                pool = [(v_, i_) for v_ in views_parsed for i_ in inplanes_parsed]
                pred.extend(pool)
                detections.append(pred)

        results.append(detections)

    return results