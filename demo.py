import cv2
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


MODEL_NAME = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"


def get_model(model_name=MODEL_NAME):
    model_path = f"Tensorflow/testing/{model_name}/saved_model"
    return tf.saved_model.load(model_path)


def create_category_index(path="Tensorflow/testing/label_map.pbtxt"):
    return label_map_util.create_category_index_from_labelmap(path, use_display_name=True)


def main():

    test_model = get_model()
    category_index = create_category_index()

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        # get detections
        detections = test_model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int32)

        image_np_with_detections = image_np.copy()
        image_np_with_detections_uint8 = image_np_with_detections.astype(np.uint8)

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections_uint8,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=10,
                min_score_thresh=0.5,
                agnostic_mode=False)

        cv2.imshow("Detecting road signs..", image_np_with_detections_uint8)

        if cv2.waitKey(10) & 0xFF==ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    

if __name__ == "__main__":
    main()
