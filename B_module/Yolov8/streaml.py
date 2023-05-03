# import streamlit as st
# import cv2
from ultralytics import YOLO
import supervision as sv

# img_file_buffer = st.camera_input("Take a picture")

# if img_file_buffer is not None:
#     # To read image file buffer as bytes:
#     bytes_data = img_file_buffer.getvalue()
#     # Check the type of bytes_data:
#     # Should output: <class 'bytes'>
#     st.write(type(bytes_data))
#     model = YOLO("../runs/detect/custom3/weights/best.pt")
#     
#         )
#     while True:
import streamlit as st
import cv2
import numpy as np

img_file_buffer = st.camera_input("visualize predictions")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    model = YOLO("C:/Users/igorv/OneDrive/Documents/GitHub/Detect_road_problems/runs/detect/custom3/weights/best.pt")
    box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1)
    cap = cv2.VideoCapture(0)
    # st.camera_input(result=model)
    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=False)[0]
            # переменная detections содержит полученную информацию, какие объекты обнаружены
        detections = sv.Detections.from_yolov8(result)
            # получаем наши объекты
        labels = [
            # 1 это имена классов, 2 это точность что это такой объект
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        # zone.trigger(detections=detections)
        # frame = zone_annotator.annotate(scene=frame)      
        st.camera_input("result=model", frame)
        # cv2.imshow("Camera", frame)
        # Прописывем клавишу по нажатию, её мы будем закрывать окно
        if (cv2.waitKey(30) == 27):
            #  это клавиша 27 - ESC
            break
    # st.write(cv2_cap.read())

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    # st.write(result)