# Импортируем библиотеку компьютерного зрения
import cv2
# импортируем библиотеку позволяющую передавать аргументы через терминал
import argparse
#  импортируем YOLO класс из библиотеки ultralytics
from ultralytics import YOLO
# библиотеку для взаимодействия с камерой
import supervision as sv
# библиотеку для работы с массивами
import numpy as np
# Импортируем библиотеку для работы с DataFrame, будем сохранять в него распознанные объекты
import pandas as pd

class Camera_Detect():
    def __init__(self):
        # прописываем зону полигона
        self.ZONE_POLYGON = np.array([
            [0, 0],
            [0.5, 0],
            [0.5, 1],
            [0, 1]
        ])
        self.args = None
        self.frame_width = 1280
        self.frame_height = 720


    # создаем функцию, позволяющую получать аргументы из терминала
    def parse_arguments(self) -> argparse.Namespace:
        self.parser = argparse.ArgumentParser(description="YOLOv8 live")
        self.parser.add_argument(
            "--webcam-resolution", 
            default=[1280, 720], 
            nargs=2, 
            type=int
        )
        # парсим аргументы
        self.args = self.parser.parse_args()
        return self.args

    #  функция, реализуующая работу с камерой
    def detection(self):
        # получаем аргументы
        self.args = self.parse_arguments()
        # задаем разрешение на основе аргументов
        self.frame_width, self.frame_height = self.args.webcam_resolution
        # получаем видео в режиме реального времени
        self.cap = cv2.VideoCapture(0)
        # ставим ширину окна
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        # высота окна
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        # загружаем нашу обученную модель
        self.model = YOLO("../runs/detect/custom3/weights/best.pt")
        # model = YOLO("B_module/Yolov8/yolov8n.pt")
        # задаем параметры бокса объекта
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

        # zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
        # zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
        # zone_annotator = sv.PolygonZoneAnnotator(
        #     zone=zone, 
        #     color=sv.Color.red(),
        #     thickness=2,
        #     text_thickness=4,
        #     text_scale=2
        # )

        while True:
            # читаем видео с камеры
            self.ret, self.frame = self.cap.read()
            # в переменную result передаем нашу модель и размер окна
            self.result = self.model(self.frame, agnostic_nms=False)[0]
            # переменная detections содержит полученную информацию, какие объекты обнаружены
            detections = sv.Detections.from_yolov8(self.result)
            # получаем наши объекты
            labels = [
                # 1 это имена классов, 2 это точность что это такой объект
                f"{self.model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]
            self.frame = self.box_annotator.annotate(
                scene=self.frame, 
                detections=detections, 
                labels=labels
            )

            # zone.trigger(detections=detections)
            # frame = zone_annotator.annotate(scene=frame)      
            
            cv2.imshow("Camera", self.frame)
            # Прописывем клавишу по нажатию, её мы будем закрывать окно
            if (cv2.waitKey(30) == 27):
                #  это клавиша 27 - ESC
                break


if __name__ == "__main__":
    Camera_Detect().detection()