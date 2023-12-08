import os
import sys
from pathlib import Path
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import check_requirements, cv2
from distance import RGBD


def main():
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    
    cam_id = '332522075575' # 카메라 주소
    single_realsense = RGBD(cam_id)
    single_realsense.start()

    model = torch.hub.load('ultralytics/yolov5', 'custom', path = "best.pt")

    try:
        while True:
            color_frame, aligned_depth_frame = single_realsense.get_aligned_frames()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            # YOLOv5 모델에 이미지 프레임을 입력
            results = model(color_image)
            
            # 탐지된 객체들에 대한 처리
            for i, (xmin, ymin, xmax, ymax, confidence, class_id) in enumerate(results.xyxy[0]):
                # 탐지된 객체의 경계 상자 중심 좌표
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)

                # 깊이 프레임에서 객체 중심의 깊이 추출
                object_depth = depth_image[center_y, center_x]
            
                if object_depth <800:
                # 탐지된 객체와 거리 정보 출력
                    if results.names[int(class_id)] == 'box':
                        print(f"Detected {results.names[int(class_id)]} at depth: {object_depth}mm")
                        # 제어정보
                        
                        
                    elif results.names[int(class_id)] == 'cart':
                        print(f"Detected {results.names[int(class_id)]} at depth: {object_depth}mm")
                        # 제어정보
                        
                        
                    elif results.names[int(class_id)] == 'hose':
                        print(f"Detected {results.names[int(class_id)]} at depth: {object_depth}mm")
                        # 제어정보
                        
                        
                    elif results.names[int(class_id)] == 'cylinder':
                        print(f"Detected {results.names[int(class_id)]} at depth: {object_depth}mm")
                        # 제어정보
                        
                        

            # 결과 시각화
            cv2.imshow('YOLOv5s Detection', np.squeeze(results.render()))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        single_realsense.stop()
        cv2.destroyAllWindows()
    # run(**vars(opt))

    cv2.destroyAllWindows()
    single_realsense.release()

if __name__ == '__main__':
    main()