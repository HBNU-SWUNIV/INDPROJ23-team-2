import pyrealsense2 as rs
import numpy as np
import cv2


class RGBD:
    def __init__(self, cam_id):
        self.cam_id = str(cam_id)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.cam_id)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)

    def start(self):
        self.pipeline.start(self.config)
        device = self.pipeline.get_active_profile().get_device()
        depth_sensor = device.first_depth_sensor()
        # depth_sensor.set_option(rs.option.visual_preset, 3)  # High Density
        depth_sensor.set_option(rs.option.visual_preset, 1)  # High Quality

    def get_aligned_frames(self): # depth, frame, color의 위치가 안 맞을 수 있기 때문에 align 시켜준다
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        return color_frame, aligned_depth_frame

    # 특정 픽셀 깊이 
    def get_depth_at_pixel(self, depth_frame, x, y):
        depth = depth_frame.get_distance(x, y) * 1000 
        # if 175 < depth < 400 :
        #     return depth 
            
    # 특정 영역 깊이 
    # def get_depth_in_rect(self, depth_frame, rect_start, rect_end):
    #     # 지정된 직사각형 내의 깊이 데이터 추출
    #     depth = np.asanyarray(depth_frame.get_data())
    #     rect_depth = depth[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]

    #     # 깊이 데이터가 유효한 경우에만 처리
    #     if rect_depth.size > 0:
    #         rect_depth = rect_depth[rect_depth != 0]  # 0 값을 제외
    #         if rect_depth.size > 0:
    #             return np.mean(rect_depth)  # 평균 깊이 반환
    #     return None
    
    # 특정 영역 깊이     
    def get_depth_in_rect(self, depth_frame):
    # 지정된 직사각형 내의 깊이 데이터 추출
        depth = np.asanyarray(depth_frame.get_data())
        # rect_depth = depth[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]

        if depth.size > 0:
            depth = depth[depth != 0]  # 0 값을 제외
            if depth.size > 0:
                return np.mean(depth)  # 평균 깊이 반환
        return None
        # 깊이 데이터가 유효한 경우에만 처리
        # if rect_depth.size > 0:
        #     rect_depth = rect_depth[rect_depth != 0]  # 0 값을 제외
        #     if rect_depth.size > 0:
        #         return np.mean(rect_depth)  # 평균 깊이 반환
        # return None
    
    def get_depth_colormap(self, depth_frame, scaling_factor=0.05):
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=scaling_factor), cv2.COLORMAP_JET)
        return depth_colormap

    def stop(self):
        self.pipeline.stop()

def main():
    cam_id = '332522075575'
    single_realsense = RGBD(cam_id)
    single_realsense.start()

    try:
        while True:
            color_frame, depth_frame = single_realsense.get_aligned_frames()
            color_image = np.asanyarray(color_frame.get_data())
            # blue_rect_start = (160, 120)
            # blue_rect_end = (480, 360)

            # 깊이 프레임을 컬러맵으로 변환
            depth_colormap = single_realsense.get_depth_colormap(depth_frame)

            # 파란색 사각형을 RGB 이미지에 그립니다.
            # cv2.rectangle(color_image, blue_rect_start, blue_rect_end, (0, 0, 255), 2)

            # 동일한 파란색 사각형을 깊이 컬러맵 이미지에도 그립니다.
            # cv2.rectangle(depth_colormap, blue_rect_start, blue_rect_end, (0, 0, 255), 2)

            # 깊이 정보 추출
            red_depth = single_realsense.get_depth_in_rect(depth_frame)
            # red_depth = single_realsense.get_depth_in_rect(depth_frame, blue_rect_start, blue_rect_end)
            red_depth = red_depth if red_depth and red_depth < 800 else None  # 2000mm 이상 무시

            opt = parse_opt()
            check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
            run(**vars(opt))


            
            
            
            # RGB 이미지 표시
            cv2.imshow('RealSense Camera - Color', color_image)

            # 깊이 컬러맵 이미지 표시
            cv2.imshow('RealSense Camera - Depth', depth_colormap)

            # 깊이 정보 출력
            print(f"Depth: {red_depth}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        single_realsense.stop()
        cv2.destroyAllWindows()

        
if __name__ == "__main__":
    main()