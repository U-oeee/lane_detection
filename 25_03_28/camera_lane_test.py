import cv2
import time
import logging
import configparser
import numpy as np  # ✅ ROI 마스킹에 필요

from LaneKeeping.lanekeeping import LaneKeeping
from LaneDetection.detect import LaneDetection

# ✅ ROI 마스크 함수 정의
def apply_roi_mask(image):
    height, width = image.shape[:2]

    # 관심영역 (검정색 도로 영역) 설정 — 필요시 좌표 조절
    roi_vertices = np.array([[
        (0, height),
        (0, int(height * 0.4)),
        (width, int(height * 0.4)),
        (width, height)
    ]], dtype=np.int32)

    # 마스크 생성
    mask = np.zeros_like(image)
    if len(image.shape) == 3:
        ignore_mask_color = (255,) * image.shape[2]
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, roi_vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")

    cap = cv2.VideoCapture(1)  # ✅ 필요시 1로 변경

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        exit()

    log = logging.getLogger('Root logger')

    ret, src = cap.read()
    if not ret:
        print("웹캠에서 프레임을 읽을 수 없습니다.")
        cap.release()
        exit()

    camera = "455"
    lk = LaneKeeping(src.shape[1], src.shape[0], log, camera)
    ld = LaneDetection(src.shape[1], src.shape[0], camera, lk)

    print("Keys:\n- ' ' = Pause/Unpause\n- 'q' = Quit\n- 's' = Start saving frames\n- 'e' = Stop saving frames")

    start_saving_frames = False
    time_sum = 0
    frames_used = 0

    while True:
        ret, src = cap.read()
        if not ret:
            print("프레임을 읽는 중 오류가 발생했습니다.")
            break

        # ✅ 관심 영역 마스크 적용
        src = apply_roi_mask(src)

        frames_used += 1
        start = time.time()
        ld_frame = src.copy()
        results = ld.lanes_detection(ld_frame)
        angle, out_frame = lk.lane_keeping(results)
	# ⬇️ 이동 방향 화살표 추가
        desired_lane = lk.desired_lane(results["left_coef"], results["right_coef"])
        lk.visualize_direction_arrow(out_frame, desired_lane)
        cv2.imshow('Lane Keeping', out_frame)
        end = time.time()

        time_sum += end - start

        if start_saving_frames:
            cv2.imwrite(f"./frames/{frames_used}.jpg", out_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord(' '):
            while True:
                key2 = cv2.waitKey(100)
                if key2 == ord(' '):
                    break
                elif key2 == ord('q'):
                    exit()
        elif key == ord('s'):
            start_saving_frames = True
            print("Start saving frames...")
        elif key == ord('e'):
            start_saving_frames = False
            print("Stop saving frames...")

    cap.release()
    cv2.destroyAllWindows()

    if frames_used > 0:
        print(f"\n평균 처리 시간: {time_sum / frames_used:.4f}초/frame\n")

