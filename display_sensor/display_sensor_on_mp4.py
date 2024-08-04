import cv2
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
ACC_SR = 1000
START= (350,665)

END_X_X = 300
END_X_Y = 200

END_Y_X = 350
END_Y_Y = 150

END_Z_X = 300
END_Z_Y = 150
MAX = 15
def process_frame(frame, x, y, z):
    c = x*100
    end_x = START[0]  + int(c)
    end_y = START[1] -100 - int(np.sqrt(100**2 -c))
    frame = cv2.arrowedLine(frame,START,(end_x,end_y),color=(255,0,0),thickness=9) 
    return frame

root_path = '/dsi/gannot-lab1/datasets/MMSCG_meta/MMCSG/'
root_path = Path(root_path)
sub = 'dev'

name = '187253264474937_0001_4225_20225'
video_path = root_path/(f'video/{sub}/{name}.mp4')
acc_path = root_path/(f'gyroscope/{sub}/{name}.npy')

if video_path.is_file() and acc_path.is_file():
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_frame_delay = 1.0 / fps
    acc_frame_delay = int(video_frame_delay*ACC_SR)
    acc = np.load(acc_path)
    if not cap.isOpened():
        print(f"Error: Couldn't open video file {video_path}")

    # Draw rectangle
    i = 0
    while True:    
        result, frame = cap.read()
        acc_frame = acc[i*acc_frame_delay:(i+1)*acc_frame_delay,:]
        acc_frame = np.mean(acc_frame,0)#x,y,z = left,up,forward
        if result:

            x,y,z = np.around(acc_frame,3)
            processed_frame = process_frame(frame,x,y,z)
            
            cv2.imshow('video', processed_frame)
        i +=1   
        key = cv2.waitKey(int(video_frame_delay * 1000))

        if key == 27 or key & 0xFF == ord('q'):
            break
        # time.sleep(frame_delay)

    cap.release()
    cv2.destroyAllWindows()

