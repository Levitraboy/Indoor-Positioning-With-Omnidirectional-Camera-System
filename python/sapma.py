import math
from threading import Thread
import cv2
import time
import numpy as np
import imutils
from skimage import transform as tf
from sklearn.metrics import mean_squared_error
from pose_est import *
from unit_sticker import unit_sticker
from e2c import e2c
from cube2sph import map_cube
from blue_detection import color_detection
import sys

text_calculated_L_BFGS_B_coord = open("coords_calculated_L-BFGS-B_coord_duz.txt", "w", newline='')
text_calculated_BFGS_coord = open("coords_calculated_BFGS_coord_duz.txt", "w", newline='')
text_calculated_L_BFGS_B_rot = open("coords_calculated_L-BFGS-B_rot_duz.txt", "w", newline='')
text_calculated_BFGS_rot = open("coords_calculated_BFGS_rot_duz.txt", "w", newline='')

def w2txt(BFGS, L_BFGS_B):
    np.savetxt(text_calculated_L_BFGS_B_rot, L_BFGS_B[0])
    np.savetxt(text_calculated_BFGS_rot, BFGS[0])
    np.savetxt(text_calculated_L_BFGS_B_coord, L_BFGS_B[1])
    np.savetxt(text_calculated_BFGS_coord, BFGS[1])

def mandel():
    #cv2.namedWindow("resized", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("resized", 1366, 768)

    scene_change = False
    sticker_select = True
    add_new_marker = False
    leftest_marker = False
    rightest_marker = False
    cw = 256
    k = 8

    stickers = np.array([[0, 8, 4], [10, 9, 4], [10, 6, 4], [10, 3, 4], [8, 0, 4], [5, 0, 4], [2, 0, 4], [0, 4, 4]],dtype=np.double)
    pixel_stickers = np.zeros((8,3))
    v = cv2.VideoCapture("path") # video path 

    while True:
        kk = 0
        ret, frame = v.read()
        a = time.time()
        if ret:
            frame = e2c(frame) # equirectangular projection to cubic format conversion
        else:
            break

        if sticker_select:
            tracker = cv2.legacy_MultiTracker.create()
            if not scene_change:
                old_coordinates = np.zeros((8,4))
                for i in range(k):
                    cv2.imshow("resized", frame)
                    bbi = cv2.selectROI("resized",frame)
                    tracker_i = cv2.legacy_TrackerCSRT.create()
                    tracker.add(tracker_i, frame, bbi)
                    old_coordinates[i] = bbi

                sticker_select = False

            else:
                for i in range(k):

                    tracker_i = cv2.legacy_TrackerCSRT.create()
                    tracker.add(tracker_i, frame, (boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]))
                old_coordinates = np.array(boxes) 
                scene_change = False
                sticker_select = False

        # frame = imutils.resize(frame, width=1920, height=1080)
        (H, W) = frame.shape[:2]
        (success, boxes) = tracker.update(frame)

        frame_copy = frame.copy()
        tform = tf.estimate_transform('similarity', old_coordinates, boxes)

        old_coordinates = boxes
        if success:
            leftmost = min(boxes[:,0])
            leftmost_w = boxes[np.nanargmin(boxes[:,0])][2]
            rightmost = np.amax(boxes[:,0])
            rightmost_w = boxes[np.nanargmax(boxes[:,0])][2]
            
            for box in boxes:
                (x,y,w,h) = [int(a) for a in box]
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h),(100,255,0),1)

                cv2.putText(frame_copy, str(kk+1), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                if (x+w/2)//cw == 0 and (y+h/2)//cw == 1:
                    face = "left"
                elif (x+w/2)//cw == 1 and (y+h/2)//cw == 1:
                    face = "front"
                elif (x+w/2)//cw == 2 and (y+h/2)//cw == 1:
                    face = "right"
                elif (x+w/2)//cw == 3 and (y+h/2)//cw == 1:
                    face = "back"
                elif (x+w/2)//cw == 1 and (y+h/2)//cw == 0:
                    face = "top"
                elif (x+w/2)//cw == 1 and (y+h/2)//cw == 2:
                    face = "bottom"

                _u = (x+w/2)%cw
                _v = (y+h/2)%cw

                _u, _v = map_cube(_u,_v,face,cw, 1920, 1080)
                pixel_stickers[kk] = [_u, _v, 1]
                kk = kk + 1

            unit = unit_sticker(pixel_stickers,1920,1080,len(boxes))
            pose = PoseEstimator(stickers, unit)
            pose_solve_L_BFGS_B = pose.solve(method='L-BFGS-B')
            pose_solve_BFGS = pose.solve() 
            print(pose_solve_BFGS)
            txt_thread = Thread(target = w2txt, args=(pose_solve_BFGS, pose_solve_L_BFGS_B))
            txt_thread.start()
            # print ('L-BFGS-B:', pose.solve(method='L-BFGS-B'))

        b = time.time()
        cv2.putText(frame_copy, "FPS: {}".format(1/(b-a)), (10, H - ((2 * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("resized",frame_copy)

        if int(leftmost) <= 5:
            boxes[0][0] = 1024 #1920
            first_sticker = boxes[0]
            stickers_deleted_left = stickers[0]
            stickers = np.delete(stickers, 0, 0)
            pixel_stickers = np.delete(pixel_stickers, 0, 0)
            boxes = np.delete(boxes, 0, 0)

            k = len(boxes)
            sticker_select = True
            scene_change = True
            add_new_marker = True
            leftest_marker = True
        elif int(rightmost+rightmost_w)  >= 1019: #1915
            last_sticker = boxes[-1]
            stickers_deleted_right = stickers[-1]
            stickers = np.delete(stickers, -1, 0)
            pixel_stickers = np.delete(pixel_stickers, -1, 0)
            #sticker_order.pop(-1)
            boxes = np.delete(boxes, -1, 0)

            k = len(boxes)
            sticker_select = True
            scene_change = True
            add_new_marker = True
            rightest_marker = True

        if add_new_marker:
            if leftest_marker == True:
                first_sticker = tf.matrix_transform(first_sticker, tform.params)#mt is the same dst

                if first_sticker[0][0] + first_sticker[0][2] < 1014: # 1910

                    stickers = np.append(stickers, [stickers_deleted_left], axis=0)

                    boxes = color_detection(frame)
                    pixel_stickers = np.zeros((len(boxes),3))

                    k = len(boxes)
                    sticker_select = True
                    scene_change = True
                    leftest_marker = False
                    if rightest_marker == False:
                        add_new_marker = False
                elif first_sticker[0][0] - cw*4 > 10:
                    stickers = np.insert(stickers, 0, [stickers_deleted_left], axis=0)

                    boxes = color_detection(frame)
                    pixel_stickers = np.zeros((len(boxes),3))


                    k = len(boxes)
                    sticker_select = True
                    scene_change = True
                    leftest_marker = False
                    if rightest_marker == False:
                        add_new_marker = False

            elif rightest_marker == True:
                last_sticker = tf.matrix_transform(last_sticker, tform.params)#mt is the same dst
                #print("\n",last_sticker)                if last_sticker[0][0] > 1034: #1930
                    last_sticker[0][0] = last_sticker[0][0] - 1024 #1920
                    stickers = np.insert(stickers, 0, [stickers_deleted_right], axis=0)

                    boxes = color_detection(frame)
                    pixel_stickers = np.zeros((len(boxes),3))
                    k = len(boxes)
                    sticker_select = True
                    scene_change = True
                    rightest_marker = False
                    if leftest_marker == False:
                        add_new_marker = False
                
                elif last_sticker[0][0] + last_sticker[0][2] < 1014: #1910
                    stickers = np.append(stickers, [stickers_deleted_right], axis=0)

                    boxes = color_detection(frame)
                    pixel_stickers = np.zeros((len(boxes),3))

                    k = len(boxes)
                    sticker_select = True
                    scene_change = True
                    rightest_marker = False
                    if leftest_marker == False:
                        add_new_marker = False

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    v.release()
    cv2.destroyAllWindows()
    text_calculated_L_BFGS_B_coord.close()
    text_calculated_BFGS_coord.close()
    text_calculated_L_BFGS_B_rot.close()
    text_calculated_BFGS_rot.close()


main_thread = Thread(target = mandel) # to run main definiton that algorithm runs
main_thread.start()
