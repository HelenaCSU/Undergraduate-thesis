#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from time import clock
# Create some random colors
color = np.random.randint(0, 255, (100, 3))

lk_params = dict( winSize  = (31, 31),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
sub_params=dict(winSize=(5, 5),
                zeroZone=(-1,-1),
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,0.03))

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []    #跟踪点
        # self.cam = video.create_capture(video_src)
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.prev_gray = None
        self.prev_frame = None
        self.tracks = []
        cv2.namedWindow('lk_track')
        cv2.setMouseCallback('lk_track', self.selectPoint)


    def selectPoint(self, event, x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.tracks.append([(x, y)])

    def setPoint(self):
        ret, frame = self.cam.read()
        frame = cv2.resize(frame, (1024, 768), interpolation=cv2.INTER_CUBIC)
        self.prev_frame = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = frame_gray
        vis = frame.copy()
        while(1):
            cv2.imshow('lk_track', vis)
            ch = cv2.waitKey(1) & 0xff
            if ch == 27:
                break
    def run(self):
        while True:
            ret, frame = self.cam.read()
            frame  = cv2.resize(frame, (1024,768), interpolation=cv2.INTER_CUBIC)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)


                good = d < 0.5  # 影响点消失
                # good = [st==1]
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), st):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                # p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                # if p is not None:
                #     for x, y in np.float32(p).reshape(-1, 2):
                #         self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = cv2.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    app = App('2.MP4')
    app.setPoint()
    app.run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
