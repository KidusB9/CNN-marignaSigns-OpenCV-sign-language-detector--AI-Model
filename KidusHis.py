import cv2
import numpy as np
import pickle

def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    crop = None
    for i in range(10):
        row_img = None
        for j in range(5):
            if row_img is None:
                row_img = img[y:y+h, x:x+w]
            else:
                row_img = np.hstack((row_img, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            x += w + d
        if crop is None:
            crop = row_img
        else:
            crop = np.vstack((crop, row_img))
        x = 420
        y += h + d
    return crop

def get_hand_hist():
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError("Cannot open webcam")

    x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    hist = None

    try:
        while True:
            ret, img = cam.read()
            if not ret:
                continue
            img = cv2.flip(img, 1)
            imgCrop = build_squares(img)

            keypress = cv2.waitKey(1)
            if keypress == ord('c'):
                hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
                flagPressedC = True
                hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            elif keypress == ord('s') and hist is not None:
                break

            if flagPressedC:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
                disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                cv2.filter2D(dst, -1, disc, dst)
                blur = cv2.medianBlur(dst, 15)
                ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.merge((thresh, thresh, thresh))
                cv2.imshow("Threshold", thresh)
            cv2.imshow("Set Hand Histogram", img)
    finally:
        cam.release()
        cv2.destroyAllWindows()
        if hist is not None:
            with open("hist.pkl", "wb") as f:
                pickle.dump(hist, f)

if __name__ == "__main__":
    get_hand_hist()
