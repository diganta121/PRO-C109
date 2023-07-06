import numpy as np
import pyautogui
import time
import imutils
import cv2
import mediapipe as mp

from pynput.keyboard import Key, Controller

keyboard = Controller()


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4
thumb_status = False


def draw_circle(tip, lm_list, state=False):
    x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)

    if state == True:
        cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)
    elif state == False:
        cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)


def finger_count(lm_list):
    global finger_tips
    finger_fold_status = []

    for tip in finger_tips:
        # getting the landmark tip position and drawing blue circle
        # writing condition to check if finger is folded i.e checking if finger tip starting value is smaller than finger starting position which is inner landmark. for index finger
        # if finger folded changing color to green

        state = lm_list[tip - 3].y < lm_list[tip].y

        draw_circle(tip, lm_list, state=state)

        if state:
            finger_fold_status.append(True)
        else:
            finger_fold_status.append(False)
    return finger_fold_status


def screenshot():
    image = pyautogui.screenshot("screenshot.png")
    time.sleep(0.5)


def readimg(name):
    img = cv2.imread(name)
    cv2.imshow("screenshot", imutils.resize(img, width=600))


def trigger_sign(finger_status):
    # check if the correct sign is done
    # put ring finger down to take screenshot
    if finger_status == [False, False, True, False]:
        print("ss ")  # take screenshot
        screenshot()
        readimg("screenshot.png")


while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # accessing the landmarks by their position
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # array to hold true or false if finger is folded

            # checking if all fingers are folded
            finger_fold_status = finger_count(lm_list)

            trigger_sign(finger_fold_status)
            print(finger_fold_status)

            mp_draw.draw_landmarks(
                img,
                hand_landmark,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                mp_draw.DrawingSpec((0, 255, 0), 4, 2),
            )

    cv2.imshow("hand tracking", img)
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()
