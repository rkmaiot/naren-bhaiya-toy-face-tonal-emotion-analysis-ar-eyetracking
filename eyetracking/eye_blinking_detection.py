import cv2
import dlib
from math import hypot


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img = cv2.imread('distracted2.png')


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


font = cv2.FONT_HERSHEY_SIMPLEX
Frame_count = 1
PosWindow1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
PosWindow2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(
        eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(
        eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(
        eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(
        eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (255, 0, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (255, 0, 0), 2)

    hor_line_lenght = hypot(
        (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot(
        (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    Frame_count = Frame_count+1

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()

        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio(
            [36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio(
            [42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        PosWindow1.append(landmarks.part(27).x)
        PosWindow2.append(landmarks.part(27).y)
        PosWindow1.pop(0)
        PosWindow2.pop(0)

        max_x = max(PosWindow1)
        max_y = max(PosWindow2)
        min_x = min(PosWindow1)
        min_y = min(PosWindow2)
        PosWindow1 = [max_x if x == 1 else x for x in PosWindow1]

        PosWindow2 = [max_x if x == 1 else x for x in PosWindow2]
        print(PosWindow1)
        if(Frame_count > 25):
            print(max_x-min_x)
            Move_x = max_x-min_x
            Move_y = max_y-min_y

            if(Move_x > 15 or Move_y > 15):
                cv2.putText(frame, "You are getting distracted. Try to focus now.",
                            (150, 50), font, 1, (0, 0, 255), 2)
                img_height, img_width, _ = img.shape
                x = 0
                y = 0
                frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print('image dimensions (HxW):', img_height, "x", img_width)
                print('frame dimensions (HxW):', int(
                    frame_height), "x", int(frame_width))
                frame[y:y+img_height, x:x+img_width] = img

        Pixell_xy = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
        for pix_val in Pixell_xy:
            image = cv2.circle(frame, (landmarks.part(
                pix_val).x, landmarks.part(pix_val).y), 1, (0, 0, 255), 2)

        if blinking_ratio < 4:
            print("blink ration -> ", blinking_ratio)
            cv2.putText(frame, "You are getting distracted. Try to focus now.",
                        (150, 50), font, 1, (0, 0, 255), 2)
            img_height, img_width, _ = img.shape
            x = 0
            y = 0
            frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame[y:y+img_height, x:x+img_width] = img

        for pix_val in range(27, 28):
            image = cv2.circle(frame, (landmarks.part(
                pix_val).x, landmarks.part(pix_val).y), 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(20)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
