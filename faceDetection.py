import cv2
import mediapipe as mp
import tracker_servo as tracker
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    resultsFace = face_detection.process(image)
    resultsHands = hands.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if resultsFace.detections:
      for detection in resultsFace.detections:

        ###########################################
        ### Start My tracking code ######################

        # print("Face detection landmarks: ", detection.location_data.relative_keypoints[2].x)
        xNose = detection.location_data.relative_keypoints[2].x
        yNose = detection.location_data.relative_keypoints[2].y
        imageSize = image.shape
        noseX = int(imageSize[1] * xNose)
        noseY = int(imageSize[0] * yNose)
        imgCenterX = int(imageSize[1]/2)
        yMiddle = .3 #set to area of y Middle (how far up to track the face)
        imgCenterY = int(imageSize[0]*yMiddle)
        cv2.circle(image,(imgCenterX,imgCenterY),10,(0,50,255),2)
        print("Nose location: ", xNose, yNose)
        cv2.circle(image,(noseX,noseY),10,(0,255,100),2)

        ##### move camera #########
        slack = .07 #amout of tolerance before starting movement
        if noseX > (imgCenterX + int(slack * imageSize[1])):
          tracker.left()
        if noseX < (imgCenterX - int(slack * imageSize[1])):
          tracker.right()
        if noseY > (imgCenterY + int(slack * imageSize[0])):
          tracker.down()
        if noseY < (imgCenterY - int(slack * imageSize[0])):
          tracker.up()

        ### End My tracking code ######################
        ###########################################
        mp_drawing.draw_detection(image, detection)

    if resultsHands.multi_hand_landmarks:
      for hand_landmarks in resultsHands.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()