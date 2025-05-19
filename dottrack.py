import cv2
import numpy as np

def detect_circles(image):
    _, labels, _, _ = cv2.connectedComponentsWithStats(image)
    label_mask = np.zeros_like(image)
    label_mask[labels == 1] = 255

    # cv2.imshow('Label Mask', label_mask)
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 500 
    params.maxArea = 100000
    
    params.filterByCircularity = True
    params.minCircularity = 0.7 

    params.filterByConvexity = True
    params.minConvexity = 0.8
    
    params.filterByInertia = True
    params.minInertiaRatio = 0.6 

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs on the inverted mask
    keypoints = detector.detect(label_mask)
    
    # debug visualization
    # debug_view = cv2.cvtColor(label_mask, cv2.COLOR_GRAY2BGR)
    # for kp in keypoints:
    #     x, y = int(kp.pt[0]), int(kp.pt[1])
    #     r = int(kp.size / 2)
    #     cv2.circle(debug_view, (x, y), r, (0, 255, 0), 2)
    # cv2.imshow('Detected Blobs', debug_view)
    
    circles = []
    for kp in keypoints:
        circles.append(np.array([int(kp.pt[0]), int(kp.pt[1]), int(kp.size / 2)]))
    
    # print("Number of circles detected:", len(circles))
    return circles


def track(path):
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        #resize fram based on aspect ratio
        # Get the original dimensions
        original_height, original_width = frame.shape[:2]
        # Calculate the new dimensions
        new_width = 640
        new_height = int((new_width / original_width) * original_height)

        frame= cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        if not ret:
            print("End of video")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(gray, 55, 245, cv2.THRESH_BINARY)
        #invert the thresholded image

        # #display the thresholded image, resized to 640x480
        # cv2.imshow('Thresholded Image', thresh)

        circles = detect_circles(thresh)

        # draw the circles on image here 
        for circle in circles:
            cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        cv2.imshow('Circles', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

    cv2.destroyAllWindows()
    cap.release()

track("video.mp4")