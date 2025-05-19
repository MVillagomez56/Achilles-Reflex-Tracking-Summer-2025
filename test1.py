# import cv2
# import numpy as np

# def detect_circles(image):
#     _, labels, _, _ = cv2.connectedComponentsWithStats(image)

#     # Extract the pixels associated with the current label
#     label_mask = np.zeros_like(image)
#     label_mask[labels == 1] = 255


#     cv2.imshow('Label Mask', label_mask)
    

#     params = cv2.SimpleBlobDetector_Params()

#     # Set Area filtering parameters
#     params.filterByArea = True
#     params.minArea = 500  # Adjust based on your circle size
#     params.maxArea = 10000000
    
#     # Circularity filtering - decrease threshold to catch more circle-like shapes
#     params.filterByCircularity = True
#     params.minCircularity = 0.7  # Lower value to catch more imperfect circles
#  # Add convexity filtering - helps with stability
#     params.filterByConvexity = True
#     params.minConvexity = 0.8
    
#     # Add inertia filtering - helps with stability
#     params.filterByInertia = True
#     params.minInertiaRatio = 0.6  # Higher values = more circular objects

#     # Create a detector with the parameters
#     detector = cv2.SimpleBlobDetector_create(params)

#     # Detect blobs on the inverted mask
#     keypoints = detector.detect(label_mask)
    
#     # Debug visualization - show detected blobs on inverted mask
#     debug_view = cv2.cvtColor(label_mask, cv2.COLOR_GRAY2BGR)
#     for kp in keypoints:
#         x, y = int(kp.pt[0]), int(kp.pt[1])
#         r = int(kp.size / 2)
#         cv2.circle(debug_view, (x, y), r, (0, 255, 0), 2)
#     cv2.imshow('Detected Blobs', debug_view)
    
#     circles = []
#     for kp in keypoints:
#         circles.append(np.array([int(kp.pt[0]), int(kp.pt[1]), int(kp.size / 2)]))
    
#     print("Number of circles detected:", len(circles))
#     return circles

# cap = cv2.VideoCapture("video.mp4")

# while True:
#     ret, frame = cap.read()
#     #resize fram based on aspect ratio
#     # Get the original dimensions
#     original_height, original_width = frame.shape[:2]
#     # Calculate the new dimensions
#     new_width = 640
#     new_height = int((new_width / original_width) * original_height)

#     frame= cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
#     if not ret:
#         print("End of video")
#         break

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Apply Gaussian blur
#     gray = cv2.GaussianBlur(gray, (11, 11), 0)
#     # Threshold the image
#     _, thresh = cv2.threshold(gray, 55, 245, cv2.THRESH_BINARY)
#     #invert the thresholded image

#     #display the thresholded image, resized to 640x480
#     cv2.imshow('Thresholded Image', thresh)

#     circles = detect_circles(thresh)

#     # draw the circles on image here 
#     for circle in circles:
#         cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
#     # Display the image with circles
#     cv2.imshow('Circles', frame)
#     # Display the original frame
#     cv2.imshow('Original Frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break   

# cv2.destroyAllWindows()
# cap.release()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add this global variable to store position history
position_history = []
initial_y = None
smoothing_window = 5  # Adjust based on desired smoothness
plt.figure(figsize=(10, 6))
line, = plt.plot([], [], 'b-')
plt.xlabel('Frame')
plt.ylabel('Vertical Displacement (pixels)')
plt.title('Circle Vertical Movement')
plt.grid(True)
plt.ion()  # Turn on interactive mode

def detect_circles(image):
    _, labels, _, _ = cv2.connectedComponentsWithStats(image)

    label_mask = np.zeros_like(image)
    label_mask[labels == 1] = 255

    # cv2.imshow('Label Mask', label_mask)

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 10000000
    
    params.filterByCircularity = True
    params.minCircularity = 0.7
    
    # Convexity filtering
    params.filterByConvexity = True
    params.minConvexity = 0.8
    

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(label_mask)
    
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

def update_displacement_graph(circles, frame_count):
    global position_history, initial_y, timestamp_ms
    
    # Calculate current timestamp in milliseconds
    current_time_ms = int((frame_count / fps) * 1000)
    timestamp_ms.append(current_time_ms)
        
    # Get current position if a circle was detected
    if circles and len(circles) > 0:
        # Use the first circle if multiple were detected
        current_y = circles[0][1]
        
        # Initialize initial_y if not set
        if initial_y is None:
            initial_y = current_y
            
        # Calculate displacement
        displacement = initial_y - current_y  # Positive = upward movement
        
        # Add to history
        position_history.append(displacement)
        
        # Apply moving average to reduce jitter
        if len(position_history) > smoothing_window:
            # Get subset of recent values
            recent_values = position_history[-smoothing_window:]
            # Apply smoothing
            smoothed_value = sum(recent_values) / len(recent_values)
            # Replace the most recent value with smoothed value
            position_history[-1] = smoothed_value
    else:
        # If no circle detected, use the last known position or zero
        if position_history:
            position_history.append(position_history[-1])
        else:
            position_history.append(0)
    
    # Update the plot (only show last 100 frames for clarity)
    display_range = 100
    if len(position_history) > display_range:
        x_data = list(range(frame_count - display_range + 1, frame_count + 1))
        y_data = position_history[-display_range:]
    else:
        x_data = list(range(frame_count - len(position_history) + 1, frame_count + 1))
        y_data = position_history
    
    # Update plot
    display_range = 100
    if len(position_history) > display_range:
        x_data = timestamp_ms[-display_range:]
        y_data = position_history[-display_range:]
    else:
        x_data = timestamp_ms
        y_data = position_history
    
    # Update plot
    # line.set_xdata(x_data)
    # line.set_ydata(y_data)
    # plt.xlim(min(x_data), max(x_data))
    # if y_data:
    #     plt.ylim(min(min(y_data)-5, -5), max(max(y_data)+5, 5))
    # plt.xlabel('Time (ms)')
    # plt.draw()
    # plt.pause(0.01)


cap = cv2.VideoCapture("video.mp4")
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  

# Create a list to store timestamps in milliseconds
timestamp_ms = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break
    
    frame_count += 1
    
    # Get the original dimensions
    original_height, original_width = frame.shape[:2]
    # Calculate the new dimensions
    new_width = 640
    new_height = int((new_width / original_width) * original_height)

    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Threshold the image
    _, thresh = cv2.threshold(gray, 55, 245, cv2.THRESH_BINARY)

    # Detect circles
    circles = detect_circles(thresh)

    # Update the displacement graph
    update_displacement_graph(circles, frame_count)

    # # Draw circles on image
    # for circle in circles:
    #     cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
    #     # Add text with displacement
    #     if initial_y is not None:
    #         displacement = initial_y - circle[1]
    #         cv2.putText(frame, f"Disp: {displacement:.1f}px", 
    #                   (circle[0] - 50, circle[1] - circle[2] - 10),
    #                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # # Display the image with circles
    # cv2.imshow('Circles', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z


import matplotlib.ticker as ticker
def ms_formatter(x, pos):
    seconds = x / 1000
    return f"{seconds:.1f}s"

smoothed_displacement = baseline_als(np.array(position_history), lam=10, p=0.01, niter=10)
plt.plot(timestamp_ms, smoothed_displacement, 'r-', label='Smoothed Displacement')
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(ms_formatter))
plt.legend()
plt.show()
plt.pause(0.01)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(ms_formatter))


plt.savefig('smoothed_displacement.png', dpi=300, bbox_inches='tight')

cv2.destroyAllWindows()
cap.release()
plt.close('all')