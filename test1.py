import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy import sparse
from scipy.sparse.linalg import spsolve


position_history = []
initial_y = None
smoothing_window = 5 
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
    params.filterByConvexity = True
    params.minConvexity = 0.8
    

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(label_mask)
    circles = []
    for kp in keypoints:
        circles.append(np.array([int(kp.pt[0]), int(kp.pt[1]), int(kp.size / 2)]))
    
    return circles

def update_displacement_graph(circles, frame_count):
    global position_history, initial_y, timestamp_ms
    
    current_time_ms = int((frame_count / fps) * 1000)
    timestamp_ms.append(current_time_ms)
        
    # Get current position if a circle was detected
    if circles and len(circles) > 0:
        # Use the first circle if multiple were detected
        current_y = circles[0][1]
        
        if initial_y is None:
            initial_y = current_y
        
        displacement = initial_y - current_y
        position_history.append(displacement)
    else:
        if position_history:
            position_history.append(position_history[-1])
        else:
            position_history.append(0)


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
    original_height, original_width = frame.shape[:2]
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   


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