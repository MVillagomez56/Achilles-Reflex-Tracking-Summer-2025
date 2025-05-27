import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import signal
from scipy.signal import detrend
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.ticker as ticker

def detect_circles(image):
    _, labels, _, _ = cv2.connectedComponentsWithStats(image)
    label_mask = np.zeros_like(image)
    label_mask[labels == 1] = 255

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 10000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.8

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(label_mask)
    return [np.array([int(kp.pt[0]), int(kp.pt[1]), int(kp.size / 2)]) for kp in keypoints]

def process_frame(frame):
    original_height, original_width = frame.shape[:2]
    new_width = 640
    new_height = int((new_width / original_width) * original_height)
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    #convert to grayscale and threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(gray, 55, 245, cv2.THRESH_BINARY)
    
    return thresh

def track_movement(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # we need this for the ms_formatter
    if fps <= 0:
        fps = 30

    position_history = []
    timestamp_ms = []
    initial_y = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        frame_count += 1
        current_time_ms = int((frame_count / fps) * 1000)
        timestamp_ms.append(current_time_ms)

        thresh = process_frame(frame)
        circles = detect_circles(thresh)

        if circles and len(circles) > 0:
            # we ensure we are tracking the largest circle **
            current_y = circles[0][1]
            if initial_y is None:
                initial_y = current_y
            # we calculate the displacement of the largest circle from the initial y position
            displacement = initial_y - current_y
            position_history.append(displacement)
        else:
            if position_history:
                position_history.append(position_history[-1])
            else:
                position_history.append(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(position_history), np.array(timestamp_ms)

def baseline_als(y, lam, p, niter=10):
    #least squares smoothing
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def ms_formatter(x, pos):
    return f"{x/1000:.1f}s"

def estimate_polynomial_trend(y, timestamps, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X = timestamps.reshape(-1, 1)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model.predict(X_poly)

def calculate_differences(y):
    diff = np.diff(y)
    return np.insert(diff, 0, 0)  # add 0 for size

def analyze_movement(position_history, timestamp_ms):
    #linear detrending
    detrended_linear = detrend(position_history, type='linear')
    
    #polynomial detrending
    trend = estimate_polynomial_trend(position_history, timestamp_ms, degree=2)
    detrended_poly = position_history - trend
    
    #first order differencing
    differences = calculate_differences(position_history)
    
    return detrended_linear, detrended_poly, differences

def plot_analysis(position_history, timestamp_ms, detrended_linear, detrended_poly, differences, video_name):
    plt.figure(figsize=(20, 15))

    #original data
    plt.subplot(2, 2, 1)
    plt.plot(timestamp_ms, position_history, 'b-', label='original data', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(ms_formatter))
    plt.legend()
    plt.title('original time series')
    plt.grid(True)

    #linear detrending
    plt.subplot(2, 2, 2)
    plt.plot(timestamp_ms, detrended_linear, 'b-', label='linear detrended', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(ms_formatter))
    plt.legend()
    plt.title('linear detrending (scipy)')
    plt.grid(True)

    #polynomial detrending
    plt.subplot(2, 2, 3)
    plt.plot(timestamp_ms, detrended_poly, 'b-', label='detrended data', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(ms_formatter))
    plt.legend()
    plt.title('polynomial detrending')
    plt.grid(True)

    #first order differencing
    plt.subplot(2, 2, 4)
    plt.plot(timestamp_ms, differences, 'b-', label='first differences', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(ms_formatter))
    plt.legend()
    plt.title('first order differencing')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'graphs/{video_name}_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close()
    return

def run_analysis(video_name):
    
    video_path = f"videos/{video_name}"
    
    position_history, timestamp_ms = track_movement(video_path)
    detrended_linear, detrended_poly, differences = analyze_movement(position_history, timestamp_ms)
    
    plot_analysis(position_history, timestamp_ms, detrended_linear, detrended_poly, differences, video_name)
    
def main():
    for video_name in os.listdir("videos"):
        run_analysis(video_name)

main()

