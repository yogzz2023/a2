import numpy as np
from filterpy.kalman import KalmanFilter

# Initialize Kalman Filter
kf = KalmanFilter(dim_x=4, dim_z=2)

# Define the state transition matrix
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

# Define the measurement function
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])

# Define the measurement noise covariance matrix
kf.R = np.array([[0.1, 0],
                 [0, 0.1]])

# Define the process noise covariance matrix
kf.Q = np.eye(4) * 0.01

# Initialize the state and covariance matrix
kf.x = np.array([0, 0, 0, 0])
kf.P = np.eye(4) * 500

# Load data from the CSV file
data = np.genfromtxt('test.csv', delimiter=',', skip_header=1)

# Extract measurement data
measurements = data[:, 6:10]

# Perform Kalman filtering
for measurement in measurements:
    if np.isnan(measurement).any():
        continue
    if measurement.size == 4:
        measurement = measurement[:2].reshape(2, 1)  # Reshape the measurement data
        kf.update(measurement)
        kf.predict()

# Extract predicted range, azimuth, elevation, and time
predicted_range = kf.x[0]
predicted_azimuth = kf.x[1]
predicted_elevation = kf.x[2]
predicted_time = kf.x[3]

# Print the predicted values
print("Predicted Range:", predicted_range)
print("Predicted Azimuth:", predicted_azimuth)
print("Predicted Elevation:", predicted_elevation)
print("Predicted Time:", predicted_time)