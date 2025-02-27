import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = np.eye(F.shape[0])  # Initial state covariance
        self.x = np.zeros((F.shape[0], 1))  # Initial state

    def predict(self):
        # Predict state and covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Update step
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Compute association probabilities
        association_probabilities = self.compute_association_probabilities(S)

        # Update state based on association probabilities
        predicted_states = []
        for i, prob in enumerate(association_probabilities):
            K = np.dot(np.dot(self.P, self.H.T), inv(S))
            x_i = self.x + np.dot(K, y)
            self.P = self.P - np.dot(np.dot(K, self.H), self.P)
            predicted_states.append(x_i)
        
        # Select the predicted state with highest association probability
        max_prob_index = np.argmax(association_probabilities)
        self.x = predicted_states[max_prob_index]

        # Print predicted state and covariance
        print("Predicted State:")
        print(self.x)
        print("\nPredicted Covariance:")
        print(self.P)
        
        # Innovation
        innovation = y
        print("\nInnovation:")
        print(innovation)
        
        # Kalman Gain
        kalman_gain = np.dot(np.dot(self.P, self.H.T), inv(S))
        print("\nKalman Gain:")
        print(kalman_gain)
        
        # Updated state
        updated_state = self.x
        print("\nUpdated State:")
        print(updated_state)
        
        # Updated covariance
        updated_covariance = self.P
        print("\nUpdated Covariance:")
        print(updated_covariance)
        
        # JPDA (Joint Probabilistic Data Association)
        print("\nJPDA:")
        print("Measurement Likelihood:", 1 / (2 * np.pi * np.sqrt(np.linalg.det(S))) * np.exp(-0.5 * np.dot(y.T, np.dot(inv(S), y))))
        print("Association Probabilities:", association_probabilities)

    def compute_association_probabilities(self, S):
        # Compute association probabilities using Mahalanobis distance
        num_measurements = self.H.shape[0]
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
        association_probabilities = np.zeros(num_measurements)
        for i in range(num_measurements):
            d = np.dot(np.dot((self.H[i] - np.dot(self.H[i], self.x)).T, inv_S), (self.H[i] - np.dot(self.H[i], self.x)))
            association_probabilities[i] = np.exp(-0.5 * d) / ((2 * np.pi) ** (self.H.shape[0] / 2) * np.sqrt(det_S))
        return association_probabilities

def main():
    # Read data from CSV file, read only the specified columns
    data = pd.read_csv("test.csv", usecols=[10, 11, 12, 13])

    # Extract data into separate arrays
    measurements = data.values

    # Define state transition matrix
    F = np.eye(4)  # Assume constant velocity model for simplicity

    # Define measurement matrix
    H = np.eye(4)  # Identity matrix since measurement directly reflects state

    # Define process noise covariance matrix
    Q = np.eye(4) * 0.1  # Process noise covariance

    # Define measurement noise covariance matrix
    R = np.eye(4) * 0.01  # Measurement noise covariance, adjusted variance

    # Initialize Kalman filter
    kf = KalmanFilter(F, H, Q, R)

    # Lists to store predicted values for all variables
    predicted_ranges = []
    predicted_azimuths = []
    predicted_elevations = []
    predicted_times = []

    # Predict and update for each measurement
    for i, z in enumerate(measurements, start=1):
        # Predict
        kf.predict()

        # Update with measurement
        kf.update(z[:, np.newaxis])

        # Get predicted state
        predicted_state = kf.x.squeeze()

        # Append predicted values for all variables
        predicted_ranges.append(predicted_state[0])
        predicted_azimuths.append(predicted_state[1])
        predicted_elevations.append(predicted_state[2])
        predicted_times.append(i)

        # Print predicted values
        print(f"Measurement {i}:")
        print("Predicted Range:", predicted_state[0])
        print("Predicted Azimuth:", predicted_state[1])
        print("Predicted Elevation:", predicted_state[2])
        print()  # Add an empty line for separation

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plot measured and predicted range against time
    plt.plot(predicted_times, measurements[:, 0], label='Measured Range', marker='o')
    plt.plot(predicted_times, predicted_ranges, label='Predicted Range', linestyle='--', marker='o')
    plt.xlabel('Time (measurement index)')
    plt.ylabel('Range')
    plt.title('Range Measurement and Prediction vs. Time')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))

    # Plot measured and predicted azimuth against time
    plt.plot(predicted_times, measurements[:, 1], label='Measured Azimuth', marker='o')
    plt.plot(predicted_times, predicted_azimuths, label='Predicted Azimuth', linestyle='--', marker='o')
    plt.xlabel('Time (measurement index)')
    plt.ylabel('Azimuth')
    plt.title('Azimuth Measurement and Prediction vs. Time')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))

    # Plot measured and predicted elevation against time
    plt.plot(predicted_times, measurements[:, 2], label='Measured Elevation', marker='o')
    plt.plot(predicted_times, predicted_elevations, label='Predicted Elevation', linestyle='--', marker='o')
    plt.xlabel('Time (measurement index)')
    plt.ylabel('Elevation')
    plt.title('Elevation Measurement and Prediction vs. Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
