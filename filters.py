import numpy as np

class KalmanFilter2D:
    def __init__(self, dt=1.0/60.0, process_noise=0.05, measurement_noise=0.2):
        """
        A 2D Kalman Filter tracking position [x, y] and velocity [vx, vy].
        dt: Time step between updates.
        process_noise: Determines how much we trust our model dynamics vs measurements (larger = more responsive, noisier).
        measurement_noise: Determines how noisy the measurements are (larger = smoother, more lag).
        """
        self.dt = dt
        # State vector: [x, y, vx, vy]^T
        self.x = np.zeros(4, dtype=np.float32)
        # Covariance matrix
        self.P = np.eye(4, dtype=np.float32) * 10.0
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        self.initialized = False

    def update(self, z):
        """
        Update the filter with a new measurement z = [x, y]
        """
        if not self.initialized:
            self.x[0] = z[0]
            self.x[1] = z[1]
            self.x[2] = 0.0
            self.x[3] = 0.0
            self.initialized = True
            return z
        
        # Prediction
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        
        # Innovation
        y = z - self.H.dot(self.x)
        
        # Innovation covariance
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        
        # Kalman gain
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        
        # State update
        self.x = self.x + K.dot(y)
        
        # Covariance update
        self.P = (np.eye(4, dtype=np.float32) - K.dot(self.H)).dot(self.P)
        
        return self.x[:2]

    def predict(self):
        """
        Predict the next state if measurement is missing
        """
        if not self.initialized:
            return np.zeros(2, dtype=np.float32)
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x[:2]

    def reset(self):
        self.initialized = False
        self.x.fill(0)
        self.P = np.eye(4, dtype=np.float32) * 10.0
