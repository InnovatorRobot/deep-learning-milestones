import numpy as np


def tanh(x):
    return np.tanh(x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


class Dense:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(out_features, in_features) * 0.01
        self.biases = np.zeros(out_features)

    def forward(self, x):
        # Y = W * X + b
        return np.dot(self.weights, x) + self.biases


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        C, H, W = x.shape
        out_H = H // self.stride
        out_W = W // self.stride

        output = np.zeros((C, out_H, out_W))

        for c in range(C):  # For every channel independently
            for y in range(out_H):
                for x in range(out_W):
                    # Find the correct 2x2 window in the input
                    y_start = y * self.stride
                    x_start = x * self.stride

                    # Cut it out and calculate the max
                    window = x[
                        c, y_start : y_start + self.pool_size, x_start : x_start + self.pool_size
                    ]
                    output[c, y, x] = np.max(window)

        return output


class AvgPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        C, H, W = input_data.shape
        out_H = H // self.stride
        out_W = W // self.stride

        output = np.zeros((C, out_H, out_W))

        for c in range(C):  # For every channel independently
            for y in range(out_H):
                for x in range(out_W):
                    # Find the correct 2x2 window in the input
                    y_start = y * self.stride
                    x_start = x * self.stride

                    # Cut it out and calculate the average
                    window = input_data[
                        c, y_start : y_start + self.pool_size, x_start : x_start + self.pool_size
                    ]
                    output[c, y, x] = np.mean(window)

        return output


class Flatten:
    def __init__(self):
        pass

    def forward(self, x):
        return x.flatten()


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.k = kernel_size
        self.s = stride
        self.p = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(out_channels)

    def forward(self, input_data):
        C, H, W = input_data.shape

        # Calculate output dimensions
        out_H = (H - self.k + 2 * self.p) // self.s + 1
        out_W = (W - self.k + 2 * self.p) // self.s + 1

        output = np.zeros((self.out_channels, out_H, out_W))

        # Apply padding
        if self.p > 0:
            input_data = np.pad(
                input_data, ((0, 0), (self.p, self.p), (self.p, self.p)), mode="constant"
            )

        # Convolution operation
        for f in range(self.out_channels):
            for y in range(out_H):
                for x in range(out_W):
                    image_chunk = input_data[
                        :, y * self.s : y * self.s + self.k, x * self.s : x * self.s + self.k
                    ]
                    output[f, y, x] = np.sum(image_chunk * self.weights[f]) + self.biases[f]

        return output
