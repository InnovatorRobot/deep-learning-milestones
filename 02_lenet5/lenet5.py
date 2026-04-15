from network_utils import Flatten, Conv2D, Dense, AvgPool2D, softmax, tanh


class LeNet5:
    def __init__(self):
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = AvgPool2D(pool_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = AvgPool2D(pool_size=2, stride=2)
        self.dense1 = Dense(in_features=16 * 5 * 5, out_features=120)
        self.dense2 = Dense(in_features=120, out_features=84)
        self.dense3 = Dense(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = tanh(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = tanh(x)
        x = self.pool2.forward(x)

        x = Flatten().forward(x)

        x = self.dense1.forward(x)
        x = tanh(x)

        x = self.dense2.forward(x)
        x = tanh(x)

        x = self.dense3.forward(x)
        probabilities = softmax(x)
        return probabilities


if __name__ == "__main__":
    import numpy as np

    # Create a completely random 32x32 image (1 channel for black and white)
    dummy_image = np.random.rand(1, 32, 32)

    model = LeNet5()

    predictions = model.forward(dummy_image)

    print("Probabilities for each digit (0-9):")
    for i, prob in enumerate(predictions):
        print(f"Digit {i}: {prob*100:.2f}%")

    print(f"\nThe network guesses this random noise is the number: {np.argmax(predictions)}")
