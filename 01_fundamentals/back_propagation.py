import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(output):
    return output * (1 - output)


x = 1.0  # Input value
t = 1.0  # Target output
w1 = 0.5  # Weight from input to hidden layer
w2 = 0.5  # Weight from hidden layer to output layer
lr = 0.5  # Learning rate

# Forward pass
z1 = w1 * x
h = sigmoid(z1)

z2 = w2 * h
y = sigmoid(z2)

error = 0.5 * (t - y) ** 2  # Mean Squared Error

print(f"Initial prediction, error and real value: y = {y}, error = {error}, target = {t}")

# Backward pass
# Gradients for Weight 2 (w2)
dE_dy = -(t - y)  # Derivative of Error wrt output
dy_dz2 = sigmoid_derivative(y)  # Derivative of output wrt z2
dz2_dw2 = h  # Derivative of z2 wrt w2

dE_dw2 = dE_dy * dy_dz2 * dz2_dw2  # Chain rule

# Gradients for Weight 1 (w1)
dE_dh = dE_dy * dy_dz2 * w2  # Derivative of Error wrt hidden layer output
dh_dz1 = sigmoid_derivative(h)  # Derivative of hidden layer output wrt z1
dz1_dw1 = x  # Derivative of z1 wrt w

dE_dw1 = dE_dh * dh_dz1 * dz1_dw1  # Chain rule

# --- WEIGHT UPDATE ---
w1 = w1 - (lr * dE_dw1)
w2 = w2 - (lr * dE_dw2)

print(f"New w1: {w1:.4f} (Increased from 0.5)")
print(f"New w2: {w2:.4f} (Increased from 0.8)\n")

print("=== TRAINING LOOP (Watching the network learn) ===")
# Let's train it for 50 more epochs (rounds) to see it reach the target!
epochs = 100
for epoch in range(1, epochs + 1):
    # Forward pass
    z1 = w1 * x
    h = sigmoid(z1)
    z2 = w2 * h
    y = sigmoid(z2)

    # Backward pass
    dE_dy = -(t - y)
    dy_dz2 = sigmoid_derivative(y)

    dE_dw2 = dE_dy * dy_dz2 * h
    dE_dw1 = (dE_dy * dy_dz2 * w2) * sigmoid_derivative(h) * x

    # Update Weights
    w1 -= lr * dE_dw1
    w2 -= lr * dE_dw2

    if epoch % 10 == 0:
        current_error = 0.5 * (t - y) ** 2
        print(f"Epoch {epoch:2d} -> Prediction: {y:.4f} | Error: {current_error:.6f}")

print("\nSuccess! The network learned to predict the target.")
