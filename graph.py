import pickle
import matplotlib.pyplot as plt

# Load training history
try:
    with open("history.pkl", "rb") as f:
        history = pickle.load(f)
except FileNotFoundError:
    print("Error: Training history not found. Train the model first.")
    exit()

# Extract accuracy and loss values
train_loss = history['loss']  # Training Loss
val_loss = history['val_loss']  # Validation Loss (Testing)
train_acc = history['accuracy']  # Training Accuracy
val_acc = history['val_accuracy']  # Validation Accuracy (Testing)
epochs = range(1, len(train_loss) + 1)

# Plot Training vs Testing Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo-', label="Training Accuracy")
plt.plot(epochs, val_acc, 'ro-', label="Testing Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.legend()

# Plot Training vs Testing Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label="Training Loss")
plt.plot(epochs, val_loss, 'ro-', label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Testing Loss")
plt.legend()

# Show plots
plt.tight_layout()
plt.show()
