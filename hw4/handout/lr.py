import numpy as np
import argparse
import matplotlib.pyplot as plt

def sigmoid(x: np.ndarray) -> np.ndarray:
    e = np.exp(-x)
    return 1 / (1 + e)

def calculate_nll(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    epsilon = 1e-15  # to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    nll = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return nll

def train(
    theta: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epoch: int, 
    learning_rate: float
) -> tuple:
    train_nll_history = []
    val_nll_history = []
    
    for epoch in range(num_epoch):
        for i in range(len(X_train)):
            error = sigmoid(np.dot(X_train[i], theta)) - y_train[i]
            for j in range(len(theta)):
                gradient = error * X_train[i][j]
                theta[j] = theta[j] - learning_rate * gradient
        
        # Calculate NLL for training and validation sets
        train_predictions = sigmoid(np.dot(X_train, theta))
        val_predictions = sigmoid(np.dot(X_val, theta))
        train_nll = calculate_nll(train_predictions, y_train)
        val_nll = calculate_nll(val_predictions, y_val)
        train_nll_history.append(train_nll)
        val_nll_history.append(val_nll)
    
    return (train_nll_history, val_nll_history)

def predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    predictions = sigmoid(np.dot(X, theta))
    return predictions >= 0.5

def compute_error(y_pred: np.ndarray, y: np.ndarray) -> float:
    return np.mean(y_pred != y)
    
def load_formatted_data(file_path: str) -> tuple:
    labels = []
    features = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            components = line.split('\t')
            labels.append(components[0])
            features_vector = [float(val) for val in components[1:]]
            features.append(features_vector)
    
    labels = np.array(labels, dtype=float)
    features = np.array(features)
    return (labels, features)

def write_metrics(file_path: str, train_error: float, test_error: float):
    with open(file_path, 'w') as file:
        file.write(f"error(train): {train_error:.6f}\n")
        file.write(f"error(test): {test_error:.6f}\n")

def write_predictions(file_path: str, predictions: np.ndarray):
    with open(file_path, 'w') as file:
        for prediction in predictions:
            file.write(f"{int(prediction)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float, help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    # Load the data
    train_labels, train_features = load_formatted_data(args.train_input)
    validation_labels, validation_features = load_formatted_data(args.validation_input)
    test_labels, test_features = load_formatted_data(args.test_input)

    # Augment features with intercept term
    train_features = np.insert(train_features, 0, 1, axis=1)
    validation_features = np.insert(validation_features, 0, 1, axis=1)
    test_features = np.insert(test_features, 0, 1, axis=1)

    # Initialize theta
    theta = np.zeros(train_features.shape[1])

    # Train the model and get NLL history
    train_nll_history, val_nll_history = train(
        theta, 
        train_features, 
        train_labels.astype(int), 
        validation_features,
        validation_labels.astype(int),
        args.num_epoch, 
        args.learning_rate
    )

    # Plot NLL history
    epochs = range(1, args.num_epoch + 1)
    plt.plot(epochs, train_nll_history, label='Training NLL')
    plt.plot(epochs, val_nll_history, label='Validation NLL')
    plt.xlabel('Epochs')
    plt.ylabel('Negative Log-Likelihood')
    plt.legend()
    plt.title('Training and Validation NLL per Epoch')
    plt.show()

    # Make predictions
    train_predictions = predict(theta, train_features)
    validation_predictions = predict(theta, validation_features)
    test_predictions = predict(theta, test_features)

    # Compute and write errors
    train_error = compute_error(train_predictions, train_labels.astype(int))
    validation_error = compute_error(validation_predictions, validation_labels.astype(int))
    test_error = compute_error(test_predictions, test_labels.astype(int))
    write_metrics(args.metrics_out, train_error, test_error)

    # Write predictions
    write_predictions(args.train_out, train_predictions)
    write_predictions(args.test_out, test_predictions)
# python3 feature.py largedata/train_large.tsv largedata/val_large.tsv largedata/test_large.tsv glove_embeddings.txt largeoutput/formatted_train_large.tsv largeoutput/formatted_val_large.tsv largeoutput/formatted_test_large.tsv
# python3 lr.py  largeoutput/formatted_train_large.tsv  largeoutput/formatted_val_large.tsv \largeoutput/formatted_test_large.tsv largeoutput/formatted_train_labels.txt largeoutput/formatted_test_labels.txt largeoutput/formatted_metrics.txt 500  0.1
# python3 feature.py smalldata/train_small.tsv smalldata/val_small.tsv smalldata/test_small.tsv glove_embeddings.txt smalloutput/formatted_train_small.tsv smalloutput/formatted_val_small.tsv smalloutput/formatted_test_small.tsv