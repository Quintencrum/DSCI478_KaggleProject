import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import process_data as data


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


class LogisticRegression(tf.Module):

    def __init__(self):
        self.built = False

    def __call__(self, x, train=True):
        # Initialize the model parameters on the first call
        if not self.built:
            # Randomly generate the weights and the bias term
            rand_w = tf.random.uniform(shape=[x.shape[-1], 1], seed=478)
            rand_b = tf.random.uniform(shape=[], seed=478)
            self.w = tf.Variable(rand_w)
            self.b = tf.Variable(rand_b)
            self.built = True
        # Compute the model output
        z = tf.add(tf.matmul(x, self.w), self.b)
        z = tf.squeeze(z, axis=1)
        if train:
            return z
        return tf.sigmoid(z)

def predict_class(y_pred, thresh=0.5):
    return tf.cast(y_pred > thresh, tf.float32)

def accuracy(y_pred, y):
    # Return the proportion of matches between `y_pred` and `y`
    y_pred = tf.math.sigmoid(y_pred)
    y_pred_class = predict_class(y_pred)
    check_equal = tf.cast(y_pred_class == y,tf.float32)
    acc_val = tf.reduce_mean(check_equal)
    
    return acc_val

def log_loss(y_pred, y):
    # Compute the log loss function
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
    
    return tf.reduce_mean(ce)

def train(log_reg, xs_train, ys_train, xs_val, ys_val,
          batch_size = 10, epochs = 20, learning_rate = 0.01,
          plot=True):
    train_dataset = tf.data.Dataset.from_tensor_slices((xs_train, ys_train))
    train_dataset = train_dataset.shuffle(buffer_size=xs_train.shape[0]).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((xs_val, ys_val))
    val_dataset = val_dataset.shuffle(buffer_size=xs_val.shape[0]).batch(batch_size)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(epochs):
        batch_losses_train, batch_accs_train = [], []
        batch_losses_val, batch_accs_val = [], []

        # Iterate over the training data
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                y_pred_batch = log_reg(x_batch)
                batch_loss = log_loss(y_pred_batch, y_batch)
            batch_acc = accuracy(y_pred_batch, y_batch)
            # Update the parameters with respect to the gradient calculations
            grads = tape.gradient(batch_loss, log_reg.variables)
            for g,v in zip(grads, log_reg.variables):
                v.assign_sub(learning_rate * g)
            # Keep track of batch-level training performance
            batch_losses_train.append(batch_loss)
            batch_accs_train.append(batch_acc)

        # Iterate over the validation data
        for x_batch, y_batch in val_dataset:
            y_pred_batch = log_reg(x_batch)
            batch_loss = log_loss(y_pred_batch, y_batch)
            batch_acc = accuracy(y_pred_batch, y_batch)
            # Keep track of batch-level testing performance
            batch_losses_val.append(batch_loss)
            batch_accs_val.append(batch_acc)

        # Keep track of epoch-level model performance
        train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
        val_loss, val_acc = tf.reduce_mean(batch_losses_val), tf.reduce_mean(batch_accs_val)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if epoch % 20 == 0:
            train_dataset = tf.data.Dataset.from_tensor_slices((xs_train, ys_train))
            train_dataset = train_dataset.shuffle(buffer_size=xs_train.shape[0]).batch(batch_size)
            val_dataset = tf.data.Dataset.from_tensor_slices((xs_val, ys_val))
            val_dataset = val_dataset.shuffle(buffer_size=xs_val.shape[0]).batch(batch_size)
        
            print(f"""Epoch: {epoch}
                    Training log loss: {train_loss:.3f}
                    Validation log loss: {val_loss:.3f}""")
    else:
        print(f"""Epoch: {epoch}
                Training log loss: {train_loss:.3f}
                Validation log loss: {val_loss:.3f}""")

    if plot:
        make_plots(epochs, train_losses, val_losses, train_accs, val_accs)

    return train_losses, val_losses, train_accs, val_accs

def make_plots(epochs, train_losses, val_losses, train_accs, val_accs):
    plt.plot(range(epochs), train_losses, label = "Training loss")
    plt.plot(range(epochs), val_losses, label = "Testing loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log loss")
    plt.legend()
    plt.title("Log loss vs training iterations")
    plt.show()

    plt.plot(range(epochs), train_accs, label = "Training accuracy")
    plt.plot(range(epochs), val_accs, label = "Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy vs Validation iterations")
    plt.show()

def plot_confusion_matrix():
    pass

def main():
    xs_train, ys_train, xs_val, ys_val = data.main("train")

    xs_train = tf.convert_to_tensor(xs_train, dtype=tf.float32)
    ys_train = tf.convert_to_tensor(ys_train, dtype=tf.float32)
    xs_val = tf.convert_to_tensor(xs_val, dtype=tf.float32)
    ys_val = tf.convert_to_tensor(ys_val, dtype=tf.float32)
    
    log_reg = LogisticRegression()
    train_losses, val_losses, train_accs, val_accs = train(log_reg, xs_train, ys_train, xs_val, ys_val)

def main2():
    xs_train, ls_train, xs_val, ls_val = data.main("train")
    
    # Preprocess the data
    xs_train = xs_train.reshape(-1, 784).astype('float32')
    ls_train = tf.one_hot(ls_train, depth=10)
    xs_val = xs_val.reshape(-1, 784).astype('float32')
    ls_val = tf.one_hot(ls_val, depth=10)

    # Define the logistic regression model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=(28*28,), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(xs_train, ls_train, epochs=20, batch_size=32, validation_data=(xs_val, ls_val),
              shuffle=True)

    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(xs_val, ls_val)
    print(f'\nValidation loss: {loss}, Validation accuracy: {accuracy}')  


if __name__ == "__main__":
    main2()
