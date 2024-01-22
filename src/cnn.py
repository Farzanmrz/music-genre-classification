import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class CNN:
    def __init__(self):
        self.train_data = np.loadtxt(open('processed_dataset/train_data.csv'), delimiter=",")
        self.test_data = np.loadtxt(open('processed_dataset/test_data.csv'), delimiter=",")

        self.train_labels = np.loadtxt(open('processed_dataset/train_labels.csv'), delimiter=",")
        self.test_labels = np.loadtxt(open('processed_dataset/test_labels.csv'), delimiter=",")

        self.cnn_model = tf.keras.Sequential([

            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

    def train_and_predict(self):
        print("[PROCESS] Training CNN classifier and gathering prediction accuracy...")
        self.cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        hist = self.cnn_model.fit(self.train_data, self.train_labels,
                                  validation_data=(self.test_data, self.test_labels),
                                  epochs=110, verbose=0)

        loss, accuracy = self.cnn_model.evaluate(self.test_data, self.test_labels, batch_size=128)

        # plot to find the appropriate value of epoch
        epochs = range(1, 111)
        plt.plot(epochs, hist.history['accuracy'])
        plt.title('Epochs vs Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

        print(f"[SCORE] CNN classifier correctly classified {round(accuracy * 100, 2)}% of testing instances\n")
        return accuracy