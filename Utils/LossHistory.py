import keras

class LossHistory(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        x, y = self.test_data
        loss, acc, cat_acc = self.model.evaluate(x, y, verbose=0)
        self.losses.append(loss)
        self.accuracies.append(acc)