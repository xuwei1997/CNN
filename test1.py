import keras
from keras.models import Sequential
from keras.datasets import  cifar10
from keras.models import load_model

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    Y_test = keras.utils.to_categorical(Y_test)
    X_test = X_test[:10000] / 255
    Y_test = Y_test[:10000]

    model=Sequential()
    model = load_model('my_model_1.h5')
    loss_and_metrics = model.evaluate(X_test, Y_test)
    print ("")
    print (loss_and_metrics)