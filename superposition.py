from keras.models import Sequential
from keras.layers import Dense, Activation,convolutional,pooling,core
import keras
from keras.datasets import  cifar10
from keras.models import load_model

if __name__ == "__main__":
    (X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
    Y_train=keras.utils.to_categorical(Y_train)
    Y_test=keras.utils.to_categorical(Y_test)
    print (X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    model=Sequential()
    model = load_model('my_model_1.h5')

    for i in range(5):
        print (i)
        print ("...........................................................................")
        for k in range(0,50000,32):
            X_data=X_train[k:k+32]/255
            Y_data=Y_train[k:k+32]
            l_a=model.train_on_batch(X_data,Y_data)
            print (k)
            print (l_a)

    print (model.metrics_names)

    loss_and_metrics = model.evaluate(X_test, Y_test)
    print ("")
    print (loss_and_metrics)

    model.save('my_model_1.h5')