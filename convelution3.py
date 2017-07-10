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

    model.add(convolutional.Conv2D(filters=32,kernel_size=3,strides=1,padding="same",data_format="channels_last",input_shape=X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(pooling.MaxPool2D(pool_size=2,strides=2,padding="same",data_format="channels_first"))
    model.add(core.Dropout(0.2))

    model.add(convolutional.Conv2D(filters=48, kernel_size=3, strides=1, padding="same", data_format="channels_last", input_shape=X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(pooling.MaxPool2D(pool_size=2,strides=2,padding="same",data_format="channels_first"))
    model.add(core.Dropout(0.2))

    model.add(convolutional.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", data_format="channels_last",input_shape=X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(pooling.MaxPool2D(pool_size=2, strides=2, padding="same", data_format="channels_first"))
    model.add(core.Dropout(0.2))

    model.add(core.Flatten())
    model.add(Dense(units=512))
    model.add(Activation("relu"))
    model.add(core.Dropout(0.2))
    model.add(Dense(units=10))
    model.add(Activation("softmax"))

    opt=keras.optimizers.Adam(lr=0.0001,decay=1e-6)

    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])

    for i in range(15):
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