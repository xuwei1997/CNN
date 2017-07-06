import pickle
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pretreatment(data):
    X= data[b'data']
    Y=data[b'labels']
    Y=np.array(Y)
    return (X,Y)

if __name__ == "__main__":
    data=unpickle('/root/tf/cnn/cifar-10-batches-py/data_batch_1')
    X,Y=pretreatment(data)
    print (X,Y)
    print (X.size)
    print (Y.size)