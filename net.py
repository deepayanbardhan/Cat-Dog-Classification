import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras import optimizers
import pickle

import numpy as np
import os

def load_train(filepath):
    os.chdir(filepath)
    
    cat_train=[]
    dog_train=[]
    
    image=os.listdir()
    for img in image:
        X=cv2.imread(img)
        X=cv2.resize(X,(64,64))
        X=X/255.0
        if "cat" in img:
            cat_train.append(X)
        else:
            dog_train.append(X)
            
    cat_train=np.array(cat_train)
    dog_train=np.array(dog_train)
    
    return (cat_train,dog_train)
    
def load_test(filepath):
    os.chdir(filepath)
    image=os.listdir()
    
    test=[]
    
    for img in image:
        X=cv2.imread(img)
        X=cv2.resize(X,(64,64))
        X=X/255.0
        test.append(X)
    
    test=np.array(test)
    
    return (test)

def NN(cat,dog,test):
    
    #print (cat.shape) 8,30000
    xtrain=np.concatenate((cat,dog),0)
    ytrain=np.asarray([1 if i<len(cat) else 0 for i in range(len(cat) +len(dog))])
    xtest=test

    model=Sequential()
    model.add(Dense(64, activation="sigmoid",input_shape=xtrain.shape[1:]))
    model.add(Dense(16, activation="relu"))
    #model.add(Dense(8, activation="sigmoid"))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",optimizer="Adam",metrics=['accuracy'])
    
    model.fit(xtrain,ytrain,validation_split=0.1,batch_size=32,epochs=10,shuffle=True)
    
    outcomes=model.predict(xtest)
    ytest = [np.argmax(i) for i in outcomes]
    
    return (outcomes,ytest)

def CNN(cat,dog,test):
    
    #cat=np.reshape(cat,(cat.shape[0],64,64,3))
    #dog=np.reshape(dog,(dog.shape[0],64,64,3))
    xtrain=np.concatenate((cat,dog),0)
    ytrain=np.asarray([1 if i<len(cat) else 0 for i in range(len(cat)+len(dog))])
    #xtest=np.reshape(test,(test.shape[0],64,64,3))
    xtest=test
    indices=np.arange(xtrain.shape[0])
    np.random.shuffle(indices)
    xtrain=xtrain[indices]
    ytrain=ytrain[indices]
    
    
    model=Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(64,64,3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
   # model.add(BatchNormalization())
   # model.add(Dropout(rate=0.25))
    model.add(Conv2D(32,kernel_size=(3,3),activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
   # model.add(BatchNormalization())
   # model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    
    model.fit(xtrain,ytrain, batch_size=8, epochs=10, shuffle=1,validation_split=0.2)
    

    outcomes=model.predict(xtest)
    ytest=[np.argmax(i) for i in outcomes] #change to round-off
    
    return (outcomes,ytest)
    
train_path="D:\\deepayan\\study\\Coding\\kaggle\\dogs-vs-cats\\train_few"
#train_path='/mnt/C4B869A5B8699728/deepayan/study/Coding/kaggle/dogs-vs-cats/few'
#test_path='/mnt/C4B869A5B8699728/deepayan/study/Coding/kaggle/dogs-vs-cats/few'
test_path="D:\\deepayan\\study\\Coding\\kaggle\\dogs-vs-cats\\test_few"
cat,dog=load_train(train_path)
#with open('var.pickle','wb') as f:
#    pickle.dump(dog,f)
#with open('var.pickle','rb') as f:
#    dog=pickle.load(f)
test=load_test(test_path)
outcomes,ytest=CNN(cat,dog,test)

