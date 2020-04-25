import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D

import numpy as np
import os

def load_train(filepath):
    os.chdir(filepath)
    
    cat_train=[]
    dog_train=[]
    
    image=os.listdir()
    for img in image:
        X=cv2.imread(img)
        X=cv2.resize(X,(100,100))
        X=X.flatten()
        X=X/255.0
        if "cat" in img:
            cat_train.append(X)
        else:
            dog_train.append(X)
            
    cat_train=np.array(np.matrix(cat_train)) 
    dog_train=np.array(np.matrix(dog_train))
    
    return (cat_train,dog_train)
    
def load_test(filepath):
    os.chdir(filepath)
    image=os.listdir()
    
    test=[]
    
    for img in image:
        X=cv2.imread(img)
        X=cv2.resize(X,(100,100))
        X=X.flatten()/255.0
        test.append(X)
    
    test=np.array(np.matrix(test))
    
    return (test)

def NN(cat,dog,test):
    
    #print (cat.shape) 8,30000
    xtrain=np.concatenate((cat,dog),0)
    ytrain=np.asarray([1 if i<len(cat) else 0 for i in range(len(cat) +len(dog))])
    xtest=test

    model=Sequential()
    model.add(Dense(64, activation="sigmoid",input_shape=xtrain.shape[1:]))
    model.add(Dense(16, activation="sigmoid"))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",optimizer="Adam",metrics=['accuracy'])
    
    model.fit(xtrain,ytrain,validation_split=0.1,batch_size=32,epochs=10,shuffle=True)
    
    outcomes=model.predict(xtest)
    ytest = [np.argmax(i) for i in outcomes]
    
    return (outcomes,ytest)

def CNN(cat,dog,test):
    
    cat=np.reshape(cat,(cat.shape[0],100,100,3))
    dog=np.reshape(dog,(dog.shape[0],100,100,3))
    xtrain=np.concatenate((cat,dog),0)
    ytrain=np.asarray([1 if i<len(cat) else 0 for i in range(len(cat)+len(dog))])
    print(ytrain.shape)
    xtest=np.reshape(test,(test.shape[0],100,100,3))
    
    model=Sequential()
    model.add(Conv2D(64,kernel_size=3,activation="sigmoid",input_shape=(100,100,3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(128,kernel_size=3,activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32,activation="sigmoid"))
    model.add(Dense(1,activation="softmax"))
    model.compile(loss="binary_crossentropy",optimizer="Adam",metrics=['accuracy'])
    print(model.summary())
    model.fit(xtrain,ytrain, batch_size=32, epochs=1, shuffle=1)
    
    outcomes=model.predict(xtest)
    ytest=[np.argmax(i) for i in outcomes]
    
    return (outcomes,ytest)
    
train_path="C:\\Users\\Deepayan\\Downloads\\dogs-vs-cats\\train\\few"
#train_path='/mnt/C4B869A5B8699728/deepayan/study/Coding/kaggle/dogs-vs-cats/few'
#test_path='/mnt/C4B869A5B8699728/deepayan/study/Coding/kaggle/dogs-vs-cats/few'
test_path="C:\\Users\\Deepayan\\Downloads\\dogs-vs-cats\\train\\few"
cat,dog=load_train(train_path)
test=load_test(test_path)
outcomes,ytest=CNN(cat,dog,test)
