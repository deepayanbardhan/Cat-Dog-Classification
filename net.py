import cv2
import tensorflow as tf
import numpy as np
import os

def load_train(filepath):
    os.chdir(filepath)
    
    cat_train=[]
    dog_train=[]
    
    image=os.listdir()
    for img in image:
        X=cv2.imread(img)
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
        X=X.flatten()/255.0
        test.append(X)
    
    test=np.array(np.matrix(test))
    
    return (test)

def CNN():
    train_path="C:\\Users\\Deepayan\\Downloads\\dogs-vs-cats\\train\\few"
    test_path="C:\\Users\\Deepayan\\Downloads\\dogs-vs-cats\\train\\few"
    cat,dog=load_train(train_path)
    test=load_test(test_path)
    xtrain=np.concatenate((cat,dog),1)
    ytrain=np.asarray([1 if i<len(cat) else 0 for i in range(len(cat) +len(dog))])
    xtest=test
    
    model=Sequential()
    model.add()
    model.add()
    model.add()
    model.compile()
    
    z=model.fit(xtrain,ytrain,validation_split=0.1)
    
    outcomes=model.predict(xtest)
    ytest = [np.argmax(i) for i in outcomes]
    
CNN()