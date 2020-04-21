import cv2
import tensorflow as tf
import numpy as np

def load_train(filepath):
    os.chdir(filepath)
    cat_train=[] # cat[i]=rgb 2d image, so each index is 4 dimension
    dog_train=[]
    
    return (cat_train,dog_train)
    
def load_test(filepath):
    os.chdir(filepath)
    test=[] # cat[i]=rgb 2d image, so each index is 4 dimension
    
    return (test)

def CNN():
    train_path="C:\\Users\\Deepayan\\Downloads\\dogs-vs-cats\\train\few"
    test_path=""
    load_train(train_path)
    load_test(test_path)
    
    xtrain=
    ytrain=
    xtest=
    
    model=Sequential()
    model.add()
    model.add()
    model.add()
    model.compile()
    
    z=model.fit(xtrain,ytrain,validation_split=0.1)
    model.predict(xtest)
    