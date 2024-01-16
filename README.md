This code build with model VGG16 with changed Dense layers

Dense: 128, 64, 32 - relu

Out: 4 - sigm

U can run this code buy

python3 main.py [-option] [-option] [-option]

-l - load weights from ./weights.h5


-train - start train

-train-save - train and save weights as ./new-weights.h5

-train-save-show - same as up, but also showing 


-p - predict from DataSet/test
