"""
Uses a convnet trained on recognizing a single number and takes the sum of the two argmax.

Major problem : What if there is two time the same number on the image?

"""
from data.preprocess import get_data
import numpy as np
from keras.models import model_from_json
from skimage.transform import resize
from operator import itemgetter
# load json and create model
json_file = open('../trained_models/cnn1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("../trained_models/cnn1.h5")
print("Loaded model from disk")

X,Y = get_data()

num_example = len(X)

#Downsizing...
XX = np.zeros(shape=(num_example,1,28,28),dtype="float32")
for i,x in enumerate(X):
    XX[i] = resize(x,(28,28))

outs = loaded_model.predict(XX)

def argmax2(x):
    a = [(i,j) for i,j in enumerate(x)]
    a = sorted(a,key=itemgetter(1))
    return (a[0][0], a[1][0])

good = 0
wrong = 0

for x,y in zip(outs,Y):
    w = np.sum(argmax2(x))
    print(w,y)
    if w == y :
        good += 1
    else :
        wrong += 1

print(good,wrong)


#print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100)