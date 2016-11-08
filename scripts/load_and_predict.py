"""Load a trained model and make prediction with it"""

from data.preprocess import get_test_data
import numpy as np
from keras.models import model_from_json
from skimage.transform import resize
from operator import itemgetter

outputfile = "prediction1.csv"
#modelfile = '../trained_models/cnn_task3_attempt1.json'
#weightsfile = "../trained_models/cnn_task3_gridsearch_result1.9416833.h5"
modelfile = '../cnn_task3_gridsearch_result2.json'
weightsfile = "../cnn_task3_gridsearch_result2.h5"
modelfile = '../cnn_task3_gridsearch_result3.json'
weightsfile = "../cnn_task3_gridsearch_result3.h5"

# load json and create model
json_file = open(modelfile, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(weightsfile)
print("Loaded model from disk")

X = get_test_data()

X = X.astype('float32')
X /= 255
print(X.shape[0], 'samples')
# Reshaping data for convolution
X = X.reshape(X.shape[0], 1, 60, 60)
num_example = len(X)

outs = loaded_model.predict(X)
outs = np.argmax(outs,axis=1)
print(outs.shape)

with open(outputfile,'w') as w:
    w.write("Id,Prediction\n")

    for e, pred in enumerate(outs):
        w.write(str(e)+","+str(pred)+"\n")

    w.close()