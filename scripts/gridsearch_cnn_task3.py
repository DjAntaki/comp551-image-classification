from data.preprocess import get_data
from models import convnets
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import utils
import numpy as np
import itertools

# Model configuration
## Not to change part
img_rows, img_cols = 60,60
num_channels = 1
input_shape = (num_channels, img_rows, img_cols)
nb_classes = 19

##Gridsearch part
nb_filters_options = [32]
pool_size_options = [(2, 2)] # size of pooling area for max pooling
kernel_size_options = [(3, 3)] # convolution kernel size
dropout_options = [0.4]
activation_options = ["relu"]

#Training configuration
##Same for all models part
num_examples = 100000
batch_size = 32
nb_epoch = 60
patience = 5

##Gridsearch part
optimizer_options = ['adadelta','adamax','RMSprop','nadam']
loss_function_options = ['categorical_crossentropy']

#Other configuration
savename = "cnn_task3_gridsearch_result1"

###
###

print("Retrieving and augmenting data...")
X,Y = get_data(num_examples)

#Splitting data in train and validation set
train_set, valid_set = utils.split_train_valid(zip(X,Y))
X_train, y_train = map(np.array,zip(*train_set))
X_valid, y_valid = map(np.array, zip(*valid_set))

#Reshaping data for convolution
X_train = X_train.reshape(X_train.shape[0], num_channels, img_rows, img_cols)
X_valid = X_valid.reshape(X_valid.shape[0], num_channels, img_rows, img_cols)

#Making sure its in float32 and normalizing values in [0,1] interval.
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_train /= 255
X_valid /= 255
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)

for nb_filters, pool_size, kernel_size, dropout, activation, optimizer, loss_function in itertools.product(nb_filters_options, pool_size_options, kernel_size_options, dropout_options, activation_options, optimizer_options, loss_function_options):

    print("Building model...")
    model = convnets.build_cnn(input_shape, nb_classes, kernel_size, pool_size, nb_filters, dropout,activation)

    print("Compiling loss function")
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    eas = EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')

    print("Training model...")
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_valid, Y_valid),callbacks=[eas])

    if False :
        print("Saving model...")
        # serialize weights to HDF5
        model.save_weights(savename+".h5")
        # serialize model to JSON
        model_json = model.to_json()
        with open(savename+".json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")