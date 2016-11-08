from data.preprocess import get_data, get_data_train_valid
from models import convnets
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
from models.keras_optimizer import get_optimizer
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
pool_size_options = [(2,2)]
#pool_size_options = [(2, 2),(3,3)] # size of pooling area for max pooling
kernel_size_options = [(4,4)] # convolution kernel size
#dropout_options = [0]
dropout_options = [0., 0.35, 0.5]
activation_options = ["relu"]
depth_options = [2,3]

#Training configuration
##Same for all models part
num_examples = 100000
n_perturbed = 30000
n_perturbed = 0
#num_examples = 100
batch_size = 32
nb_epoch = 80
early_stopping_patience = 4
split_lr_patience = 2
split_lr_factor = 0.2
split_lr_min = 0.0005

##Gridsearch part
optimizer_options = ['RMSprop']
loss_function_options = ['categorical_crossentropy']
learning_rate_options = [0.06,0.05,0.04]

#Other configuration
qwe = '6'
savename = "cnn_task3_gridsearch_result" +qwe

###
###

print("Retrieving and augmenting data...")
X_train, y_train , X_valid, y_valid = get_data_train_valid(num_examples,n_perturbed,threshold=254,show=True)

# Making sure its in float32 and normalizing values in [0,1] interval.
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_train /= 255
X_valid /= 255
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'test samples')
# Reshaping data for convolution
X_train = X_train.reshape(X_train.shape[0], num_channels, img_rows, img_cols)
X_valid = X_valid.reshape(X_valid.shape[0], num_channels, img_rows, img_cols)
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)

# Preparing stuff for gridsearch
#

hash_template = "learning_rate:{}, depth:{}, nb_filters:{}, kernel_size:{}, pool_size:{}, dropout:{}, activation:{},loss_function:{}, optimizer:{}"
#f = open("mlp_results_avg_embedding_glove",'w')
f = open("cnn_gridsearch_all_results"+qwe,'w')
best_model_yet = ""
best_accuracy_yet = float('-inf')


num_configs = np.prod(map(len,[learning_rate_options,depth_options,nb_filters_options, pool_size_options, kernel_size_options, dropout_options, activation_options, optimizer_options, loss_function_options]))

print(num_configs)
if num_configs > 80 :
    x = raw_input(str(num_configs)+" configurations are going to be tried. This could be long, do you wish to continue? (y/n)")
    if x.lower() != 'y':
        import sys
        sys.exit()
# Gridsearch
for learning_rate, depth, nb_filters, pool_size, kernel_size, dropout, activation, optimizer, loss_function in itertools.product(learning_rate_options,depth_options,nb_filters_options, pool_size_options, kernel_size_options, dropout_options, activation_options, optimizer_options, loss_function_options):

    print("Building model...")
    model = convnets.build_cnn(input_shape, nb_classes, depth, kernel_size, pool_size, nb_filters, dropout,activation)

    model_hash = hash_template.format(learning_rate, depth, nb_filters, kernel_size, pool_size, dropout,activation,loss_function,optimizer)

    print("Model Hash : "+model_hash)

    print("Compiling loss function")
    model.compile(loss=loss_function,
                  optimizer=get_optimizer(optimizer,learning_rate),
                  metrics=['accuracy'])

    eas = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=0, mode='auto')
    split_lr = ReduceLROnPlateau(monitor='val_loss', factor=split_lr_factor,
                      patience=split_lr_patience, min_lr=split_lr_min)
    history = History()

    print("Training model...")
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_valid, Y_valid),callbacks=[eas,split_lr,history])


    f.write(model_hash + "\n")
    f.write(", ".join(map(lambda i : i[0]+":"+str(i[1][-1]),list(history.history.items()))))
    f.write(str(history.history) + "\n")

    acc = history.history["val_acc"][-1]
    if acc > best_accuracy_yet:
        best_accuracy_yet = acc
        best_model_yet = model_hash
        print("New best model!")
        print("Model: ", best_model_yet)
        print("Score: ", best_accuracy_yet)
        print("Saving model...")
        # serialize weights to HDF5
        model.save_weights(savename+".h5")
        # serialize model to JSON
        model_json = model.to_json()
        with open(savename+".json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")
    else :
        print("best model yet:", best_model_yet)
        print("best score yet:", best_accuracy_yet)

f.close()
