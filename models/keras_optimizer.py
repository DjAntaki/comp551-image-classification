def get_optimizer(optimizer,learning_rate = None):
    """
    Quick access to keras optimizer.

    TODO : add options (clipnorm and clipvalue)
    """
    from keras.optimizers import Adadelta, SGD, RMSprop, Adagrad, Adam, Adamax, Nadam

    if optimizer == 'Adadelta':
        if learning_rate is None:
            opt = Adadelta()
        else:
            opt = Adadelta(lr=learning_rate)
    elif optimizer == 'SGD':
        if learning_rate is None:
            opt = SGD(lr=learning_rate)
        else:
            opt = SGD()
    elif optimizer == 'RMSprop':
        if learning_rate is None:
            opt = RMSprop(lr=learning_rate)
        else:
            opt = RMSprop()
    elif optimizer == 'Adagrad':
        if learning_rate is None:
            opt = Adagrad(lr=learning_rate)
        else:
            opt = Adagrad()
    elif optimizer == 'Adam':
        if learning_rate is None:
            opt = Adam(lr=learning_rate)
        else:
            opt = Adam()
    elif optimizer == 'Adamax':
        if learning_rate is None:
            opt = Adamax(lr=learning_rate)
        else:
            opt = Adamax()
    elif optimizer == 'Nadam':
        if learning_rate is None:
            opt = Nadam(lr=learning_rate)
        else:
            opt = Nadam()
    else:
        print('Optimizer {} not defined, using Adadelta'.format(optimizer))
        opt = Adadelta(lr=learning_rate)
    return opt