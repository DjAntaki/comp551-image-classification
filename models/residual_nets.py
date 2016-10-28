"""
Residual neural network implementation in Lasagne
Source : https://github.com/alrojo/lasagne_residual_network
"""

# ##################### Build the neural network model #######################

# helper function for projection_b
def ceildiv(a, b):
    return -(-a // b)


def build_cnn(input_var=None, n=1, num_filters=8, cudnn='no'):
    import lasagne  # For some odd reason it can't read the global import, please PR/Issue if you know why
    projection_type = 'B'
    # Setting up layers
    if cudnn == 'yes':
        import lasagne.layers.dnn
        conv = lasagne.layers.dnn.Conv2DDNNLayer  # cuDNN
    else:
        conv = lasagne.layers.Conv2DLayer
    nonlin = lasagne.nonlinearities.rectify
    nonlin_layer = lasagne.layers.NonlinearityLayer
    sumlayer = lasagne.layers.ElemwiseSumLayer
    # batchnorm = BatchNormLayer.BatchNormLayer
    batchnorm = lasagne.layers.BatchNormLayer

    # Setting the projection type for when reducing height/width
    # and increasing dimensions.
    # Default is 'B' as B performs slightly better
    # and A requires newer version of lasagne with ExpressionLayer
    projection_type = 'B'
    if projection_type == 'A':
        expression = lasagne.layers.ExpressionLayer
        pad = lasagne.layers.PadLayer

    if projection_type == 'A':
        # option A for projection as described in paper
        # (should perform slightly worse than B)
        def projection(l_inp):
            n_filters = l_inp.output_shape[1] * 2
            l = expression(l_inp, lambda X: X[:, :, ::2, ::2],
                           lambda s: (s[0], s[1], ceildiv(s[2], 2), ceildiv(s[3], 2)))
            l = pad(l, [n_filters // 4, 0, 0], batch_ndim=1)
            return l

    if projection_type == 'B':
        # option B for projection as described in paper
        def projection(l_inp):
            # twice normal channels when projecting!
            n_filters = l_inp.output_shape[1] * 2
            l = conv(l_inp, num_filters=n_filters, filter_size=(1, 1),
                     stride=(2, 2), nonlinearity=None, pad='same', b=None)
            l = batchnorm(l)
            return l

    # helper function to handle filters/strides when increasing dims
    def filters_increase_dims(l, increase_dims):
        in_num_filters = l.output_shape[1]
        if increase_dims:
            first_stride = (2, 2)
            out_num_filters = in_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = in_num_filters

        return out_num_filters, first_stride

    # block as described and used in cifar in the original paper:
    # http://arxiv.org/abs/1512.03385
    def res_block_v1(l_inp, nonlinearity=nonlin, increase_dim=False):
        # first figure filters/strides
        n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> sum -> nonlin
        l = conv(l_inp, num_filters=n_filters, filter_size=(3, 3),
                 stride=first_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        if increase_dim:
            # Use projection (A, B) as described in paper
            p = projection(l_inp)
        else:
            # Identity shortcut
            p = l_inp
        l = sumlayer([l, p])
        l = nonlin_layer(l, nonlinearity=nonlin)
        return l

    # block as described in second paper on the subject (by same authors):
    # http://arxiv.org/abs/1603.05027
    def res_block_v2(l_inp, nonlinearity=nonlin, increase_dim=False):
        # first figure filters/strides
        n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)
        # BN -> nonlin -> conv -> BN -> nonlin -> conv -> sum
        l = batchnorm(l_inp)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(3, 3),
                 stride=first_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        if increase_dim:
            # Use projection (A, B) as described in paper
            p = projection(l_inp)
        else:
            # Identity shortcut
            p = l_inp
        l = sumlayer([l, p])
        return l

    def bottleneck_block(l_inp, nonlinearity=nonlin, increase_dim=False):
        # first figure filters/strides
        n_filters, first_stride = filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> nonlin -> conv -> BN -> sum
        # -> nonlin
        # first make the bottleneck, scale the filters ..!
        scale = 4  # as per bottleneck architecture used in paper
        scaled_filters = n_filters / scale
        l = conv(l_inp, num_filters=scaled_filters, filter_size=(1, 1),
                 stride=first_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=scaled_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(1, 1),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        if increase_dim:
            # Use projection (A, B) as described in paper
            p = projection(l_inp)
        else:
            # Identity shortcut
            p = l_inp
        l = sumlayer([l, p])
        l = nonlin_layer(l, nonlinearity=nonlin)
        return l

    # Bottleneck architecture with more efficiency (the post with Kaiming He's response)
    # https://www.reddit.com/r/MachineLearning/comments/3ywi6x/deep_residual_learning_the_bottleneck/
    def bottleneck_block_fast(l_inp, nonlinearity=nonlin, increase_dim=False):
        # first figure filters/strides
        n_filters, last_stride = filters_increase_dims(l_inp, increase_dim)
        # conv -> BN -> nonlin -> conv -> BN -> nonlin -> conv -> BN -> sum
        # -> nonlin
        # first make the bottleneck, scale the filters ..!
        scale = 4  # as per bottleneck architecture used in paper
        scaled_filters = n_filters / scale
        l = conv(l_inp, num_filters=scaled_filters, filter_size=(1, 1),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=scaled_filters, filter_size=(3, 3),
                 stride=(1, 1), nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        l = batchnorm(l)
        l = nonlin_layer(l, nonlinearity=nonlin)
        l = conv(l, num_filters=n_filters, filter_size=(1, 1),
                 stride=last_stride, nonlinearity=None, pad='same',
                 W=lasagne.init.HeNormal(gain='relu'))
        if increase_dim:
            # Use projection (A, B) as described in paper
            p = projection(l_inp)
        else:
            # Identity shortcut
            p = l_inp
        l = sumlayer([l, p])
        l = nonlin_layer(l, nonlinearity=nonlin)
        return l

    res_block = res_block_v1

    # Stacks the residual blocks, makes it easy to model size of architecture with int n
    def blockstack(l, n, nonlinearity=nonlin):
        for _ in range(n):
            l = res_block(l, nonlinearity=nonlin)
        return l

    # Building the network
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    # First layer! just a plain convLayer
    l1 = conv(l_in, num_filters=num_filters, stride=(1, 1),
              filter_size=(3, 3), nonlinearity=None, pad='same')
    l1 = batchnorm(l1)
    l1 = nonlin_layer(l1, nonlinearity=nonlin)

    # Stacking bottlenecks and increasing dims! (while reducing shape size)
    l1_bs = blockstack(l1, n=n)
    l1_id = res_block(l1_bs, increase_dim=True)

    l2_bs = blockstack(l1_id, n=n)
    l2_id = res_block(l2_bs, increase_dim=True)

    l3_bs = blockstack(l2_id, n=n)

    # And, finally, the 10-unit output layer:
    network = lasagne.layers.DenseLayer(
        l3_bs,
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network