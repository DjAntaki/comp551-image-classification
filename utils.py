import theano
import numpy as np
from theano import tensor as T

def split_train_valid(data, train=0.6, shuffle=False):
    n_examples = len(data)
    if shuffle :
        from random import shuffle
        shuffle(data)

    split1 = int(train*n_examples)
    return data[:split1], data[split1:]

def get_2_argmax_func():
    test_input = np.array([1,2,3,4,5])
    inp1 = T.vector()
    max_ind = T.argmax(inp1)
    print(max_ind.eval({inp1:test_input}))
    max_ind2 = T.argmax(T.set_subtensor(inp1[max_ind],T.min(inp1)))
    #max_ind2 = T.argmax(T.concatenate([inp1[::max_ind],inp1[max_ind+1::]]))
    print(max_ind2.eval({inp1:test_input}))

    #fn = theano.function([inp1], max_ind)
    fn = theano.function([inp1], T.concatenate([max_ind, max_ind2]))
