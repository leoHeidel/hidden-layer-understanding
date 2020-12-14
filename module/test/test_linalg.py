import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import module

def test_get_matrix():
    layer = keras.layers.Conv2D(5, (2,3))
    x = tf.zeros([1,11,13,17])
    x = layer(x)
    shape = module.linalg.get_matrix(layer.weights[0].numpy()).shape
    assert shape  == (5,2*3*17)

def test_get_matching():
    w = np.array([
        [0,1,2,3],
        [10,0,1,2],
        [4,4,4,3],
        [10,10,10,0],
    ])
    matching_1 = module.linalg.get_matching(w)    
    matching_2 = module.linalg.get_matching(w.T)

    assert matching_1[3] == 3
    assert matching_2[3] == 3

    for i in range(len(w)):
        assert i in matching_1
        assert i in matching_2
    