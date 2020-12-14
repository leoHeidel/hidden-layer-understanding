import numpy as np

def get_matrix(weights):
    '''
    Get a matrix out of the weights. 
    Convenient for Convolutionnal layers.
    take the weights as a numpy array.
    '''
    if len(weights.shape) == 4:
        #Convolution weights
        h,w,m,n = weights.shape
        return np.reshape(weights, (h*w*m, n)).T
    return weights.T

def get_matching(weights):
    """
    Compute a stable matching from the matching weigts
    using the stable marriage algorithm
    """
    size = weights.shape[0]
    singles_m = set(range(size))
    m = np.argsort(weights, axis=1)
    m_idx = np.zeros(size, dtype=np.int32) 
    w_idx = np.zeros(size, dtype=np.int32) - 1
    while singles_m:
        current_m  = singles_m.pop()
        current_w = m_idx[current_m]
        m_idx[current_m] = m_idx[current_m] + 1
        while w_idx[current_w] != -1 or weights[current_w, w_idx[current_w]] < weights[current_m, w_idx[current_w]]:
            current_w = m_idx[current_m]
            m_idx[current_m] = m_idx[current_m] + 1
        if w_idx[current_w] != -1:
            singles_m.add(w_idx[current_w])
        w_idx[current_w] = current_m
    return w_idx
        

def track_svd(weights, previous=None):
    """
    Track the svd of the weights as linear operation.
    If given a previous svd, ensure that the eigen vectors of
    the new svd match the previous by using a stable coupling.
    """

    mat = get_matrix(weights)