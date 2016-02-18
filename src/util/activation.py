import theano.tensor as T

def normalized_tanh(x):
    s = T.tanh(x)
    return s / T.sqrt((s**2).sum())
        
