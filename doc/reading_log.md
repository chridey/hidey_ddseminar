# 2016-01-29 Bengio et al., 2003

The main advantage of a neural language model (and all related deep learning models) is the improved ability to generalize to unseen data.
Other models often set aside some probability mass for out-of-vocabulary words and/or use some form of smoothing to assign probabilities to 
sequences never seen before.  In some cases this involves "backing off", using a shorter sequence to predict the probability for a 
longer one.

Whereas many probabilistic models require discrete representations of words, representing a word by a vector 
allows the model to predict sequences of words never seen in the training data by using the neighbors of words it has seen.
