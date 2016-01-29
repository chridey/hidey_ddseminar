# 2016-01-29 Bengio et al., 2003

The main advantage of a neural language model (and all related deep learning models) is the improved ability to generalize to unseen data.
Other models often set aside some probability mass for out-of-vocabulary words and/or use some form of smoothing to assign probabilities to 
sequences never seen before.  In some cases this involves "backing off", using a shorter sequence to predict the probability for a 
longer one.

Whereas many probabilistic models require discrete representations of words, representing a word by a vector 
allows the model to predict sequences of words never seen in the training data by using the neighbors of words it has seen.

One disadvantage to this model is that by using a
fixed context width, the number of parameters scales by the vocabulary size for each additional word in the context window.

I found the future work section on energy minimization models interesting and I am curious if any of these methods are used, specifically
in regards to representing out-of-vocabulary words.

Discussion:

1) Why does it help to combine the neural language model with an interpolated trigram model?

2) What effect does merging rare words have and is there a better way of modeling infrequent words?

