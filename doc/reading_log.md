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

#2016-02-04 Mikolov et al., 2014; Pennington et al., 2014; Arora et al., 2015
Mikolov et al. describe improvements to their previous Skip-gram model in the form of negative sampling, subsampling, and collocation
determination (they also describe the Hierarchical Softmax as introduced by Morin and Bengio for language modeling).

The negative sampling technique changes the objective function to model the probability of a word/context pair
as a Bernoulli random variable.  The negative samples are provided as a contrast with the current word by using a 
different context, so that it is now maximizing the probability of word/context pairs in the corpus actually being in the corpus combined with word/context pairs not in the corpus not actually being in the corpus (k determines that there are k times as many negative
samples as true cooccurrences).

They also include a simple method for determining collocations based on PMI and run several passes over the training data to pick up longer n-grams.  They show that the additive compositionality property holds for phrases as well.  Some phrases are not compositional, especially named entities and it makes sense to model them as one unit (i.e. Boston Globe describes a newspaper, not a globe).

Pennington et al. derive a different word embedding algorithm based on the intuition that ratios of cooccurrence probabilities P_ik and P_jk should be high for words i, j, and k if k is more similar to i than j.  The result is essentially a weighted matrix factorization.
They also compare their model to skip-gram and other window-based methods and are able to derive their algorithm from the window-based approach.  The complexity of the model is often better than window-based methods when consuming the entire corpus at once.

Arora et al. derive a similar word embedding model from a generative perspective.  The resulting model is similar to GloVe but was derived theoretically rather than empirically.

Discussion:
1) What are the advantages/disadvantages to determinining collocations this way compared to other methods like NER?
It seems like phrases like 'Battle of Midway' are a problem for both systems.

2) Is there a theoretical justification for using the U^(3/4) distribution that appears in the work of both Pennington and Mikolov?

