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

#2016-02-18 Arora et al., 2016; Levy and Goldberg, 2014; Le and Mikolov, 2014

Le and Mikolov describe a model for sentences and documents that builds on Mikolov's previous work in word embeddings.  The distributed memory 
paragraph model just adds a new vector of parameters to the k-sized context window vectors, where each paragraph is represented by its own vector.  
The distributed bag of words model, alternatively, samples a window of text from the paragraph and then trains the paragraph vector to predict a word 
in the window, requiring only parameters for the paragraph vectors.

Levy and Goldberg derive a matrix factorization algorithm from the skip gram model with negative sampling.  They show that it is equivalent to 
factorizing a word-context matrix where the association is measured by PMI and shifted by log k.  They describe a new metric (shifted positive PMI),
where the negative entries are replaced by 0 to encourage sparseness and show improved results on some tasks using a symmetric SVD.

Arora et al. show interesting results on the linearity of word embeddings.  They show that word vectors can be represented as a sparse linear
combination of "sense" vectors, that indicate how that vector is used in context.  Also interesting are the results given for the atoms of discourse, 
which can be combined linearly to form meta-discourse vectors.

Discussion:

1) Why is it better in some cases (in Levy and Goldberg) to use a larger value for k?  Is this because of the importance of balancing the dataset?

# 2016-02-25 Bhattacharya and Dunson, 2012; Zhou et al., 2014

Bhattacharya and Dunson describe a new model for modeling high-dimensional categorical data with dependencies between variables.  
They create a simplex factor model with Dirichlet priors and derive an MCMC algorithm for inference.
They describe applications for DNA sequencing and show improvements over a random forest baseline.
Zhou et al describe improvements for sparse tensor factorization using a hierarchical model.  
They derive a Gibbs sampling algorithm for inference.
They show improvements over standard PARAFAC factorization for a gene sequencing dataset and a sociology dataset.

Discussion:

1) What are some examples of tensor representations for natural language data?  Interaction between events seems like it would be a good model for
tensor factorization (for example the GDELT data with two actors, actions, and time) 

# 2016-03-04 Hoffman et al., 2013, Ranganath et al., 2014

I found the paper by Hoffman et al to be well-written (maybe because it contains a full review of variational 
inference and seems to be written for non-statisticians).  I thought it was interesting how stochastic optimization
was motivated, in part by the fact that updating the global parameters is very inefficient under
traditional variational inference, where the entire data set needs to be observed before one update to the
global parameters.  Furthermore, the explanation for using the natural gradient makes sense.
In a machine learning context, for gradient descent the data are
either scaled to unit length or to have mean 0 and unit variance, depending on the application, but as far as I
understand it doesn't make sense to do this with probability distributions.

The second paper was a more difficult read for me, as it assumed some background knowledge, particularly for
Rao-Blackwellization.  I found the application very interesting, as it could apply to the model I am interested
in working with, which is not conditionally conjugate so would require a variational inference approach.

Discussion:
1) The second article mentions that using stochastic optimization to optimize the ELBO may have gradients with
variance too large to be useful, which I didn't follow.  Why is the reduction in variance needed?

# 2016-03-11 Taddy 2013; Taddy 2015

Taddy describes methods for multinomial inverse regression.  The general problem in inverse regression is to
determine the probability p(x|y) instead of p(y|x), and many dimensionality reduction methods are formulated
in this way.  In the 2013 paper, each element is drawn from a multinomial distribution where the parameters
are a linear combination of parameters.  These parameters are learned as part of the modeling resulting in 
dimensionality reduction.  On each coefficient he used a Laplace prior to enforce sparsity, which is not surprising.
However, the major difference in this model is learning the lambda parameters individually, which prevents
overpenalization.

The second paper describes a distributed method for multinomial inverse regression.  The improvement over the
previous work is to estimate the normalizer, which is usually costly, and where the counts for 
the multinomial distribution are estimated by factorizing into Poisson distributions.

# 2016-03-25 Gershman and Blei 2013; Neal, 2000 

Gershman and Blei introduce Bayesian nonparametric models by motivating the model selection problem, i.e. how many clusters or factors to use
(which I thought was a good start to the paper).  They then distinguish between a finite mixture model and a CRP mixture model.  
The next section describes latent factor models with motivation from a Bayesian perspective, in comparison to the common ML or MAP estimation methods.  
With Bayesian nonparametrics, the corresponding model is the Indian buffet process.
(I would have preferred more detail in this section with an example).

Neal begins by introducing DP mixture models.
I like that this paper describes a dirichlet process mixture model as equivalent to a finite mixture model with K approaching infinity.
The paper then describes Gibbs sampling for conditionally conjugate models and derives the conditional distributions
where the assumption of exchangeability makes the conditional distribution easier to calculate.  However, this leads to slow 
convergence, as the thetas can only change for one observation at a time.  Instead, they derive a collapsed Gibbs sampling
algorithm with the mixing proportions integrated out.

# 2016-04-01 Zhou and Carin 2013


