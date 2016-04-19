# 2016-01-28

The goal of this project is to develop predictive models for causality from text.

The data was obtained from Simple and English Wikipedia by aligning paraphrases of sentences that contain explicit causal markers
and bootstrapping using an SVM with lexical semantic features to identify additional causal relations.  In total, there are
9190 causal examples and many negative examples (which could also be obtained by sampling).

Some examples of causal sentences include:
#####Hurricane Kathleen causes death and destruction in California and Arizona.
#####The explosion made people evacuate the building.

The plan for the project is to learn event embeddings and predictions using a deep learning model 
(initially with pre-trained detected events, possibly later with joint event detection and prediction).

# 2016-02-04

The main goals of this project are to jointly learn event embeddings and predictions of causality between events.
Often in NLP the terms "event" and "causality" are not well-defined.  Ideally we would have full logical representations
of events with predicates and arguments but that is not the case for this data.  However we can derive dependency parses
with high accuracy.  I will use dependency parses because events are verb-centric and the head of a dependency parse
is usually a verb.  Causal inference in this case is a learning problem, where we have positively labeled examples
of sentences containing events linked by an explicit marker.

I will examine recursive neural networks using dependency parses to model the compositionality of events.
The learning framework would be as a binary classifier where the output node is the sigmoid function:

$$ y = \sigma(e_1^T(t) W e_2(t) + \beta^T\phi) $$

e_1 is a compositional representation of the dependencies in phrase 1, W is a classification matrix, \phi represents
fine-grained features (traditional NLP features), and \beta is a vector of weights for those features.
Later work will incorporate global context embeddings into the objective function
(previous work by Hu and Walker on movie scenes indicates that the genre of a movie affects how accurate causal predictions are)

The plan for this project:

1. Improve and clean initial data set, create dependency parses
2. Create a recursive neural network for predicting causality between event pairs (see Ji and Eisenstein, 2015)
3. Incorporate global context features and additional fine-grained features
4. Evaluate model on other data sets such as the Penn Discourse Treebank

Summary of data:
- Because the data consists of paraphrases, we have multiple representations of the same event pairs, which we can leverage during training.
- One downside to using event pairs connected by explicit markers is that no one states obvious causal relationships.  
- In contrast to newswire text, Wikipedia articles also have human-labeled categories, which can be treated as a vector and used to infer global context embeddings to improve causal inference (see step 3)

# 2016-02-18

Much of the previous work in discourse classification includes pairs of words as features.  These pairs of words are derived from contexts (sometimes the Cartesian product from sentences in the training set or sometimes derived from a large corpus using explicit discourse connectives).  One problem with this approach is that the space of possible word pairs is very large and the space is discrete.  Continuous word representatations, however, may be able to generalize better.

Iyyer et al. created a dependency recursive neural network for question answering.  The parameters of their model are a $V x d$ dimensional word embedding matrix We, an $r x d x d$ tensor Wr where r is the number of dependency relations, a $d$ long bias vector, and a $d x d$ composition matrix Wv.  The embedding for each node in the parse tree is calculated as $h_n = f(W_v^T We(x) + b + \sum_{k \in K(n)} Wr(rel(n,k))^T h_k)$ where node k is a child of node n and f is the normalized tanh function.  They use a max margin objective function where each node in the tree is similar to a correct answer vector and dissimilar from wrong answers.

Yi and Eisenstein use recursive neural networks over constituent parses to predict discourse relations between sentences.  Their model uses binarized parse trees, so the model is then $h_n = f(W_v^T [h_{k1}; h_{k2}])$, where Wv is a $2d x d$ matrix, f is tanh, and k1 and k2 are the children of n.

Thus far I have implemented a model using a dependency recursive neural network with softmax prediction of 3 outputs: non-causal, reason, or result using the Theano framework (see code in src).  The model uses the dependency embeddings of Iyyer but involves an additional classification matrix between two dependency embeddings.

# 2016-02-22

Current Results:

epoch: 14 batch: 20 cost: 0.000204721649827 time: 19.6837859154

precision: [ 0.98676332  0.42857143  0.51333333] recall: [ 0.92414404  0.8597561   0.77      ] 

...
epoch: 49 batch: 40 cost: 4.4286021224e-05 time: 23.6845619678

precision: [ 0.98675914  0.43119266  0.50326797] recall: [ 0.92384888  0.8597561   0.77      ]

# 2016-02-25

Update: /proj/nlp crashed and CRF needs to find an outside vendor to see if they can recover the data.  The past few days I have been re-running
some data processing and cleanup before I can try any new models.

The current model performs relatively well for this task, without using any other features.  For the next step, I plan to take advantage
of the parallel aspect of the training data (each training point is a sentence from English Wikipedia or Simple Wikipedia, but every sentence is also
part of a paraphrase pair between a sentence from English Wikipedia and one from Simple Wikipedia).

The idea is to add a constraint to the model to maximize the similarity between pairwise embeddings ($\lambda_2 e11^Te12 + lambda_2 e21^Te22).

Another possibility is to create artificial training data by swapping the clauses between parallel sentences.

# 2016-03-03

This week I finished implementing updates to the model (adding the pairwise similarity constraint and the manually derived features).
I am still waiting on data recovery to test out improvements.

The downside of this model is that it is limited to labeled training data and observed structure (not causal or causal with direction)
so I spent some time researching and developing a generative model for the next step in the project.

I spent time reading over relevant research on causality detection.  There's an interesting paper in the psychology literature
about how humans learn a theory of causality (Goodman et al, 2010).  They create a hierarchical Bayesian model,
where they represent logical dependencies as Bayesian network structures based on Judea Pearl's theory of causation (representing
dependence, independence, intervention).  The generative story is then:
1) for every system s in S
		generate predicate A_s, relation R_s (directed connection between node)  uniformly from theory T
		for every trial t in T_s
			generate d_t from the conditional probability table R_theta with prior alpha on theta
However, they represent these events as discrete binary events and are able to use a beta-binomial model.
There are also a limited number of causal structures (543).
Any model with text would not be conditionally conjugate.  The next step is to research whether dynamic topic models might apply here.  

I also read over some of the theory by Judea Pearl theory on causality.
Here are some relevant blog posts:
http://www.michaelnielsen.org/ddi/if-correlation-doesnt-imply-causation-then-what-does/
http://www.michaelnielsen.org/ddi/guest-post-judea-pearl-on-correlation-causation-and-the-psychology-of-simpsons-paradox/

# 2016-03-05

Krishman and Eisenstein (2015) use social networks to predict terms of address formality.  The latent variables are 
hidden social network structure and the observed variables are interactions.  The latent network structure is parameterized with
a log-linear model and the output labels are multinomials conditioned on the hidden structure.
They apply mean field inference to determine the parameters.  This might work for causality detection, where the hidden structure is
a network between 3 or more variables and the outputs are the connectives between events.

# 2016-03-11

For the next stage I plan to implement a model to maximize PMI directly in order to learn event embeddings.  The current model requires 
a classification matrix in the final stage in order to determine the relationship between events.  Since we are trying to predict
causality, one possibility is to maximize the PMI between two events directly, similar to factorization of a PMI matrix for word embeddings.
Rather than maximizing $e_1(t)^T A e_t(2)$, the goal is to factorize $e_1(t)^T e_t(2)$ to minimize the difference between that dot product
and the PMI of discrete events.  For this, it is easier assume that the probability of an event factorizes into the probability of a predicate
and its arguments, where $p(e) = p(p) \pi_{a_i \in a_p} p(a_i | p) $ where the arguments are all independent of one another.

# 2016-03-25

The past 2 weeks have been spent
implementing the unsupervised algorithm described in the previous note.  
There were several steps involved in this process:

1. creating dependency parses of all of English Wikipedia (~4,900,000 articles)
2. calculating discrete probabilities for the formula in the previous note
	- Assuming PMI(e_1, e_2) = p(e_1, e_2)/(p(e_1)*p(e_2))
	- where p(e) = p(p) p(s | p) \pi_{a_i \in a_p} p(a_i | p)
	- and p(e_1, e_2) = p(p_1, p_2) p(s_1 | p_1, p_2) p(s_2 | p_1, p_2) \pi_{a_i \in a_{p_1} \cup a_{p_2}} p(a_i | p_1, p_2)
	- where p is a predicate, s is the subject of the predicate, and a_i is an argument of the predicate
3. modifying the dependency RNN to handle dot product of event embeddings without the classification matrix

Previous work identified the predicate and arguments from the dependency parse (Do et al, 2011; Guritkevich, 2008) 
where they determined predicates to be verbs or deverbal nouns.  These probabilities are calculated for all
pairs of events in an article (however, we use a 3 sentence window).
The deverbal nouns are determined heuristically.

# 2016-03-30

Currently working on implementing skip-gram with negative sampling for events.  Rather than sampling from a joint distribution
over events, I am assuming the event distribution factorizes as $p(e) = p(p) p(s | p) \pi_{a_i \in a_p} p(a_i | p)$, since
many events only occur once.
The events themselves are represented as $e = f(W_{subj} \dot x_{subj} + W_{pred} \dot x_{pred} + \sum_a W_a \dot x_a)$.

# 2016-03-31

Using the Wikipedia articles from September, 2015, there are X total events and Y unique events.
There are Z total words and A unique words, where a "word" is a word or multi-word phrase combined
using the relations "compound", "name", and "mwe" from the dependency parse.
Using the compounded Wikipedia corpus, I trained a word2vec model with a minimum count of 1 to initialize the
word embeddings for use in the event model.

Overall, the event similarity is more difficult than for word2vec as there are
fewer total events than words and more unique events than words.

Determining the correct window size will require some tuning.
For words, a 10 word window (5 in either direction) is enough to capture semantic and syntactic dependencies.
For events, a smaller window might be better.  Would we expect an event at the beginning of an article to directly affect an event much later on?

# 2016-04-04

(For related work)
Levy and Goldberg (2014) used dependency-based word embeddings where the context of a word is a relation/word pair for all of its children in a dependency parse.
They showed that using dependency contexts rather than bag of word contexts resulted in different measures of similarity.  For example, the most similar
vectors to 'florida' using BoW were 'jacksonville', 'tampa', 'fla', whereas using dependencies resulted in 'carolina' and 'california', for example.
They describe this as a difference between domain similarity and functional similarity.
For causality, we would expect functional similarity to be more useful as there is some sense of exchangeability in this context.

# 2016-04-06

Finished processing all the events into tuples of predicate, subject(s), object(s), and indirect object(s).

Implemented negative sampling for events - first sample a predicate, then sample all its arguments given the predicate.

# 2016-04-07

Implemented the shallow dependency RNN with negative sampling.

Initially, I was using the scan function in theano, which is very slow and not optimized and switched to the batched\_dot function which
works much better (from ~50 data per second to ~1000).

# 2016-04-14

Trained a model using the full vocabulary from Wikipedia (min count of 1).  This results in 119,000,000 events and 12,600,000 words.  Under this
scenario it takes about 6 seconds to sample 10,000 events and 30 seconds to train these events (slower than 1000 per second). 

For testing, I am using a dataset from 2011 where the researchers had annotated news articles for cause and effect between events.  Each event has its 
predicate marked and a pair of events is marked as either causal or related. 

The current trained model performs very poorly, around 10% precision and recall.  I saved several models at different stages of training and the
performance improves from 5% but seems to be converging very slowly.  Iterating over the entire dataset would take several days at this rate.  

The next steps will be to set the min count higher.  For a min count of 10, there are only about 600,000 words in the vocabulary, which is much faster
to train.  Furthermore I will also test using different contexts, in this case using the discourse connective and only considering those events
during training.
is to train 
