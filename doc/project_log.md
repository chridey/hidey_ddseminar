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
of the parallel aspect of the training data (each training point is a sentence from English Wikipedia or Simple Wikipedia, but 

The idea is to add a constraint to the model to maximize the similarity between pairwise embeddings ($\lambda_2 e11^Te12 + lambda_2 e21^Te22).

Another possibility is to create artificial training data by swapping the clauses between parallel sentences.

