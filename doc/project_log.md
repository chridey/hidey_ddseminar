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
