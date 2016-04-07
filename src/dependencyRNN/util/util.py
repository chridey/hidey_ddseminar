import numpy as np

def matchEmbeddings(model, vocab, dimension=100):
    embeddings = [None for i in range(len(vocab))]
    for index,word in enumerate(vocab):
        if word in model:
            embedding = model[word]
        elif word.endswith('_(passive)') and word.replace('_(passive)', '') in model :
            embedding = model[word.replace('_(passive)', '')]
        else:
            try:
                print('{} not found at index {}'.format(word, index))
            except UnicodeEncodeError:
                print('{} not found at index {}'.format('UNICODE', index))
            embedding = np.random.uniform(-2, -2, dimension)
        embeddings[vocab[word]] = embedding
            
    return np.array(embeddings)
