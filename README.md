# Assignment 2

This project uses BERT and doc2vec models to create information retrieval systems.

## Group Information

**Students:**

- Ishan Phadte 300238878
- Lauren Gu 300320106
- Angus Leung 300110509

**Division of Work:**

- Ishan Phadte: BERT Model
- Lauren Gu: Report
- Angus Leung: Doc2Vec

## Instructions

## Functionality

#### Parsing

Before the information retrieval systems may be used, the documents and queries are preprocessed. The seperate documents in each file are extracted by identifying \<DOC> tags. The \<DOCNO> and \<TEXT> of each document are extracted.

### BERT

To prepare documents and queries for retrieval with the BERT model, preprocessing removes punctuation and stopwords and converts text to lowercase. Text is then tokenized using BertTokenizer from the transformers library. Then, text is encoded as follows:

```bash

```

### Doc2Vec

To prepare documents and queries for retireval with Doc2Vec, preprocessing removes punctuation and stopwords and converts text to lowercase. Text is then tokenized using nltk word_tokenize().

The Doc2Vec model we use is declared as follows:

```bash
model = Doc2Vec(vector_size=100, epochs=10)
```

where the vector_size dictates the dimensionality of the output vectos for documents and words, and epochs specifies the number of iterations over the corpus the model is to do during training.

The model is trained using on the documents from the corpus where it learns to generate embeddings for documents and words so that similar documents have similar embeddings in vector space.

To retrieve documents, Doc2Vec will then used what it has learned during training to infer an embedding of a document and a query. The cosine similarity between the two inferred embeddings is calcualted. A larger cosine similarity value indicates higher similarity between a query and docuemnt. For each query, the cosine similarity for each document is caluclated and ranked to provide a list of document results sorted by similarity.
