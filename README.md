# Assignment 2

This project uses BERT and doc2vec models to create information retrieval systems.
EXPLANATION OF WHAT BOTH MODELS DO!!!

## Group Information

**Students:**

- Ishan Phadte 300238878
- Lauren Gu 300320106
- Angus Leung 300110509

**Division of Work:**

- Ishan Phadte: BERT
- Lauren Gu: Report, BERT and Doc2Vec edits
- Angus Leung: Doc2Vec, BERT

## Instructions

Install dependencies by executing:

```bash
python install_dependencies.py
```

### BERT

Run BERT using:

```bash
python BERT.py
```

Results can be found in a document called "BERT_results.txt"

### Doc2Vec

Run Doc2Vec using:

```bash
python Doc2Vec.py
```

Results will be found in a document called "doc2vec_results.txt"

## About the Models

### BERT

BERT (Bidirectional Encoder Representations from Transformers) is a pretrained model that utilizes unsupervised learning techniques. During pretraining, it is trained to predict words based on their surrounding context and to determine whether two sentences follow each other in a text. BERT considers the context to the left and right of a word making it bidirectional. This allows it to capture rich semantic relationships.

BERT encodes text using two main mechanisms:

1. Self-Attention Mechanism: This mechanism allows BERT to weigh the importance of each word of an input based on its context. It applies these weightings to all words in a sequence capturing long-range relationships between words.

2. Feed-Forward Mechanism: After processing the input sequence through self-attention layers, BERT applies feed-forward neural network layers to further refine the representations learned from the self-attention mechanism. These layers apply nonlinear transformations to the weighted representations.

After training, BERT can be used to generate word embeddings that capture the semantic meaning and importance of words in their context. The resulting embeddings can be used to retrieve relevant documents.

### Doc2Vec

Doc2Vec:

Doc2Vec is an unsupervised learning model that generates vector representations of text. There are two variations of Doc2Vec: distrbuted memory (DM) and distributed bad of words (DBOW). In this project, we use the ditributed memory version, so this introduction will focus on this implementation. The DM variant learns to predict what a word should be based on its context in a paragraph and the paragraph vector. Doc2Vec is trained by iterating over the corpus several time. Each iteration, it learns the context of words. To retrieve documents, Doc2Vec applies the training to predict a vector embedding of the query and documents then uses, in our implementation, the cosine similarity between the documents and vectors to identify relevant documents.

## Functionality

### Parsing

Before the information retrieval systems may be used, the documents and queries are preprocessed. The seperate documents in each file are extracted by identifying \<DOC> tags. The \<DOCNO> and \<TEXT> of each document are extracted.

### BERT

#### **Preprocessing**

To prepare documents and queries for retrieval with the BERT model, preprocessing removes punctuation and stopwords applies porter stemming, and converts text to lowercase. Documents are tokenized using BertTokenizer from the transformers library. The tokens list must be 510 tokens long, so lists longer than this are truncated and lists shorter than this are padded.

We use the pretrained tiny bert model "prajjwal1/bert-tiny" which is a derivative of the original Google BERT model that is more lightweight as we found using the original BERT model produced runtimes that were too long. The model creates embeddings of both the documents and queries. To reduce runtime, these embeddings are then saved locally so that future runs may load the document and query embeddings rather than recalculating them.

#### **Retrieval**

To retrieve documents, we calculate the cossine similarity between the the embeddings created in preprocessing are rank the documents from most to least similar.

### Doc2Vec

#### **Preprocessing**

To prepare documents and queries for retrieval with the BERT model, preprocessing removes punctuation and stopwords applies porter stemming, and converts text to lowercase. Text is then tokenized using nltk word_tokenize().

The Doc2Vec model we use is declared as follows:

```bash
model = Doc2Vec(vector_size=100, epochs=10)
```

where the vector_size dictates the dimensionality of the output vectors for documents and queries, and epochs specifies the number of iterations over the corpus the model is to do during training. We save the model locally so that it does not need to be retrained each run in order to reduce runtime.

The model is trained using on the documents from the corpus where it learns to generate embeddings for documents and words so that similar documents have similar embeddings in vector space. While ideally, a model should learn on training data and be tested on seperate testing data, in this particular case, it was not possible to use different data for testing and training as we were only given one collection of documents. However, it is generally not acceptable to use the same data for testing and training as this will produce inflated positive retrieval results.

#### **Retrieval**

To retrieve documents, Doc2Vec will use what it has learned during training to infer an embedding of a document and a query. We save the vector embeddings for the documents and queries so that subsequent runs do not have to recalculate these embeddings, reducing runtime. The cosine similarities between a query and the document infered vectors are calculated to determine relevant documents.

## Results and Analysis

To evaluate the performance of the models, we will use trec eval and we will compare runtimes.
