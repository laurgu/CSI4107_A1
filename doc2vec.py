# Import packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from Parseing import read_documents, get_queries,preprocessTokenizeDoc

# #import nltk
# #nltk.download('punkt')
# #nltk.download('stopwords')

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import re

import xml.etree.ElementTree as ET

import time 

from sklearn.metrics.pairwise import cosine_similarity
def main():
    print("Importing docs...")
    docs = read_documents('coll') 

    # Tokenize and preprocess docs
    print("Preprocessing and tokenizing docs...")
    taggedDocs = []
    for doc in docs.values():
        docTokens = preprocessTokenizeDoc(doc['TEXT'])
        taggedDocs.append(TaggedDocument(words=docTokens, tags=[doc['DOCNO']]))
    print("Number of docs:", len(taggedDocs))

    # Create model
    print("Training model...")
    model = Doc2Vec(vector_size=200, epochs=10)
    model.build_vocab(taggedDocs)
    model.train(taggedDocs, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("doc2vec_model")

    # Load pre-trained model
    # print("Loading pre-trained model...")
    # model = Doc2Vec.load("doc2vec_model")

    # Compute document vectors
    print("Computing document vectors...")
    doc_vecs = {}
    counter = 1
    for docno, doc in docs.items():
        if counter < 10:
            print(doc['TEXT'])
        docVector = model.infer_vector(preprocessTokenizeDoc(doc['TEXT']))
        doc_vecs[docno] = docVector
        counter += 1

    # Compute query vectors
    print("Computing query vectors...")
    queries = get_queries("Queries.txt")
    query_vecs = {}
    for counter, query in enumerate(queries, start=1):
        queryVector = model.infer_vector(preprocessTokenizeDoc(query))
        query_vecs[counter] = queryVector

    # Retrieve and write results
    print("Retrieving and writing results...")
    with open('doc2vec_results.txt', 'w') as file:
        start_time = time.time()
        for counter, queryVector in query_vecs.items():
            results = {}
            for docno, docVector in doc_vecs.items():
                cossim = cosine_similarity([queryVector], [docVector])[0][0]
                results[docno] = cossim
            rankings = sorted(results.items(), key=lambda item: item[1], reverse=True)[:1000]
            for rank, (docno, cossim) in enumerate(rankings, start=1):
                line = f"{counter} Q0 {docno} {rank} {cossim} run_name"
                file.write(line + '\n')
        end_time = time.time()
        run_time = end_time - start_time
        print("Retrieval time:", run_time, "seconds")

if __name__ == "__main__":
    main()
