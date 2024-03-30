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


# Import docs 
def main():
    print("Importing docs...")
    docs = read_documents('coll')   # Stores a dict with docno as key and has docno, head, dateline, text

    # Tokenize and preprocess docs
    print("Preprocessing and tokenizing docs...")
    taggedDocs = []

    for doc in docs.values():
        docTokens = preprocessTokenizeDoc(doc['TEXT'])
        taggedDocs.append(TaggedDocument(words=docTokens, tags=[doc['DOCNO']]))
        
    print("Number of docs:", str(len(taggedDocs)))

    start_time = time.time()
    # Create model
    print("Training model...")
    model = Doc2Vec(vector_size=100, epochs=10)
    model.build_vocab(taggedDocs)
    model.train(taggedDocs, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("doc2vec_model")

    end_time = time.time()

    print("Training time: ", end_time-start_time, " seconds")
        
    # model = Doc2Vec.load("doc2vec_model")
    # Tokenize and preprocess query
    queries = get_queries("Queries.txt")

    doc_vecs = {}

    # Compute document vectors
    for docno, doc in docs.items():
        docVector = model.infer_vector(preprocessTokenizeDoc(doc['TEXT']))
        doc_vecs[docno] = docVector

    start_time = time.time()
    print("Retrieving...")
    # Cosine Similarity calculations for list of docs vs query
    with open('doc2vec_results.txt', 'w') as file:
        for query in queries:
            results = {}  # Initialize results dictionary for each query
            queryVector = model.infer_vector(preprocessTokenizeDoc(query))
            
            for docno, docVector in doc_vecs.items():  # Iterate over document vectors
                cossim = cosine_similarity([queryVector], [docVector])[0][0]
                results[docno] = cossim
            
            # Sort documents by cosine similarity in descending order
            rankings = sorted(results.items(), key=lambda item: item[1], reverse=True)
            
            # Write the rankings to the file
            rank = 1
            for docno, cossim in rankings:
                line = f"1 Q0 {docno} {rank} {cossim} run_name"
                file.write(line + '\n')
                rank += 1

                if rank == 1001:
                    break

    end_time = time.time()
    run_time = end_time - start_time
    print("Retrieval time:", run_time, " seconds")

if __name__ == "__main__":
    main()
