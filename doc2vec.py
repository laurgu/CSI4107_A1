# Import packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from Parseing import read_documents, get_queries,preprocessTokenizeDoc, expand_query
import os
import json
import time 

from sklearn.metrics.pairwise import cosine_similarity

def save_vectors_to_file(vectors, filename):
    vectors_converted = {key: vector.tolist() for key, vector in vectors.items()}
    with open(filename, 'w') as file:
        json.dump(vectors_converted, file)

def load_vectors_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    
def main():

    taggedDocs = []
 
    if not os.path.exists('doc2vec_model'):
        print("No exisiting model found. Training model...")
        docs = read_documents('coll')

        for doc in docs.values():
            docTokens = preprocessTokenizeDoc(doc['TEXT'])
            taggedDocs.append(TaggedDocument(words=docTokens, tags=[doc['DOCNO']]))  

        print("No existing model found. Training model...")
        model = Doc2Vec(vector_size=100, epochs=10)
        model.build_vocab(taggedDocs)
        model.train(taggedDocs, total_examples=model.corpus_count, epochs=model.epochs)
        model.save("doc2vec_model")

    # Load pre-trained model
    print("Loading pre-trained model...")
    model = Doc2Vec.load("doc2vec_model")

    if not os.path.exists('doc_vectors.json') or not os.path.exists('query_vecs.json'):

        print("No existing document and query vectors found.")

        if not taggedDocs:
            docs = read_documents('coll')

            for doc in docs.values():
                docTokens = preprocessTokenizeDoc(doc['TEXT'])
                taggedDocs.append(TaggedDocument(words=docTokens, tags=[doc['DOCNO']]))  

        # Compute document vectors
        print("Computing document vectors...")        
        doc_vecs = {}

        for taggedDoc in taggedDocs:
            docVector = model.infer_vector(taggedDoc.words)
            doc_vecs[taggedDoc.tags[0]] = docVector

        save_vectors_to_file(doc_vecs, 'doc_vectors.json')

        # Compute query vectors
        print("Computing query vectors...")
        queries = get_queries("Queries.txt")
        query_vecs = {}
        for counter, query in enumerate(queries, start=1):
            queryVector = model.infer_vector(preprocessTokenizeDoc(query))
            query_vecs[counter] = queryVector
        
        save_vectors_to_file(query_vecs, 'query_vecs.json')

    print("Loading vectors...")
    doc_vecs = load_vectors_from_file('doc_vectors.json')
    query_vecs = load_vectors_from_file('query_vecs.json')

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
