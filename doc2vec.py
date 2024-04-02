# Import necessary packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from Parseing import read_documents, get_queries, preprocessTokenizeDoc
import os
import json
import time 
from sklearn.metrics.pairwise import cosine_similarity

# Function to save vectors to a JSON file
def save_vectors_to_file(vectors, filename):
    if not os.path.exists('doc2vec_res'):
        os.makedirs('doc2vec_res')
    vectors_converted = {key: vector.tolist() for key, vector in vectors.items()}
    with open(filename, 'w') as file:
        json.dump(vectors_converted, file)

# Function to load vectors from a JSON file
def load_vectors_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    
# Main function
def main():
    # Initialize empty list for tagged documents
    taggedDocs = []
    
    # Start timing
    start_time = time.time()
    
    # Check if pre-trained model exists, if not, train the model
    if not os.path.exists('doc2vec_res/doc2vec_model'):
        print("No existing model found. Training model...")
        docs = read_documents('coll')

        # Preprocess documents and create TaggedDocument objects
        for doc in docs.values():
            docTokens = preprocessTokenizeDoc(doc['TEXT'])
            taggedDocs.append(TaggedDocument(words=docTokens, tags=[doc['DOCNO']]))

        # Train Doc2Vec model
        print("Training model...")
        model = Doc2Vec(window=10, alpha=0.001, workers=4)
        model.build_vocab(taggedDocs)
        model.train(taggedDocs, total_examples=model.corpus_count, epochs=model.epochs)
        
        # Save trained model
        model.save('doc2vec_res/doc2vec_model')

    # Load pre-trained model
    print("Loading pre-trained model...")
    model = Doc2Vec.load('doc2vec_res/doc2vec_model')

    # Check if document and query vectors already exist, if not, compute them
    if not os.path.exists('doc2vec_res/doc_vectors.json') or not os.path.exists('doc2vec_res/query_vecs.json'):
        print("No existing document and query vectors found.")

        # If taggedDocs is empty, read and preprocess documents
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
        # Save document vectors to a file
        save_vectors_to_file(doc_vecs, 'doc2vec_res/doc_vectors.json')

        # Compute query vectors
        print("Computing query vectors...")
        queries = get_queries("Queries.txt")
        query_vecs = {}
        for counter, query in enumerate(queries, start=1):
            queryVector = model.infer_vector(preprocessTokenizeDoc(query))
            query_vecs[counter] = queryVector
        # Save query vectors to a file
        save_vectors_to_file(query_vecs, 'doc2vec_res/query_vecs.json')

    # Load document and query vectors from files
    print("Loading vectors...")
    doc_vecs = load_vectors_from_file('doc2vec_res/doc_vectors.json')
    query_vecs = load_vectors_from_file('doc2vec_res/query_vecs.json')

    # End timing
    end_time = time.time()
    print("Training + Calculate Vectors Runtime: ", (end_time-start_time), " seconds")
    
    # Retrieve and write retrieval results
    print("Retrieving and writing results...")
    with open('doc2vec_results.txt', 'w') as file:
        start_time = time.time()
        for counter, queryVector in query_vecs.items():
            results = {}
            for docno, docVector in doc_vecs.items():
                # Compute cosine similarity between query and document vectors
                cossim = cosine_similarity([queryVector], [docVector])[0][0]
                results[docno] = cossim
            # Sort results by cosine similarity and write to file
            rankings = sorted(results.items(), key=lambda item: item[1], reverse=True)[:1000]
            for rank, (docno, cossim) in enumerate(rankings, start=1):
                line = f"{counter} Q0 {docno} {rank} {cossim} run_name"
                file.write(line + '\n')
        end_time = time.time()
        run_time = end_time - start_time
        print("Retrieval time:", run_time, "seconds")

# Execute main function if script is run directly
if __name__ == "__main__":
    main()
