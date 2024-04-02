import os
import re
import time
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from Parseing import preprocessTokenizeDoc, read_documents, get_queries

# Function to encode embeddings for tokens
def encode_embeddings(tokenizer, tokens, model):
    # Tokenize input
    input_ids = tokenizer.batch_encode_plus(tokens, return_tensors="pt", pad_to_max_length=True)['input_ids']
    # Compute embeddings using RoBERTa model
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state
    # Average pooling of embeddings
    avg_embeddings = torch.mean(embeddings, dim=1).squeeze().numpy()
    return avg_embeddings

# Function to retrieve and rank documents based on query and document embeddings
def retrieve_and_rank_documents_with_BERT(q_embed, doc_embed, num_results):
    similarity_scores = {}
    for doc_id, document_embedding in doc_embed.items():
        similarity_scores[doc_id] = cosine_similarity([q_embed], [document_embedding])[0][0]
    # Sort documents by similarity scores and select top results
    ranked_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:num_results]
    return ranked_documents

# Function to embed documents and queries
def embed_docs_queries(query_path, doc_path, tokenizer, model):
    # Read documents and queries
    documents = read_documents(doc_path)
    queries = get_queries(query_path)

    # Embed documents
    document_embeddings = {}
    for doc_id, doc_data in documents.items():
        processed_doc = preprocessTokenizeDoc(doc_data['TEXT'])
        document_embedding = encode_embeddings(tokenizer, processed_doc, model)
        document_embeddings[doc_id] = document_embedding
    
    # Embed queries
    query_embeddings = {}
    for i, query in enumerate(queries, start=1):
        processed_query = preprocessTokenizeDoc(query)
        query_embed = encode_embeddings(tokenizer, processed_query, model)
        query_embeddings[i] = query_embed
    
    if not os.path.exists('roberta_res'):
        os.makedirs('roberta_res')

    # Save embeddings
    np.save('roberta_res/document_embeddings.npy', document_embeddings)
    np.save('roberta_res/query_embeddings.npy', query_embeddings)

# Main function
def main():
    startTime = time.time()
    
    print("Creating model and tokenizer...")
    # Load RoBERTa model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    # Check if embeddings already exist, if not, create them
    if not os.path.exists('roberta_res/document_embeddings.npy') or not os.path.exists('roberta_res/query_embeddings.npy'):
        print("No existing embeddings found. Creating embeddings...")
        embed_docs_queries('Queries.txt', 'coll', tokenizer, model) 

    print("Loading embeddings...")
    # Load embeddings from saved files
    document_embeddings = np.load('roberta_res/document_embeddings.npy', allow_pickle=True).item()
    query_embeddings = np.load('roberta_res/query_embeddings.npy', allow_pickle=True).item()

    print("Calculating and saving results...")
    # Calculate and save retrieval results
    with open('RoBERTa_results.txt', 'w') as file:
        for query_id, query_embed in query_embeddings.items():
            relevant_documents = retrieve_and_rank_documents_with_BERT(query_embed, document_embeddings, 1000)
            for rank, (docno, cossim) in enumerate(relevant_documents, start=1):
                line = f"{query_id} Q0 {docno} {rank} {cossim} run_name"
                file.write(line + '\n')

    endTime = time.time()
    runTime = endTime - startTime 
    print(str(runTime), "seconds taken")

# Execute main function if script is run directly
if __name__ == "__main__":
    main()
