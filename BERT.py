import os
import re
import time
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from Parseing import preprocessTokenizeDoc, read_documents, expand_query, get_queries

def encode_embeddings(tokens, tokenizer, model):
    tokens = tokens[:510]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    input_ids = input_ids + [tokenizer.pad_token_id] * (512 - len(input_ids)) if len(input_ids) < 512 else input_ids[:512]
    input_tensor = torch.tensor(input_ids).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        embeddings = outputs[0]
    avg_embeddings = torch.mean(embeddings, dim=1).squeeze().numpy()
    
    return avg_embeddings

def retrieve_and_rank_documents_with_BERT(q_embed, doc_embed, num_results):
    similarity_scores = {}
    for doc_id, document_embedding in doc_embed.items():
        similarity_scores[doc_id] = cosine_similarity([q_embed], [document_embedding])[0][0]
    ranked_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:num_results]
    return ranked_documents

def embed_docs_queries(query_path, doc_path, tokenizer, model):
    documents = read_documents(doc_path)
    queries = get_queries(query_path)

    document_embeddings = {}
    for doc_id, doc_data in documents.items():
        processed_doc = preprocessTokenizeDoc(doc_data['TEXT'])
        document_embedding = encode_embeddings(processed_doc, tokenizer, model)
        document_embeddings[doc_id] = document_embedding
    
    query_embeddings = {}
    for i, query in enumerate(queries, start=1):
        processed_query = preprocessTokenizeDoc(expand_query(query))
        query_embed = encode_embeddings(processed_query, tokenizer, model)
        query_embeddings[i] = query_embed
    
    np.save('document_embeddings.npy', document_embeddings)
    np.save('query_embeddings.npy', query_embeddings)

def main():
    startTime = time.time()
    
    print("Creating model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
    model = BertModel.from_pretrained('prajjwal1/bert-tiny', ignore_mismatched_sizes=True)

    if not os.path.exists('document_embeddings.npy') or not os.path.exists('query_embeddings.npy'):
        print("No existing embeddings found. Creating embeddings...")
        embed_docs_queries('Queries.txt', 'coll', tokenizer, model) 

    print("Loading embeddings...")
    document_embeddings = np.load('document_embeddings.npy', allow_pickle=True).item()
    query_embeddings = np.load('query_embeddings.npy', allow_pickle=True).item()

    print("Calculating and saving results...")
    with open('BERT_results.txt', 'w') as file:
        for query_id, query_embed in query_embeddings.items():
            relevant_documents = retrieve_and_rank_documents_with_BERT(query_embed, document_embeddings, tokenizer, model)
            for rank, (docno, cossim) in enumerate(relevant_documents, start=1):
                line = f"{query_id} Q0 {docno} {rank} {cossim} run_name"
                file.write(line + '\n')

    endTime = time.time()
    runTime = endTime - startTime 
    print(str(runTime), "seconds taken")

if __name__ == "__main__":
    main()
