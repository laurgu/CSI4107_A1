import os
import re
import nltk
import torch
import time
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import xml.etree.ElementTree as ET
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
from Parseing import read_documents, get_queries, preprocessTokenizeDoc

def preprocess_bert(text, tokenizer):
    text = re.sub(r'[^\w\s]', '', text)    
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in tokenizer.stop_words]

    return tokens

def prepare_input_for_bert(text, tokenizer, max_length=512):
    tokens = preprocess_bert(text, tokenizer)
   
    inputs = tokenizer.encode_plus(
        tokens,
        max_length=max_length,
        add_special_tokens=True, 
        return_tensors="pt",  
    )
    return inputs["input_ids"], inputs["attention_mask"]

def document_vectors(documents, tokenizer, model, max_length=512):
    doc_vectors = {}
    for docno, text in documents.items():
        # Preprocess text
        processed_text = preprocessTokenizeDoc(text['TEXT'])
        input_ids, attention_mask = prepare_input_for_bert(processed_text, tokenizer, max_length)

        # Use `model` to get document vector
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # Extract document vector (e.g., from pooled output)
            doc_vector = outputs[0][:, 0, :]  # Extract first token from pooled output

            doc_vectors[docno] = doc_vector

    return doc_vectors

def retrieve_with_bert(queries, documents, tokenizer, model, max_length=512):
    rankings = {}
    for query in queries:
        query_vector = document_vectors({query: query}, tokenizer, model, max_length)[query]
        doc_vectors = document_vectors(documents, tokenizer, model, max_length)

        # Calculate cosine similarities for the query and each document
        similarities = {}
        for docno, doc_vector in doc_vectors.items():
            similarities[docno] = cosine_similarity(query_vector, doc_vector)[0]

        # Sort documents by similarity in descending order
        rankings[query] = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

    return rankings

def main():
    # Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Read documents and queries
    documents = read_documents('coll')
    queries = get_queries('Queries.txt')

    # Retrieve documents for each query using BERT
    ranked_documents = retrieve_with_bert(queries, documents, tokenizer, model)

    # Write results to file
    with open('bert_results.txt', 'w') as file:
        for query, rankings in ranked_documents.items():
            for rank, (docno, score) in enumerate(rankings, start=1):
                if rank >= 1001:
                    break

                line = f"{query} Q0 {docno} {rank} {score} run_name"
                file.write(line + '\n')


if __name__ == "__main__":
    main()
