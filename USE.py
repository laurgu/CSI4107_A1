#Uses the USE model via Tensorflow
#Run this code

#All Pip installs
# pip install nltk
# pip install numpy
# pip install tensorflow
# pip install tensorflow-hub


import os
import re
import nltk
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


#Neural Internet Retrival Models 

#Uses Neural Networks / Transformers
#Uses Semantic Analysis to understand the meaning of words, not just keyword matching 
#Transformers analyze entire sentences and phrases instead of viewing words sequentially 


#USE (Universal Sentence Encoder)

#Encodes Sentences into fixed length vectors 
#Uses the Neural Networks Architecture to generate Embeddings 
#Encodes means converting into a type of data usuable for analyze purposes 
#Decoding means converting back into a type of data that humans can read 

# Load Universal Sentence Encoder
use_module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
use_embed = hub.load(use_module_url)

def read_documents(folder_path):
    all_documents = {}
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            documents = read_documents_from_file(filepath)
            all_documents.update(documents)
    return all_documents

def read_documents_from_file(filepath):
    documents = {}
    with open(filepath, "r", encoding="utf-8") as file:
        doc_contents = file.read()
        doc_data = parse_documents(doc_contents)
        documents.update(doc_data)
    return documents

def parse_documents(doc_contents):
    documents = {}
    doc_pattern = re.compile(r'<DOC>(.*?)</DOC>', re.DOTALL)
    for match in doc_pattern.finditer(doc_contents):
        doc_content = match.group(1).strip()
        doc_data = parse_document(doc_content)
        documents[doc_data['DOCNO']] = doc_data
    return documents

def parse_document(doc_content):
    doc_data = {}
    doc_data['DOCNO'] = re.search(r'<DOCNO>(.*?)</DOCNO>', doc_content, re.DOTALL).group(1).strip()
    doc_data['HEAD'] = re.search(r'<HEAD>(.*?)</HEAD>', doc_content, re.DOTALL).group(1).strip() if re.search(r'<HEAD>(.*?)</HEAD>', doc_content, re.DOTALL) else ""
    doc_data['DATELINE'] = re.search(r'<DATELINE>(.*?)</DATELINE>', doc_content, re.DOTALL).group(1).strip() if re.search(r'<DATELINE>(.*?)</DATELINE>', doc_content, re.DOTALL) else ""
    doc_data['TEXT'] = re.search(r'<TEXT>(.*?)</TEXT>', doc_content, re.DOTALL).group(1).strip() if re.search(r'<TEXT>(.*?)</TEXT>', doc_content, re.DOTALL) else ""
    return doc_data

# Part 1 Preprocess Text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove markup that is not part of the text
    text = re.sub(r'<.*?>', '', text)
    
    # Tokenization using NLTK tokenizer
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    if not tokens:
        tokens.append('empty')  # Placeholder token
    
    return tokens

# Retrieves all documents with a word  
def retrieval_ranking(query, inverted_index):
    relevant_documents = set()
    query_terms = query.split()
    
    # Retrieve documents containing at least one query word
    for term in query_terms:
        if term in inverted_index:
            relevant_documents.update(inverted_index[term])
    
    # Collect document texts
    document_texts = [documents[doc_id]['TEXT'] for doc_id in relevant_documents]
    
    # Compute cosine similarity for each relevant document
    tfidf_vectorizer = TfidfVectorizer()
    document_tfidf = tfidf_vectorizer.fit_transform(document_texts)
    query_tfidf = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_tfidf, document_tfidf)[0]
    
    # Rank documents based on similarity scores
    ranked_documents = sorted(zip(relevant_documents, similarity_scores), key=lambda x: x[1], reverse=True)
    
    return ranked_documents

# Encode text using Universal Sentence Encoder
def encode_embeddings(text):
    embeddings = use_embed([text])
    return embeddings.numpy()[0]

# Retrieves and ranks documents using Universal Sentence Encoder
def retrieve_and_rank_documents_with_USE(query, documents):
    query_embedding = encode_embeddings(query)
    document_embeddings = {}
    for doc_id, doc_data in documents.items():
        document_embedding = encode_embeddings(doc_data['TEXT'])
        document_embeddings[doc_id] = document_embedding
    
    # Compute cosine similarity for each document
    similarity_scores = {}
    for doc_id, document_embedding in document_embeddings.items():
        similarity_scores[doc_id] = cosine_similarity([query_embedding], [document_embedding])[0][0]
    
    # Rank documents based on similarity scores
    ranked_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_documents

# Example usage
folder_path = "coll"  # Path to the folder containing multiple files
documents = read_documents(folder_path)

# Retrieval and ranking based on TF-IDF
inverted_index = {}
for doc_id, doc_data in documents.items():
    # Preprocess text
    text = doc_data['TEXT']
    tokens = preprocess_text(text)

    # Update inverted index
    for token in tokens:
        if token not in inverted_index:
            inverted_index[token] = []
        inverted_index[token].append(doc_id)

# relevant_documents = retrieval_ranking("Coping with overcrowded prisons", inverted_index)
# print("TF-IDF Retrieval and Ranking:")
# print("Q0\tDocId\tCosineSimilarity\tRank\t", "run_name")
# for rank, (doc_id, score) in enumerate(relevant_documents, start=1):
#     print(f"Q0\t{doc_id}\t{score}\t{rank}\t{'run_name'}")

# Retrieval and ranking based on Universal Sentence Encoder (USE)
relevant_documents_use = retrieve_and_rank_documents_with_USE("Coping with overcrowded prisons", documents)
print("\nUniversal Sentence Encoder Retrieval:")
print("Q0\tDocId\tCosineSimilarity\tRank\t", "run_name")

# for rank, (doc_id, score) in enumerate(relevant_documents_use, start=1):
#     print(f"Q0\t{doc_id}\t{score}\t{rank}\t{'run_name'}")

for rank, (doc_id, score) in enumerate(relevant_documents_use, start=1):
    if rank > 1000:
        break
    print(f"Q0\t{doc_id}\t{score}\t{rank}\t{'run_name'}")
