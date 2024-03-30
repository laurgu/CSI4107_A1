#Testing for BERT

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

#BERT (Bidirectional Encoder Representations from Transformers)
#Meaning can be found from the words before and after the word to analyze
#Bidirectional meaning before and after a word 
#Encoder meaning Encoders and Decoders 
#Representations means it represents words in vectors and numbers in high dimensional spaces 
#Uses the Transformers architecture meaning the prcossing is done in parallel meaning it captures the relationship between words

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')
model = BertModel.from_pretrained('prajjwal1/bert-tiny', ignore_mismatched_sizes=True)

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

#Part 1 Preprocess Text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove markup that is not part of the text
    text = re.sub(r'<.*?>', '', text)
    
    # Tokenization using BERT tokenizer
    tokens = tokenizer.tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    

    if not tokens:
        tokens.append('empty')  # Placeholder token
    
    return tokens

#Part 2 Inverted Index
def build_inverted_index(documents):
    inverted_index = {}
    for doc_id, doc_data in documents.items():
        text = doc_data['TEXT']
        tokens = preprocess_text(text)  # Preprocess text
        
        # Track position of tokens in document
        positions = {}
        for position, token in enumerate(tokens):
            if token not in positions:
                positions[token] = {'doc_ids': [], 'positions': []}
            positions[token]['doc_ids'].append(doc_id)
            positions[token]['positions'].append(position)
        
        # Update inverted index
        for token, token_info in positions.items():
            if token not in inverted_index:
                inverted_index[token] = {}
            for i, doc_id in enumerate(token_info['doc_ids']):
                if doc_id not in inverted_index[token]:
                    inverted_index[token][doc_id] = []
                inverted_index[token][doc_id].append(token_info['positions'][i])
    
    return inverted_index









#Retrieves all documents with a word  
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



def encode_embeddings(text):
    # Tokenize input text
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:510]
    
    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Add special tokens [CLS] and [SEP]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    
    if len(input_ids) < 512:
        input_ids = input_ids + [tokenizer.pad_token_id] * (512 - len(input_ids))
    else:
        input_ids = input_ids[:512]
        
    # Convert input IDs to tensor
    input_tensor = torch.tensor(input_ids).unsqueeze(0)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(input_tensor)
        embeddings = outputs[0]  # BERT hidden states
    
    # Average pooling of embeddings
    avg_embeddings = torch.mean(embeddings, dim=1).squeeze().numpy()
    
    return avg_embeddings
    
def retrieve_and_rank_documents_with_BERT(query, documents):
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

startTime = time.time()

# Example usage
folder_path = "coll"  # Path to the folder containing multiple files
documents = read_documents(folder_path)

print(len(documents))

inverted_index = {}
query = "Coping with overcrowded prisons"

# Example usage
folder_path = "coll"  # Path to the folder containing multiple files
documents = read_documents(folder_path)


inverted_index = {}

# Call preprocess_text and build_inverted_index functions
for doc_id, doc_data in documents.items():
    # Preprocess text
    text = doc_data['TEXT']
    tokens = preprocess_text(text)

    # Update inverted index
    for token in tokens:
        if token not in inverted_index:
            inverted_index[token] = []
        inverted_index[token].append(doc_id)

# Build inverted index with positions
inverted_index_with_positions = build_inverted_index(documents)


relevantDocuments = retrieval_ranking("Coping with overcrowded prisons",inverted_index_with_positions)


print("Q0\tDocId\tCosineSimilarity\tRank\t", "run_name")
for rank, (doc_id, score) in enumerate(relevantDocuments, start=1):
    print(f"Q0\t{doc_id}\t{score}\t{rank}\t{'run_name'}")

relevant_documents = retrieve_and_rank_documents_with_BERT(query, documents)

# Print results
print("Q0\tDocId\tCosineSimilarity\tRank\t", "run_name")
for rank, (doc_id, score) in enumerate(relevant_documents, start=1):
    print(f"Q0\t{doc_id}\t{score}\t{rank}\t{'run_name'}")
    

# Print results and save to file
with open('BERT_results.txt', 'w') as file:
    
    line = "topic_id\tQ0\tdocno\trank\tscore\ttag"
    print(line)
    file.write(line+"\n")

    rank = 1
    for docno, cossim in relevant_documents:
        line = "1\tQ0\t"+docno+"\t"+str(rank)+"\t"+str(cossim)+"\t"+"run_name"
        print(line)
        rank = rank + 1
        
        file.write(line+"\n")
        
endTime = time.time()
runTime = endTime - startTime 
print(str(runTime), "seconds taken")