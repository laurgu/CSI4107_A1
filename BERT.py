#Testing for BERT
#Trying to use BERT but there's a 512 Token Limit with it 

import os
import re
import nltk
import torch
import time
from transformers import BertTokenizer, BertModel
#from transformers import TinyBertTokenizer, TinyBertModel
from transformers import DistilBertTokenizer, DistilBertModel
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

#MAP means Mean Average Precision - measures average precision of relevant documents in the ranked search across multiple queries, providing overall algorithm quality 
#P@10 or Precision at 10 means - Measures Precision amongest the top 10 documents, providing quality of search amongst the top of the search

# Load BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# tokenizer = TinyBertTokenizer.from_pretrained('google/tinybert/tinybert-msl-6l-768')
# model = TinyBertModel.from_pretrained('google/tinybert/tinybert-msl-6l-768')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

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

    # Truncate or split tokens if length exceeds 512
    if len(tokens) > 512:
        tokens = tokens[:512]  # Truncate to 512 tokens
    

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




#Retrieves all documents with a word using the old, Part 1 Techniques  
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

#BERT Embeddings
# def encode_embeddings(text):
#     # Tokenize input text
#     tokens = tokenizer.tokenize(text)
    
#     # Convert tokens to input IDs
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
#     # Add special tokens [CLS] and [SEP]
#     input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    
#     # Convert input IDs to tensor
#     input_tensor = torch.tensor(input_ids).unsqueeze(0)
    
#     # Get BERT embeddings
#     with torch.no_grad():
#         outputs = model(input_tensor)
#         embeddings = outputs[0]  # BERT hidden states
    
#     # Average pooling of embeddings
#     avg_embeddings = torch.mean(embeddings, dim=1).squeeze().numpy()
    
#     return avg_embeddings

def encode_document_embedding(text):
    # Tokenize input text
    tokens = tokenizer.tokenize(text)

    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Add special tokens [CLS] and [SEP]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # Truncate or pad input to max length
    input_ids = input_ids[:512]  # Truncate to max length 512 (default for TinyBERT)
    
    # Convert input IDs to tensor
    input_tensor = torch.tensor(input_ids).unsqueeze(0)

    # Get TinyBERT embeddings
    with torch.no_grad():
        outputs = model(input_tensor)
        embeddings = outputs[0]  # TinyBERT hidden states

    # Average pooling of embeddings
    avg_embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

    return avg_embedding

def encode_document_embedding(text):
    # Tokenize input text
    tokens = tokenizer.tokenize(text)

    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Add special tokens [CLS] and [SEP]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # Truncate or pad input to max length
    input_ids = input_ids[:512]  # Truncate to max length 512 (default for BERT)
    
    # Convert input IDs to tensor
    input_tensor = torch.tensor(input_ids).unsqueeze(0)

    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(input_tensor)
        embeddings = outputs[0]  # BERT hidden states

    # Average pooling of embeddings
    avg_embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

    return avg_embedding

    
def retrieve_and_rank_documents_with_BERT(query, documents):
    query_embedding = encode_document_embedding(query)
    document_embeddings = {}
    for doc_id, doc_data in documents.items():
        print(doc_id)
        document_embedding = encode_document_embedding(doc_data['TEXT'])
        document_embeddings[doc_id] = document_embedding

    # Compute cosine similarity for each document
    similarity_scores = {}
    for doc_id, document_embedding in document_embeddings.items():
        print(doc_id)
        similarity_scores[doc_id] = cosine_similarity([query_embedding], [document_embedding])[0][0]

    # Rank documents based on similarity scores
    ranked_documents = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_documents

def precision_at_k(retrieved_documents, relevant_documents, k=5):

    # Take only the top k retrieved documents
    top_k_retrieved = retrieved_documents[:k]

    # Count the number of relevant documents among the top k
    relevant_count = sum(1 for doc_id, _ in top_k_retrieved if doc_id in relevant_documents)

    # Calculate precision at k
    precision = relevant_count / k if k > 0 else 0

    return precision

def calculate_average_precision(retrieved_documents, relevant_documents):
    num_relevant_retrieved = 0
    sum_precision = 0.0
    for rank, (doc_id, _) in enumerate(retrieved_documents, start=1):
        if doc_id in relevant_documents:
            num_relevant_retrieved += 1
            precision_at_rank = num_relevant_retrieved / rank
            sum_precision += precision_at_rank
    if num_relevant_retrieved == 0:
        return 0
    return sum_precision / num_relevant_retrieved

# Function to calculate Mean Average Precision (MAP)
#Errors Below
def calculate_map(topics, inverted_index):
    average_precision = 0.0
    for topic in topics:
        query = topic
        relevant_documents_set = set(inverted_index[term] for term in query.split() if term in inverted_index)
        retrieved_documents = retrieval_ranking(query, inverted_index)
        average_precision += calculate_average_precision(retrieved_documents, relevant_documents_set)
    map_score = average_precision / len(topics)
    return map_score

# Function to calculate Precision at k (P@k)
def calculate_precision_at_k(topics, inverted_index, k=5):
    total_precision = 0.0
    for topic in topics:
        query = topic
        relevant_documents_set = set(inverted_index[term] for term in query.split() if term in inverted_index)
        retrieved_documents = retrieve_and_rank_documents_with_BERT(query, documents)
        precision_at_k_score = precision_at_k(retrieved_documents, relevant_documents_set, k)
        total_precision += precision_at_k_score
    precision_at_k_score = total_precision / len(topics)
    return precision_at_k_score

topics = (
    "Coping with overcrowded prisons",
    "Accusations of Cheating by Contractors on U.S. Defense Projects",
    "Insurance Coverage which pays for Long Term Care",
    "Oil Spills",
    "Right Wing Christian Fundamentalism in U.S.",
    "Efforts to enact Gun Control Legislation",
    "Causes and treatments of multiple sclerosis MS",
    "Term limitations for members of the U.S. Congress",
    "Electric Car Development",
    "Vitamins The Cure for or Cause of Human Ailments",
    "Acid Rain",
    "Automobile Recalls",
    "Vietnam Veterans and Agent Orange",
    "Generic Drugs Illegal Activities by Manufacturers",
    "Tobacco company advertising and the young",
    "Standardized testing and cultural bias",
    "Topic: Regulation of the showing of violence and explicit sex in motion picture theaters, on television, and on video cassettes.",
    "Financing AMTRAK",
    "Cost of Garbage/Trash Removal",
    "The Consequences of Implantation of Silicone Gel Breast Devices",
    "Use of Mutual Funds in an Individual's Retirement Strategy",
    "The Effectiveness of Medical Products and Related Programs Utilized in the Cessation of Smoking.",
    "Smoking Bans",
    "Hazardous Waste Cleanup",
    "NRA Prevention of Gun Control Legislation",
    "Real ife private investigators",
    "English as the Official Language in U.S.",
    "Dog Maulings",
    "U. S. Restaurants in Foreign Lands",
    "Ineffectiveness of U.S. Embargoes/Sanctions",
    "Abuse of the Elderly by Family Members, and Medical and Nonmedical Personnel, and Initiatives Being Taken to Minimize This Mistreatment",
    "Commercial Overfishing Creates Food Fish Deficit",
    "Asbestos Related Lawsuits",
    "Corporate Pension Plans/Funds",
    "Reform of the U.S. Welfare System",
    "Topic: Difference of Learning Levels Among Inner City and More Suburban School Students",
    "Signs of the Demise of Independent Publishing",
    "Beachfront Erosion",
    "Real Motives for Murder",
    "Instances of Fraud Involving the Use of a Computer",
    "Efforts to Improve U.S. Schooling",
    "Oil Spill Cleanup",
    "Toys R Dangerous",
    "The Amount of Money Earned by Writers",
    "Stock Market Perturbations Attributable to Computer Initiated Trading",
    "School Choice Voucher System and its effects upon the entire U.S. educational program",
    "Reform of the jurisprudence system to stop juries from granting unreasonable monetary awards",
    "Gene Therapy and Its Benefits to Humankind",
    "Legality of Medically Assisted Suicides",
    "Impact of foreign textile imports on U.S. textile industry"
)



inverted_index = {}
query = "Coping with prisons"

# Example usage
folder_path = "mini"  # Path to the folder containing multiple files
documents = read_documents(folder_path)

print(len(documents))


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


#relevantDocuments = retrieval_ranking("Coping with overcrowded prisons",inverted_index_with_positions)


# print("Q0\tDocId\tCosineSimilarity\tRank\t", "run_name")
# for rank, (doc_id, score) in enumerate(relevantDocuments, start=1):
#     print(f"Q0\t{doc_id}\t{score}\t{rank}\t{'run_name'}")

print("Pre Bert")

relevant_documents = retrieve_and_rank_documents_with_BERT(query, documents)

print("Post Bert")


# Print results
print("Q0\tDocId\tCosineSimilarity\tRank\t", "run_name")
for rank, (doc_id, score) in enumerate(relevant_documents, start=1):
    print(f"Q0\t{doc_id}\t{score}\t{rank}\t{'run_name'}")

# Calculate MAP and Precision at 10
map_score = calculate_map(topics, inverted_index_with_positions)
precision_at_10_score = calculate_precision_at_k(topics, inverted_index_with_positions, k=5)

print("Mean Average Precision (MAP):", map_score)
print("Precision at 10 (P@10):", precision_at_10_score)