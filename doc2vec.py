import os
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity


#Code just to try out doc2vec and its great cause it can used for an entire document


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

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

# Part 2: Indexing
def build_inverted_index(preprocessed_documents):
    inverted_index = {}
    for docno, tokens in preprocessed_documents.items():
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = [docno]
            else:
                inverted_index[token].append(docno)
    return inverted_index

# Function to retrieve and rank documents using inverted index with positions
def encode_query(query, max_length=512):
    input_ids = tokenizer.encode(query, return_tensors='pt', max_length=max_length, truncation=True)
    attention_mask = input_ids != tokenizer.pad_token_id
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    return np.squeeze(outputs[0][:, 0, :].numpy())  # Flatten to 2 dimensions

# Train Doc2Vec model
def train_doc2vec_model(documents):
    tagged_data = [TaggedDocument(words=doc, tags=[docno]) for docno, doc in documents.items()]
    model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=20)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model


# Retrieve and rank documents using Doc2Vec
def retrieve_and_rank_documents_doc2vec(query, doc2vec_model, documents):
    query_embedding = doc2vec_model.infer_vector(query.split())
    document_scores = {}
    for docno, doc_text in documents.items():
        doc_embedding = doc2vec_model.infer_vector(doc_text)
        score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        document_scores[docno] = score
    
    ranked_documents = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_documents  # Return top 10 ranked documents



# Example usage
folder_path = "minicoll"  # Path to the folder containing multiple files
documents = read_documents(folder_path)

print(len(documents))

# Printing the contents of the first document for verification
# docno = list(documents.keys())[0]
# print(f"Document ID: {docno}")
# print(f"Headline: {documents[docno]['HEAD']}")
# print(f"Dateline: {documents[docno]['DATELINE']}")
# print(f"Text: {documents[docno]['TEXT']}")

#10 Seconds

#Part 1 
# Iterate through all documents and preprocess their text
preprocessed_documents = {}
for docno, doc_data in documents.items():
    preprocessed_text = preprocess_text(doc_data['TEXT'])
    preprocessed_documents[docno] = preprocessed_text

# for docno, preprocessed_text in preprocessed_documents.items():
#     print(f"Document ID: {docno}")
#     print("Preprocessed Text:", preprocessed_text)
#     print()

#Part 2 
# Build inverted index with positions
inverted_index_with_positions = build_inverted_index(preprocessed_documents)

# Example usage: Print inverted index with positions
# for token, positions in inverted_index_with_positions.items():
#     print(f"Token: {token}")
#     for docno, pos_list in positions.items():
#         print(f"  Document ID: {docno}")
#         print(f"  Positions: {pos_list}")
#     print()

query = "Coping with overcrowded prisons"
query_embedding = encode_query(query)

doc2vec_model = train_doc2vec_model(preprocessed_documents)

ranked_documents = retrieve_and_rank_documents_doc2vec(query, doc2vec_model, preprocessed_documents)

# Print ranked documents
for docno, score in ranked_documents:
    print(f"Document ID: {docno}, Score: {score}")
    print()