# Import packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from Parseing import read_documents

#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

import xml.etree.ElementTree as ET

import time

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

stopwordsSet = set(stopwords.words('english'))

def preprocessTokenizeDoc(text):
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', '', text)
    
    text = word_tokenize(text)
    
    tokens = []
    
    for token in text:
        if token not in stopwordsSet:
            tokens.append(token)
    
    
    return tokens

startTime = time.time()

# Import docs 
print("Importing docs...")
docs = read_documents("minicoll")   # Stores a dict with docno as key and has docno, head, dateline, text

# Tokenize and preprocess docs
print("Preprocessing and tokenizing docs...")
taggedDocs = []

for doc in docs.values():
    docTokens = preprocessTokenizeDoc(doc['TEXT'])
    taggedDocs.append(TaggedDocument(words=docTokens, tags=[doc['DOCNO']]))
    
print("Number of docs:", str(len(taggedDocs)))

# Create model
print("Training model...")
model = Doc2Vec(vector_size=100, epochs=10)
model.build_vocab(taggedDocs)
model.train(taggedDocs, total_examples=model.corpus_count, epochs=model.epochs)

# Tokenize and preprocess query
print("Calculating cosine similarities...")
query = "Coping with overcrowded prisons"

# Cosine Similarity calculations for list of docs vs query

results = {}

for doc in docs.values():

    queryVector = model.infer_vector(preprocessTokenizeDoc(query))
    docVector = model.infer_vector(preprocessTokenizeDoc(doc['TEXT']))
    
    cossim = cosine_similarity([queryVector], [docVector])
    
    results[doc['DOCNO']] = cossim[0][0]

#print(results)

# Sort docno by ranking
rankings = sorted(results.items(), key=lambda doc: doc[1], reverse=True)

# Print results and save to file
with open('doc2vec_results.txt', 'w') as file:
    
    line = "topic_id\tQ0\tdocno\trank\tscore\ttag"
    print(line)
    file.write(line+"\n")

    rank = 1
    for docno, cossim in rankings:
        line = "1\tQ0\t"+docno+"\t"+str(rank)+"\t"+str(cossim)+"\t"+"run_name"
        print(line)
        rank = rank + 1
        
        file.write(line+"\n")
        
endTime = time.time()
runTime = endTime - startTime 
print(str(runTime), "seconds taken")