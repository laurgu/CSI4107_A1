#Complete Parsing
#Parses the Profs documents
import os
import re
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


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
    # doc_data['HEAD'] = re.search(r'<HEAD>(.*?)</HEAD>', doc_content, re.DOTALL).group(1).strip() if re.search(r'<HEAD>(.*?)</HEAD>', doc_content, re.DOTALL) else ""
    # doc_data['DATELINE'] = re.search(r'<DATELINE>(.*?)</DATELINE>', doc_content, re.DOTALL).group(1).strip() if re.search(r'<DATELINE>(.*?)</DATELINE>', doc_content, re.DOTALL) else ""
    doc_data['TEXT'] = re.search(r'<TEXT>(.*?)</TEXT>', doc_content, re.DOTALL).group(1).strip() if re.search(r'<TEXT>(.*?)</TEXT>', doc_content, re.DOTALL) else ""
    return doc_data

def get_queries(filepath):
    queries = []
    with open(filepath, "r") as file:
        for line in file:
            queries.append(line.strip())
    return queries

def preprocessTokenizeDoc(text):
    # Set of English stopwords
    stopwordsSet = set(stopwords.words('english'))
    
    # Initialize Porter Stemmer
    stemmer = PorterStemmer()

    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize text
    text = word_tokenize(text)
    
    # Apply stemming and remove stopwords
    tokens = [stemmer.stem(token) for token in text if token not in stopwordsSet]
    
    return tokens

def expand_query(query):
    expanded_query = []
    for term in query.split():
        # Find synonyms for each term using WordNet
        synonyms = set()
        for synset in wordnet.synsets(term):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name())
        expanded_query.extend(synonyms)
    return ' '.join(expanded_query)