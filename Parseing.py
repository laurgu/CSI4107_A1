#Complete Parsing
#Parses the Profs documents
#Just for Reference
import os
import re


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

"""
# Example usage
folder_path = "coll"  # Path to the folder containing multiple files
documents = read_documents(folder_path)

print(len(documents))

# Printing the contents of the first document for verification
docno = list(documents.keys())[0]
print(f"Document ID: {docno}")
print(f"Headline: {documents[docno]['HEAD']}")
print(f"Dateline: {documents[docno]['DATELINE']}")
print(f"Text: {documents[docno]['TEXT']}")

"""