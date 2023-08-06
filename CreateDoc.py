from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 

import config

import os 
import tiktoken

tokenizer = tiktoken.get_encoding("p50k_base")

def tikoken_len(text):

    tokens = tokenizer.encode(text=text,
                           disallowed_special=())

    return len(tokens)

def main():

    for root, _, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE,
                                                                chunk_overlap=config.CHUNK_OVER,
                                                                length_function=tikoken_len,
                                                                separators=['\n\n', '\n', ' ', '']
                                                                )
                
                texts = text_splitter.split_documents(documents)

                print(f"num of docs {len(texts)}")

                yield texts