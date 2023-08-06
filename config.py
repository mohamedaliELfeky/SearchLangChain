import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "API tokens"

MILVUS_COLLECTION_NAME = "LangChainTest"
MILVUS_CONNECTION_ARG = {"host": "127.0.0.1", "port": "19530",
                         "collection_name":MILVUS_COLLECTION_NAME}



PERSIST_DIRECTORY = 'db'
DOCS_PATH = 'docs'
EXTENSION = '.pdf'

# docs
CHUNK_SIZE = 400
CHUNK_OVER = 200


