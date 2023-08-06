from langchain.vectorstores import Milvus

from encodeText import get_encoder
from CreateDoc import main
import config


for docs in main():

    vector_db = Milvus.from_documents(
        docs,
        get_encoder(),
        connection_args=config.MILVUS_CONNECTION_ARG,   
    )

