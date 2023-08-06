import requests
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
import config
from typing import List

models_id = ['sentence-transformers/paraphrase-MiniLM-L6-v2',
             "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
             "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
            #  "facebook/bart-base",
            "model-embeddings/multi-qa-mpnet-base-dot-v1"
             ]

MODEL_ID = models_id[0]
hf_token = "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"


api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
headers = {"Authorization": f"Bearer {hf_token}"}


def get_encoder():
    return HuggingFaceHubEmbeddings(
                                    repo_id=MODEL_ID,
                                    # model_kwargs={
                                    #                 #"temperature": 0.5,
                                    #                 #"max_length": 64,
                                    #                 "wait_for_model":True
                                    #             },
                                    # huggingfacehub_api_token=hf_token,
                                    #task="sentence-similarity"
                                    )

class embbedingAPI(Embeddings):
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.query(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.query(text)

    def query(self, texts):
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
        return response.json()



if __name__ == '__main__':
    pass
    # print(len(query(["i want to try to catch him", 'no one here are availabe'])))

