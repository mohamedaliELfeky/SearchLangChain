from encodeText import embbedingAPI, get_encoder
from MiniLama import get_LaMini
import streamlit as st 


# from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.vectorstores import Milvus, Chroma 
from langchain.chains import RetrievalQA

import config


def qa_llm():
    llm =  get_LaMini()
    embeddings = get_encoder() #SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Milvus(
                embedding_function=embeddings,
                connection_args=config.MILVUS_CONNECTION_ARG
                )
    
    
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm,
                                      chain_type="stuff", 
                                      retriever=retriever, 
                                      return_source_documents=True,                               
                                    )
    return qa

def process_answer(instruction):

    instruction = instruction

    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    
    return answer, generated_text

def main():

    st.title("Search Your PDF 🐦📄")

    with st.expander("About the App"):

        st.markdown(
            """
            This is a Generative AI powered Question and Answering app that responds to questions about your PDF File.
            """
        )

    question = st.text_area("Enter your Question")
    
    if st.button("Ask"):

        st.info("Your Question: " + question)
        st.info("Your Answer")
        
        answer, metadata = process_answer(question)
        
        st.write(answer)
        st.write(metadata)


if __name__ == '__main__':
    main()