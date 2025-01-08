import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

os.environ["OPENAI_API_KEY"] = ""
embedding_model = OpenAIEmbeddings()
raw_documents = TextLoader('dataset/corpus/multisourceqa.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
splits = text_splitter.split_documents(raw_documents)
store = FAISS.from_documents(splits, embedding_model)
store.save_local('dataset/vectorstore/multisourceqa')
