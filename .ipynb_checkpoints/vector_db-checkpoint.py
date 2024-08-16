import getpass
import os
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain import Bedrockembeddings
from Huggingface import datasets
def _set_if_undefined(var: str) -> None:
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var)
_set_if_undefined("LANGCHAIN_API_KEY")
#_set_if_undefined("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")
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Submission"
import boto3
import json, re
from langchain_community.embeddings import BedrockEmbeddings
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from llama_index.core import SimpleDirectoryReader, ServiceContext, StorageContext, VectorStoreIndex

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client, region_name: str, model_id: str):
        self.embedder = BedrockEmbeddings(
            client=client,
            region_name=region_name,
            model_id=model_id
        )
    def embed_query(self, query: str) -> Embeddings:
        return self.embedder.embed_query(query)
    def embed_documents(self, documents: list[str]) -> Embeddings:
        return self.embedder.embed_documents(documents)

from datasets import load_dataset


bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)
ds = load_dataset("neural-bridge/rag-full-20000")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0.2
)
doc_splits = text_splitter.split_documents(ds)



