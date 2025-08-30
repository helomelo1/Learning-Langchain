from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)

text = "India is a decent country."
docs = [
    "His name is Alex",
    "India is a secular country.",
    "Dogs are really cute animals."
]

text_embedding = embeddings.embed_query(text)
print(text_embedding)
print(len(text_embedding))

batch_embedding = embeddings.embed_documents(docs)
print(len(batch_embedding[0]))
print(batch_embedding[0])