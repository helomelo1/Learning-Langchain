from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=1.9,
)
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain.")
]

result = model.invoke(messages)
messages.append(AIMessage(result.content))

print(messages)