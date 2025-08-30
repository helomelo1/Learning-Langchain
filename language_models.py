from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=1.9,
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("Suggest me dog names (not in bold please)")

print(result.content)