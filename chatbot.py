from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=1.9,
)
model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful chat assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")

print(chat_history)