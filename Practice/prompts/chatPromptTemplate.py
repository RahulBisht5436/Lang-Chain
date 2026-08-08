from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import sys
from pathlib import Path

# Handling for the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.openAI_llm import llm
chatHistory = []

starterTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an experienced {role}."),
        ("ai", "How can I help you?"),
        ("human", "Provide me analysis of {topic} from a critical point of view."),
        ("human", "Provide both bottlenecks and advantages of it."),
    ]
)

coversationTemplate = ChatPromptTemplate(
    [
        (
            "system",
            "You are an experienced {role} and already started conversation on {topic}",
        ),
        MessagesPlaceholder("ChatHistory"),
        ("human", "Next Question is {question}"),
    ]
)


starterChain = starterTemplate | llm
conversationChain = coversationTemplate | llm

if True:

    role = input("What is the role?\n")
    topic = input("What is the topic?\n")
    if topic.lower() == "exit" or role.lower() == "exit":
        pass
    answer = starterChain.invoke({"role": role, "topic": topic})
    chatHistory.append({"human":answer.content})

    while answer:
        print(answer.content)
        coversationQuestion = input("Have another question ?? \n")
        chatHistory.append({"human":coversationQuestion})
        if coversationQuestion.lower() == "exit":
            break
        answer = conversationChain.invoke(
            {"role": role, "topic": topic, "question": coversationQuestion , "ChatHistory":chatHistory}
        )
        chatHistory.append({"AI":answer.content})
        
        print("This is current Chat History ===========================>>>>>>>>>>")
        print(chatHistory)
        print("============================>>>")
        
        print(answer.content)
