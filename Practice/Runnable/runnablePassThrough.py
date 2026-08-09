from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Input
question = "What is LangChain?"

# Create a parallel runnable
chain = RunnableParallel(
    original=RunnablePassthrough(),
    uppercase=lambda x: x.upper()
)

# Execute
result = chain.invoke(question)

print(result)