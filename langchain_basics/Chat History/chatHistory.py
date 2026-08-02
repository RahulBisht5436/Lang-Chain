# Import ChatPromptTemplate to create chat-based prompts
# and MessagesPlaceholder to inject previous conversation history.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import the LLM (Ollama model)
from ollama_llm import llm

# ChatMessageHistory stores all previous Human and AI messages.
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

# RunnableWithMessageHistory automatically manages conversation history
# for every invocation of the chain.
from langchain_core.runnables.history import RunnableWithMessageHistory


# -----------------------------------------------------------------------------
# Prompt Template
# -----------------------------------------------------------------------------
# The prompt consists of:
# 1. A System message -> Gives the AI its role.
# 2. A MessagesPlaceholder -> Injects previous conversation automatically.
# 3. A Human message -> Receives the latest user input.
# -----------------------------------------------------------------------------
prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            "You are Senior Developer and Need to provide a discussion on the topic"
        ),

        # This placeholder will be replaced by all previous Human/AI messages
        # maintained by RunnableWithMessageHistory.
        MessagesPlaceholder(variable_name="chat_history"),

        # Current user input
        ("human", "{input}")
    ]
)


# -----------------------------------------------------------------------------
# Create the LCEL Chain
# -----------------------------------------------------------------------------
# Flow:
# PromptTemplate ---> LLM
# -----------------------------------------------------------------------------
chain = prompt_template | llm


# Print the chain graph for visualization.
print(chain.get_graph().print_ascii())


# -----------------------------------------------------------------------------
# Create an in-memory chat history object.
#
# This object stores the entire conversation.
#
# Example after a few messages:
#
# Human : Tell me about MongoDB.
# AI    : MongoDB is...
# Human : Explain Indexes.
# AI    : MongoDB supports...
#
# Everything is stored inside this object.
# -----------------------------------------------------------------------------
history_of_chains = ChatMessageHistory()


# -----------------------------------------------------------------------------
# Wrap the chain with history support.
#
# RunnableWithMessageHistory performs these tasks automatically:
#
# 1. Fetches previous conversation.
# 2. Inserts it into the MessagesPlaceholder.
# 3. Executes the chain.
# 4. Saves the latest Human and AI messages back into history.
#
# The lambda receives the session_id and returns the corresponding history.
#
# Currently every session uses the SAME history object because we always return
# history_of_chains.
# -----------------------------------------------------------------------------
chain_with_history = RunnableWithMessageHistory(

    # Original chain
    chain,

    # Function that returns the history for a session.
    # session_id is supplied through config.
    lambda session_id: history_of_chains,

    # Name of the current user input key.
    input_messages_key="input",

    # Name of the MessagesPlaceholder variable in the prompt.
    history_messages_key="chat_history"
)


# -----------------------------------------------------------------------------
# Continuous Chat Loop
# -----------------------------------------------------------------------------
while True:

    # Read the current user question.
    question = input("What System you need to have discussion on: ")

    # Ignore empty input.
    if question:

        # Invoke the chain.
        result = chain_with_history.invoke(

            # Current user message
            {
                "input": question
            },

            # Configuration values are NOT passed to the prompt.
            # They are consumed internally by RunnableWithMessageHistory.
            config={
                "configurable": {

                    # Used by RunnableWithMessageHistory to determine
                    # which conversation history to load.
                    "session_id": "abc123"
                }
            }
        )

        # Print the AI response.
        print(result.content)

        # Blank line for readability.
        print()