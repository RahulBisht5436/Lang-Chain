# Import the Hugging Face LLM wrapper and chat interface from LangChain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Import dotenv to load environment variables from a .env file
from dotenv import load_dotenv

# Load all environment variables from the .env file into the current process
# Example:
# HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxx
load_dotenv()


# -------------------------------------------------------------------
# Create a Hugging Face LLM endpoint
# -------------------------------------------------------------------
# repo_id specifies the model hosted on Hugging Face Hub.
#
# If HUGGINGFACEHUB_API_TOKEN exists in your environment,
# LangChain will automatically use it.
#
# This object is responsible for communicating with the Hugging Face
# Inference API.
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528"
)


# -------------------------------------------------------------------
# Wrap the endpoint in a chat model
# -------------------------------------------------------------------
# HuggingFaceEndpoint behaves like a text generation model.
# ChatHuggingFace converts it into a chat model so you can use
# chat-style methods such as invoke(), stream(), and batch().
chat = ChatHuggingFace(llm=llm)

chat.get_graph().print_ascii()
# -------------------------------------------------------------------
# Program entry point
# -------------------------------------------------------------------
# This block executes only when this file is run directly.
# It will NOT execute if this file is imported into another Python file.
if __name__ == "__main__":

    # Send a prompt to the chat model.
    # invoke() sends one request and waits for the complete response.
    response = chat.invoke("What is LangChain?")

    # response is an AIMessage object.
    # The generated text is stored in its 'content' attribute.
    print(response.content)