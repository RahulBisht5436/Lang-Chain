from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm_middleware = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=200,
)

llm = ChatHuggingFace(llm=llm_middleware)

if __name__ == "__main__":

    response = llm.invoke(
        """
        Return the following information ONLY as JSON.

        {
            "name": "Rahul",
            "age": 25,
            "city": "Delhi"
        }

        Do not provide any explanation.
        """
    )

    print(response.content)