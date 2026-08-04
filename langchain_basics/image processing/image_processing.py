from llm.openAI_llm import llm
from langchain_core.prompts import ChatPromptTemplate

import base64
import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(CURRENT_DIR, "image_analy.jpg")


def encode_image(image_path: str) -> str:
    """Convert image to Base64."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


try:
    image = encode_image(IMAGE_PATH)
except Exception as e:
    print(f"❌ Error loading image:\n{e}")
    exit()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that analyzes images."),
        (
            "human",
            [
                {
                    "type": "text",
                    "text": "{input}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}",
                        "detail": "high",  # low | high | auto
                    },
                },
            ],
        ),
    ]
)

chain = prompt | llm

print("=" * 60)
print("🖼️ Image Analyzer")
print("Type 'exit' to quit.")
print("=" * 60)

while True:

    question = input("\nAsk a question: ").strip()

    if question.lower() in ("exit", "quit"):
        print("\n👋 Goodbye!")
        break

    if not question:
        print("⚠️ Please enter a question.")
        continue

    try:
        response = chain.invoke({"input": question})

        print("\n" + "=" * 60)
        print("🤖 Response:")
        print("-" * 60)
        print(response.content)
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error while calling the model:\n{e}")