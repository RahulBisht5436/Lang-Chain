import streamlit as st
import base64
from langchain_core.prompts import ChatPromptTemplate
from llm.openAI_llm import llm


st.set_page_config(
    page_title="Image Analyzer",
    page_icon="🖼️",
    layout="centered",
)

st.title("🖼️ Image Analyzer")
st.write("Upload an image and ask questions about it.")


def encode_image(uploaded_file) -> str:
    """Convert uploaded image to Base64."""
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    question = st.text_input(
        "Ask a question about the image",
        placeholder="Example: Describe this image",
    )

    if st.button("Analyze Image"):

        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        image = encode_image(uploaded_file)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that analyzes images.",
                ),
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
                                "detail": "high",
                            },
                        },
                    ],
                ),
            ]
        )

        chain = prompt | llm

        with st.spinner("Analyzing image..."):

            try:
                response = chain.invoke(
                    {
                        "input": question,
                    }
                )

                st.success("Analysis Complete")

                st.subheader("🤖 Response")
                st.write(response.content)

            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("Upload an image to begin.")