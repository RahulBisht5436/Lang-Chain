import streamlit as st

from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.globals import set_debug

from llm.openAI_llm import llm


# ---------------------------------
# Enable LangChain Debug Logs
# ---------------------------------
set_debug(True)


# ---------------------------------
# Load Tools
# ---------------------------------
tools = load_tools(["wikipedia", "ddg-search"])


# ---------------------------------
# System Prompt
# ---------------------------------
react_system_prompt = """
You are a ReAct-style AI agent.

Follow this loop carefully:

1. THINK about the problem.
2. If additional information is required,
   use one of the available tools.
3. Read the tool result.
4. Continue reasoning.
5. Repeat until you have enough information.
6. Give the final answer.

Available tools:
- wikipedia
- ddg-search
"""


# ---------------------------------
# Create Agent
# ---------------------------------
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=react_system_prompt,
)


# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="LangChain Agent", layout="wide")

st.title("🤖 LangChain ReAct Agent")

task = st.text_input("Ask me anything")

if st.button("Run") and task:

    with st.spinner("Thinking..."):

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": task,
                    }
                ]
            }
        )

    # ---------------------------------
    # Print Complete Response
    # ---------------------------------
    print("\n================ RAW RESULT ================\n")
    print(result)

    # ---------------------------------
    # Final Answer
    # ---------------------------------
    final_answer = result["messages"][-1].content

    st.success("Final Answer")

    st.write(final_answer)

    # ---------------------------------
    # Full Message Flow
    # ---------------------------------
    st.divider()

    st.header("Agent Execution")

    for index, message in enumerate(result["messages"], start=1):

        st.subheader(f"Step {index}")

        st.write("**Message Type:**", message.__class__.__name__)

        if getattr(message, "content", None):
            st.write("**Content:**")
            st.write(message.content)

        # Tool Calls
        if hasattr(message, "tool_calls") and message.tool_calls:

            st.write("**Tool Calls:**")

            st.json(message.tool_calls)

        # Additional Metadata
        if hasattr(message, "response_metadata"):

            with st.expander("Response Metadata"):

                st.json(message.response_metadata)