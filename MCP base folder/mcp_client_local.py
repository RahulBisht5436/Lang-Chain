import asyncio
import streamlit as st
import sys

from llm.openAI_llm import llm
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


client = MultiServerMCPClient({
    "tools": {
        "command": sys.executable,
        "args": ["mcp_server_local.py"],
        "transport": "stdio"
    }
})

tools = asyncio.run(client.get_tools())

agent = create_agent(llm, tools)

st.title("AI Agent (MCP Version)")
task = st.text_input("Assign me a task")

if task:
    response = asyncio.run(agent.ainvoke({"messages": task}))
    final_output = response["messages"][-1].content
    st.write(final_output)

