import requests
import sys
from pathlib import Path

from langchain_community.tools import tool

current_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(current_dir))

from llm.openAI_llm import llm


@tool
def getConverstionRate(
    baseCurrency: str,
    convertedCurrency: str
) -> float:
    """
    Provides the exchange rate between two currencies.
    """

    conversion_url = (
        f"https://api.frankfurter.dev/v2/rate/"
        f"{baseCurrency}/{convertedCurrency}"
    )

    response = requests.get(conversion_url)
    response.raise_for_status()

    return response.json()["rate"]


@tool
def convertCurreny(
    amount: float,
    rate: float
) -> float:
    """
    Multiplies the amount by the exchange rate.
    """

    return amount * rate


baseCurrency = "USD"
targetCurrency = "INR"
amount = 10


tool_llm = llm.bind_tools([
    getConverstionRate,
    convertCurreny
])


query = f"""
Convert {amount} {baseCurrency} into {targetCurrency}.

You must:
1. Get the exchange rate using getConverstionRate.
2. Use that rate with convertCurreny function.
3. Return the final converted amount.
"""

# ------------------------------------------------
# STEP 1: Ask LLM what tool it wants to call
# ------------------------------------------------

result = tool_llm.invoke(query)

print("LLM requested:")
print(result.tool_calls)


# ------------------------------------------------
# STEP 2: Execute first tool
# ------------------------------------------------

tool_call = result.tool_calls[0]
# print(tool_call)
print("===================================>>>>>>>>>")


if tool_call["name"] == "getConverstionRate":

    rate = getConverstionRate.invoke(tool_call["args"])

    print("Exchange rate:", rate)


    # ------------------------------------------------
    # STEP 3: Ask LLM to use the result
    # ------------------------------------------------

    result2 = tool_llm.invoke(
        f"""
        The tool getConverstionRate returned:

        rate = {rate}

        Now call the convertCurreny tool with:
        amount = {amount}
        rate = {rate}
        """
    )

    print("Second LLM response:")
    print(result2.tool_calls)


    # ------------------------------------------------
    # STEP 4: Execute second tool
    # ------------------------------------------------

    tool_call_2 = result2.tool_calls[0]

    if tool_call_2["name"] == "convertCurreny":

        converted_amount = convertCurreny.invoke(
            tool_call_2["args"]
        )

        print("Converted amount:", converted_amount)