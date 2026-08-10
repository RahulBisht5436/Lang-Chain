from langchain_community.document_loaders import WebBaseLoader


# =========================================================
# 1. DEFINE THE WEBSITE URL
# =========================================================

# WebBaseLoader is used to fetch and load content
# from a web page.
#
# Instead of providing a local file path like:
#
#     "Data/company.txt"
#
# we provide the URL of the web page we want to load.
#
web_url = "https://demo-linkupmobile-website.telgoo5.com/"


# =========================================================
# 2. CREATE THE WEB LOADER
# =========================================================

# WebBaseLoader will:
#
# 1. Send a request to the given website
# 2. Retrieve the HTML content
# 3. Extract the relevant text
# 4. Convert the content into LangChain Document objects
#
loader = WebBaseLoader(web_path=web_url)


# =========================================================
# 3. LOAD THE WEBSITE CONTENT
# =========================================================

# load() fetches the web page and returns
# a list of LangChain Document objects.
#
# Each Document generally contains:
#
# Document(
#     page_content="Website text...",
#     metadata={
#         "source": "https://...",
#         ...
#     }
# )
#
docs = loader.load()


# =========================================================
# 4. DISPLAY THE LOADED DOCUMENTS
# =========================================================

print(docs)