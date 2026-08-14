from twikit import Client

# Initialize the open-source client
client = Client('en-US')

# Log in with your X credentials (or load saved cookies)
client.login(
    auth_info_1='your_username',
    auth_info_2='your_email',
    password='your_password'
)

# Search and get tweets based on a keyword
tweets = client.search_tweet('Open Source AI', product='Top')

for tweet in tweets:
    print(f"User: {tweet.user.name} | Text: {tweet.text}")

# ==========================================>>>
# @rahulBisht Need to Implement this at Home 