import streamlit as st
import requests
import pickle 
from final import vectorize


# Load the trained model
model = pickle.load(open('modelfile.pkl', 'rb'))

# Your Facebook App credentials
app_id = 'your id'
app_secret = 'your app secret'
access_token ='your access token'

from urllib.parse import urlparse, parse_qs
def work(post_url):
  
# Parse the URL
  url_parts = urlparse(post_url)

# Extract the query parameters
  query_params = parse_qs(url_parts.query)

# Retrieve user ID and post ID from the query parameters
  user_id = query_params.get('id', [])[0] if 'id' in query_params else None
  post_id = query_params.get('story_fbid', [])[0] if 'story_fbid' in query_params else None

# Facebook Graph API endpoint for comments on a post
  comments_endpoint = f'https://graph.facebook.com/v15.0/{user_id}_{post_id}/comments?access_token={access_token}'

# Make a request to get comments
  comments_response = requests.get(comments_endpoint)
  comments_data = comments_response.json()

# Check if the comments exist and are accessible
  if 'error' in comments_data:
    print(f"Error: {comments_data['error']['message']}")
  else:
    # Extract comments
    comments = [comment.get('message') for comment in comments_data.get('data', [])]
    # Preprocess comments
    cleaned_comments = [vectorize([comment]) for comment in comments]

    # Use the trained model to predict the class of each comment
    predicted_classes = [model.predict(comment)[0] for comment in cleaned_comments]

    # Combine comments and predicted classes into pairs
    comment_class_pairs = zip(comments, predicted_classes)

    # Print or use the pairs as needed
    for comment, predicted_class in comment_class_pairs:
        st.success(f"\nComment: {comment}\nPredicted Class: {predicted_class}")
    
def main():
    st.title("Hate Speech Detection on Facebook comments\n By Yuvika Gupta")
    html_temp=""" 
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Hate Speech Detection </h2>
    </div>"""
    st.markdown(html_temp, unsafe_allow_html=True)

    post_url= st.text_input("Enter url for Facebook Post","...")
    if st.button("Detect"):
      work(post_url)

        
if __name__=='__main__':
    main()