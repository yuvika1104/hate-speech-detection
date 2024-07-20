# 450997460818-tcam6cv2bkn8qlihmui55vm5a7c2q01l.apps.googleusercontent.com
# https://youtu.be/K_CbgLpvH9E?si=a0yIf9JV7LcCWUjx
import streamlit as st

import pickle 
from final import vectorize

# Load the trained model
model = pickle.load(open('modelfile.pkl', 'rb'))

from googleapiclient.discovery import build

api_key = 'your api key'
youtube = build('youtube', 'v3', developerKey=api_key)

from urllib.parse import urlparse, parse_qs

def get_youtube_video_id(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    if 'v' in query_params:
        return query_params['v'][0]
    else:
        path_segments = parsed_url.path.split('/')
        if len(path_segments) == 2 and path_segments[1]:
            return path_segments[1]
        else:
            return None

def work(video_id):
    comments_request = youtube.commentThreads().list(part='snippet', videoId=video_id)
    comments_response = comments_request.execute()

    comments = []
    for item in comments_response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
        comments.append(comment)

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
    st.title("Hate Speech Detection on yt comments\n By Yuvika Gupta")
    html_temp=""" 
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Hate Speech Detection </h2>
    </div>"""
    st.markdown(html_temp, unsafe_allow_html=True)
    
    yt_url= st.text_input("Enter url of youtube video","...")
    if st.button("Detect"):
     video_id = get_youtube_video_id(yt_url)
     work(video_id)

        
if __name__=='__main__':
    main()