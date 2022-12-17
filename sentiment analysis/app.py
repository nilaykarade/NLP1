import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle

# Defining dictionary containing all emojis with their meanings.

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}


sw_nltk = stopwords.words('english')


st.title("Tweeter Sentiment Analysis")
tweet= st.text_input("Enter a tweet")
chk_sentiment=st.button("Check Sentiment")


def text_preprocessing(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    #patterns for 3 or more consecutive letters by 2 letter.
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for textdata in textdata:
        textdata = textdata.lower()
        
        # Replace all URls with 'URL'
        textdata = re.sub(urlPattern,' URL',textdata)
        # Replace all emojis.
        for emoji in emojis.keys():
            textdata = textdata.replace(emoji, "EMOJI" + emojis[emoji])        
        # Replace @USERNAME to 'USER'.
        textdata = re.sub(userPattern,' USER', textdata)        
        # Replace all non alphabets.
        textdata = re.sub(alphaPattern, " ", textdata)
        # Replace 3 or more consecutive letters by 2 letter.
        textdata = re.sub(sequencePattern, seqReplacePattern, textdata)

        tweettext = ''
        for word in textdata.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if word not in sw_nltk:
                if len(word)>1:
                    # Lemmatizing the word.
                    word = wordLemm.lemmatize(word)
                    tweettext += (word+' ')            
        
    print(tweettext)    
    return tweettext

def predict_sentiment(tweet):
    model = pickle.load(open('./models/rf_model.pkl', 'rb'))
    vectoriser = pickle.load(open('./models/tfidf.pkl', 'rb'))
    vectorised_tweet=vectoriser.transform([tweet])
    #print(vectorised_tweet)
    result=model.predict(vectorised_tweet)
    return result[0]

if chk_sentiment:
    tweet=text_preprocessing(tweet)
    
    sentiment=predict_sentiment(tweet)
    if sentiment==0:
        st.warning("NEGATIVE")
    else:        
        st.success("POSITIVE")
    
