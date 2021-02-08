import pandas as pd
import streamlit as st 


#NLP PKG
import spacy
from textblob import TextBlob
#from gensim.summarization import summarize

#Sumy pkgs
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#Sumy Summary Fn

def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document ,3)
    summary_list = [str(sentence) for sentence in summary]
    result = " ".join(summary_list)
    return result




def text_analyzer(my_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(my_text)

    allData = ['"Token :" {} , \n Lemma : {}'.format(token.text , token.lemma_) for token in docx]
    return allData
   

def entity_analyzer(my_text):
    nlp = spacy.load("en_core_web_sm")
    docx = nlp(my_text)

    entities = [(entity.text , entity.label_) for entity in docx.ents] 
    return entities   

#Pkgs


def main():
    st.title("NLP with Streamlit")

    #Tokenization
    if st.checkbox("Show Token and Lemma"):
        st.subheader("Tokenize your Data")
        message = st.text_area("Enter your text" , "Type here.." , key = 1) 
        if st.button("Analyze" , key = 1):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)


    #Named Entity 
    if st.checkbox("Show Named  Entities"):
        st.subheader("Extract your Data")
        message = st.text_area("Enter your text" , "Type here.." , key = 2) 
        if st.button("Extract"):
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)

    #Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis :"):
        st.subheader("Sentiment of you text:")
        message = st.text_area("Enter your text" , "Type here.." , key = 3) 
        if st.button("Analyze" , key = 2):
            blob = TextBlob(message)
            sentiment_result = blob.sentiment

            st.json(sentiment_result)



    #Text Summarization
    if st.checkbox("Show Text Summarization :"):
        st.subheader("Summary of your text:")
        message = st.text_area("Enter your text" , "Type here.." , key = 4) 
        #summary_options = st.selectbox("Choose your Summarizer" ,["gensim" ,"sumy"])
    
        if st.button("Summarize"):
            #if summary_options == "gensim":
            #    st.text("using gensim")
            #    summary_result = summarize(message)
            #elif summary_options == "sumy":
            st.text("using sumy") 
            summary_result = sumy_summarizer(message)
            #else:
            #    st.warning("Using default Summarizer")
            #    st.text("using gensim")
            #    summary_result = summarize(message)
            st.success(summary_result)         


            
  





if __name__ == "__main__":
    main()


