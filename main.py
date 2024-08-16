import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from newsapi import NewsApiClient
import dotenv

# Load the .env file
dotenv.load_dotenv()

# Get the value of the MY_KEY environment variable
key = os.environ['NEWS_API-KEY']
newsapi = NewsApiClient(api_key=key)
openai_api_key = os.getenv('OPENAI_API_KEY')

if not key or not openai_api_key:
    st.error("API keys for NewsAPI or OpenAI are missing.")

# Initialize Streamlit app
st.title("News Analysis")

# Get stock name or keyword from user
stock_name = st.text_input("Enter the stock name or keyword for news search:")

# Process the stock name and retrieve articles
if st.button("Fetch News Articles") and stock_name:
    st.info("Fetching news articles...")
    
    # Get articles from the News API
    all_articles = newsapi.get_everything(
        q=stock_name,
        sources='bbc-news,the-verge',
        domains='bbc.co.uk,techcrunch.com',
        language='en',
        sort_by='publishedAt',
        page=1,
        page_size=10
    )

    if all_articles['totalResults'] == 0:
        st.warning("No articles found.")
    else:
        # Combine all articles into a single string
        articles_text = " ".join([article['title'] + " " + article['description'] for article in all_articles['articles']])
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        st.info("Splitting text...")
        docs = text_splitter.split_text(articles_text)

        # Convert chunks to Document objects
        documents = [Document(page_content=doc) for doc in docs]

        # Create embeddings and store them in a FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(documents, embeddings)
        st.info("Building embedding vectors...")
        time.sleep(2)

        # Save the FAISS index to a file
        file_path = "faiss_store_openai.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
        st.success("Processing complete!")

# Handle user query
query = st.text_input("Enter your question:")
if query:
    if os.path.exists("faiss_store_openai.pkl"):
        with open("faiss_store_openai.pkl", "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(api_key=openai_api_key, temperature=0.9, max_tokens=500), retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display the answer and sources
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
    else:
        st.error("No processed data found. Please fetch articles first.")
