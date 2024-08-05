from langchain_openai import ChatOpenAI #this is a class
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # this is used to embedd the documents so you can store in vector database
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()

#This function bascially gets the raw data from the url. However, inputting this big ass thing into model leads costs too many tokens
def get_documents_from_web(url): 
    docLoader = WebBaseLoader(url) #This instansitaes the WebBaseLoader object
    docs = docLoader.load() #theis function basically acts as a getter function (is a list)

    splitter = RecursiveCharacterTextSplitter(  #initalize splitter
        chunk_size =200,
        chunk_overlap = 20
    ) 
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_vector_database(listDocs):
    embedding = OpenAIEmbeddings() #instantiate the embedding objects (it'll embed the split documents)
    vectorStore = FAISS.from_documents(listDocs, embedding=embedding) #creates the vector database with split docs
    return vectorStore

def create_chain(VectorDB):
    model = ChatOpenAI(
    temperature=0.7, #represents the factaulity of the response (0 is factual 1 is creative)
    model="gpt-4" #model you are using
    ) 

    prompt = ChatPromptTemplate.from_template("""
    Answer the users question: 
    Context: {context}
    Input: {input}                                      
    """)

    chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt
    )

    retriever = vectorDB.as_retriever() #method of the vectorDB type and makes it a retriever

    retrieval_chain = create_retrieval_chain(
        retriever, 
        chain
    )

    return retrieval_chain


#create LLM Chain (output of prompt is input for llm)
# chain = prompt | model

#same thing as the prompt | model except it can take in a list of documents as a parameter


docs = get_documents_from_web("https://app.prizepicks.com/")
vectorDB = create_vector_database(docs) #creates the vectorDB
chain = create_chain(vectorDB) #this creates a chain with the new context



response = chain.invoke({
    "input": "Are currently what is Derek Lively's points line based on the data in the vector db",
    "context": docs #docs is a list of documents
    })
#now the invoke parameter is the a dictonary of the parameters for prompt

print(response["answer"])
