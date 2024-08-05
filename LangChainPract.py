from langchain_openai import ChatOpenAI #this is a class
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    api_key="sk-proj-v92TPuij1ajLf08g1MJXT3BlbkFJmZwwvjeFhREoBdGK87nJ"  
) 
#instaniating an object that can interact with OpenAI LLM through the OpenAI API
#That's why an API key is necessary so I can access API


response = llm.invoke("Do you know the sidemen?") 
# invoke parameter is the message you want to send to model
# return is the llm output
# The invoke function acts as a wrapper for the API

