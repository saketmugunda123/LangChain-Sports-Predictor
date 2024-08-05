from langchain_openai import ChatOpenAI #this is a class
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    api_key="sk-proj-v92TPuij1ajLf08g1MJXT3BlbkFJmZwwvjeFhREoBdGK87nJ",
    temperature=0.7, #represents the factaulity of the response (0 is factual 1 is creative)
    model="gpt-4" #model you are using
) 

prompt = ChatPromptTemplate.from_template("Tell me a a joke about a {subject}")

#create LLM Chain (output of prompt is input for llm)
chain = prompt | llm

response = chain.invoke({"subject": "KSI"})
#now the invoke parameter is the a dictonary of the parameters for prompt

print(response)