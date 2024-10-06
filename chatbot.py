from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from data import prompt, config, language

import getpass
import os


# Chossing the model
model_language = input("Elige el modelo de lenguage que deseas utilizar entre los siguientes (introduce la letra en minúscula):")

open("technicaltest_ai_dxc/models","r") as file:
lines = file.readlines()
for i,line in enumerate(lines):
    print(f"({chr(i+ord("a"))})\t{line}")

print("Introduce tu clave de usuario para el lenguage seleccionado")
API_Key = getpass.getpass()

if model_language == "a":
    os.environ["OPENAI_API_Key"] = API_Key
    model = ChatOpenAI(model="gpt-3.5-turbo")

elif model_language == "b":
    os.environ["ANTHROPIC_API_Key"] = API_Key
    from langchain_anthropic import ChatAnthropic
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

elif model_language == "c":
    os.environ["AZURE_OPENAI_API_Key"] = API_Key
    from langchain_openai import AzureChatOpenAI
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

else:
    os.environ["GOOGLE_API_Key"] = API_Key
    from langchain_google_vertexai import ChatVertexAI
    model = ChatVertexAI(model="gemini-1.5-flash")



from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import SystemMessage, trim_messages


# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

def call_model(state: State):
    chain = prompt | model
    trimmed_messages = trimmer.invoke(state["messages"])
    response = chain.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    return {"messages": [response]}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory (for saving conversations)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)



# Loop to generate conersation:

query = input("¡Hola! ¿En qué puedo ayudarte?")

while query != "":
    #input_messages = messages + [HumanMessage(query)]
    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    output["messages"][-1].pretty_print()
    print("")

    query = input()

