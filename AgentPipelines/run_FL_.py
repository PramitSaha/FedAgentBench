from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import getpass
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import MessagesState, END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict, Literal
from IPython.display import display, Image
import argparse
import warnings
warnings.filterwarnings('ignore')
from langchain_community.chat_models import ChatDeepInfra

os.environ["DEEPINFRA_API_TOKEN"] = "..."
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
# os.environ['LANGCHAIN_API_KEY'] = '...'
# os.environ['LANGSMITH_PROJECT'] = 'FL_agent_trial'
# # os.environ["GOOGLE_API_KEY"] ="..."
# # os.environ["OPENAI_API_KEY"] ="..."
# groq_api_key="..."

from langchainTool import read_files, copy_directory, copy_files, write_file, edit_file, run_script, list_files_in_second_level, preview_file_content, run_selfclean_on_dataset, organize_into_subfolder, copy_folder, remove_other_files, list_folders, make_folder, copy_images, run_federated_method
from langchainTool_llama import read_file

###Models ###################################################################################################

# model = ChatOpenAI(model="o3-mini",  # or "gpt-4", or "gpt-3.5-turbo"
#     # temperature=0.3,      # your choice
#     max_tokens=4096       # optional
# )

# model = ChatDeepInfra(model="Qwen/Qwen3-30B-A3B")
model = ChatDeepInfra(model="deepseek-ai/DeepSeek-V3")

# model = ChatOpenAI(model="qwen-qwq-32b", #llama3-8b-8192",
#         base_url="https://api.groq.com/openai/v1",
#         api_key=groq_api_key)

# model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite") 
######################################################################################################



###Phase 1: client selection ###################################################################################################

def create_server_to_client_communication_prompt_round_1():

    system_prompt = """
    You are a server agent in a Federated Learning setup, responsible for communicating with the client agents.
    From the user requirement, only extract the task and modality information. 
    State this information and instruct the clients to respond with:
    - The name of the selected dataset (that matches the user requirement)
     """

    return system_prompt

def create_selector_prompt(description_path, server_instruction):

    system_prompt = f"""
    You are acting as a client agent in Federated Learning responsible for selecting the datasets in your client based on the server instructions: {server_instruction}.
    I provide you with a list of dataset descriptions: {description_path}, which is a json file that contains a list of dictionaries.
    Every dictionary contains following entries: ["dataset name", "dataset description", "dataset_path"].
    
    You have access to the tools:
    read_files: This function reads a script file (such as a Python file) so you can understand its content. 

    Here is the typical workflow you should follow:
    1. Use read_files to read {description_path}, understand its content.
    2. Choose all the datasets that match the server instructions. Remember, your choice should be mainly based on "dataset descriptions" entry.
    3. Return the chosen dataset names following {server_instruction}, so a downstream peer agent can know the information accurately. 
    IMPORTANT: Give it only in this template for each dataset: **Dataset Name** : .... If no suitable dataset for the given task exists, the client should return "no dataset" and clearly explain why before ending the conversation.
    4. Include <end> to end the conversation.
     """

    # system_prompt = f"""
    # You are acting as a client agent in Federated Learning responsible for selecting the datasets in your client based on the server instructions: {server_instruction}.
    # I provide you with a list of dataset descriptions: {description_path}, which is a json file that contains a list of dictionaries. Plan your workflow and solve the task:
    
    # You have access to the tool:
    # read_files: This function reads a script file (such as a Python file) so you can understand its content. 

    # Return the chosen dataset names following {server_instruction}, so a downstream peer agent can know the information accurately. 
    # IMPORTANT: Give it only in this template for each dataset: **Dataset Name** : .... If no suitable dataset for the given task exists, the client should return "no dataset" and clearly explain why before ending the conversation.
    # Include <end> to end the conversation.
    #  """

    return system_prompt

def create_server_to_client_communication_prompt_round_2(client_response):
    system_prompt = f"""
    You are acting as a server agent for communicating with the client agents in Federated Learning. Read the client response: {client_response} 
    If the client has returned one or more datasets, return the : "Approved. Prepare for training". 
    If the client has returned no dataset, return the message: "Client not needed for the task". 
    """
    # 4. Return a list of the dataset paths of the chosen datasets as client_list=[.].
    return system_prompt

### Phase 2: Data preparation ######################################################################################################

# def create_datacleaner_prompt(input_data_path, output_data_path, server_response_round_2, description_path):

#     system_prompt = f"""
#     You are a highly skilled data preparation and data cleaning agent specializing in the medical domain. You MUST do your tasks ONLY using the tools provided to you.
#     You MUST only follow the workflow given below sincerely and not bypass it.
#     I provide you with server instruction {server_response_round_2}. 
#     If the server mentions that the client is not needed, end the conversation and do NOT do do anything else. Instead, if it instructs to prepare for training, you have three tasks:
#     1. Check if the dataset in {input_data_path} is already organized in sub-folder format from dataset descriptions: {description_path}.
#         If not, organize the data by grouping images of each class into their respective subfolders in your destination path: {output_data_path}.
#     2. Remove all non-image files from each sub-folder.
#     3. Clean client data by removing (a) near duplicate samples, (b) off topic samples, (c) noisy label samples
    
#     Here is the typical workflow you MUST follow:
#     1. If the server instruction: {server_response_round_2} mentions that the client is not needed, print: "Client not needed <end>" and end the conversation. Do NOT do anything further.
#     2. Instead, if it instructs you to prepare for training,  DO THIS WITHOUT FAILURE: print: "Data cleaning starting".
#        After that, use "read_files" function to read and understand the dataset description file in {description_path}. 
#        Check from there, if the dataset in {input_data_path} is already organized as sub-folders. DO THIS WITHOUT FAILURE: print: "Data organization checked!". 
#        If yes, copy the folder to the destination folder {output_data_path} using the function "copy_folder" and go to step 4 below, skipping step 3.
#     3. If dataset is not organized as sub-folders, organize the data by using the organize_into_subfolder function by grouping images of each class into their respective subfolders in the destination data path: {output_data_path}. Print "Data reorganized into subfolders"
#     4. Go to each subfolder in the destination data path: {output_data_path} and remove all non-image files by using remove_other_files function. After it is done, print "Other files removed"
#     5. Flag (a) near duplicate samples, (b) off topic samples, (c) noisy label samples using run_selfclean_on_dataset function in the destination data path: {output_data_path}.

#     You have access to the tools:
#     read_files: This function reads a script file (such as a Python file) so you can understand its content. 
#     organize_into_subfolder: This function reads csv file, goes through the labels column, creates subfolders and groups images inside them based on labels column.
#     copy_folder: This function copies folder from source location to destination location.
#     remove_other_files: This function checks the file extension of all files in a given folder and deletes the files with non-image extensions.
#     run_selfclean_on_dataset: This function flags (a) near duplicate samples, (b) off topic samples, (c) noisy label samples. Use this to clean the dataset

#     Important rules you must follow:
#     - You MUST use the run_selfclean_on_dataset tool to clean data!
#     - You MUST NOT modify the raw images manually.
#     - You MUST conclude your work by writing: "Data Cleaning Complete" <end>.
#     """

def create_datacleaner_prompt(input_data_path, output_data_path, server_response_round_2, description_path):

    system_prompt = f"""
    You are a highly skilled data preparation and data cleaning agent specializing in the medical domain. You MUST do your tasks ONLY using the tools provided to you.
    You MUST plan the workflow based on the instruction given below sincerely and not bypass it.
    I provide you with server instruction {server_response_round_2}. 
    If the server mentions that the client is not needed, end the conversation and do NOT do do anything else. Instead, if it instructs to prepare for training, you have three tasks:
    1. Check if the dataset in {input_data_path} is already organized in sub-folder format from dataset descriptions: {description_path}.
        If not, organize the data by grouping images of each class into their respective subfolders in your destination path: {output_data_path}.
    2. Remove all non-image files from each sub-folder.
    3. Clean client data by removing (a) near duplicate samples, (b) off topic samples, (c) noisy label samples
    
    You have access to the following tools. Plan and reason how to use the following tools properly:
    read_files: This function reads a script file (such as a Python file) so you can understand its content. 
    organize_into_subfolder: This function reads csv file, goes through the labels column, creates subfolders and groups images inside them based on labels column.
    copy_folder: This function copies folder from source location to destination location.
    remove_other_files: This function checks the file extension of all files in a given folder and deletes the files with non-image extensions.
    run_selfclean_on_dataset: This function flags (a) near duplicate samples, (b) off topic samples, (c) noisy label samples. Use this to clean the dataset

    Important rules you must follow:
    - You MUST use the run_selfclean_on_dataset tool to clean data!
    - You MUST NOT modify the raw images manually.
    - You MUST conclude your work by writing: "Data Cleaning Complete" <end>.

    """

    # system_prompt = f"""
    # You are a highly skilled data preparation and data cleaning agent specializing in the medical domain. I provide you with server instruction {server_response_round_2}. 
    # If the server mentions that the client is not needed, end the conversation. If it instructs to prepare for training, you have three tasks:
    # 1. Check if the dataset in {input_data_path} is already organized in sub-folder format from dataset descriptions: {description_path}.
    #     If not, organize the data by grouping images of each class into their respective subfolders in your destination path: {output_data_path}.
    # 2. Remove all non-image files from each sub-folder.
    # 3. Clean client data by removing (a) near duplicate samples, (b) off topic samples, (c) noisy label samples

    # You have access to the tools:
    # read_files: This function reads a script file (such as a Python file) so you can understand its content. 
    # organize_into_subfolder: This function reads csv file, goes through the labels column, creates subfolders and groups images inside them based on labels column.
    # copy_folder: This function copies folder from source location to destination location.
    # remove_other_files: This function checks the file extension of all files in a given folder and deletes the files with non-image extensions.
    # run_selfclean_on_dataset: This function flags (a) near duplicate samples, (b) off topic samples, (c) noisy label samples. Use this to clean the dataset
    # clean_data: This function checks flagged samples from csv file and removes them.
    
    # Here is the typical workflow you should follow:
    # 1. If the server instruction: {server_response_round_2} mentions that the client is not needed, print <end> and end the conversation. Do NOT do anything further.
    # 2. Instead, if it instructs you to prepare for training, use "read_files" function to read and understand the dataset description file in {description_path}. Check from there, if the dataset in {input_data_path} is already organized as sub-folders. 
    #    If yes, copy the folder to the destination folder {output_data_path} using the function "copy_folder" and go to step 4 below, skipping step 3.
    # 3. If dataset is not organized as sub-folders, organize the data by grouping images of each class into their respective subfolders in the destination data path: {output_data_path} by using the organize_into_subfolder function.
    # 4. Go to each subfolder in the destination data path: {output_data_path} and remove all non-image files by using remove_other_files function. 
    # 5. Flag (a) near duplicate samples, (b) off topic samples, (c) noisy label samples using run_selfclean_on_dataset function.
    # 6. Remove the flagged samples using clean_data function.

    # Important rules you must follow:
    # - You MUST use the run_selfclean_on_dataset tool to clean data!
    # - You MUST NOT modify the raw images manually.
    # - You MUST clean using the CSV outputs only.
    # - You MUST conclude your work by writing: "Data Cleaning Complete" <end>.
    # """

    return system_prompt

###Phase 3: Label harmonization ######################################################################################################

def label_harmonizer_prompt(input_data_path, output_data_path):

    system_prompt = f"""
    You are an intelligent agent for medical image label harmonization in a Federated Learning setup.
    Your goal is to group existing class folders into harmonized target categories (e.g., 'malignant', 'benign') by reorganizing the folder structure. 
    This involves identifying the current class folders, mapping them to new target labels, and copying images accordingly.

    You have access to the tools:
    - list_folders(path): Returns a list of subfolder names in the given path.
    - make_folder(path): Creates a new directory at the specified path.
    - copy_images(src_folder, dst_folder): Copies all image files from the source to the destination folder.

    Here is the typical workflow you should follow:

    1. Inspect class structure: Use `list_folders("{input_data_path}")` to get all existing class folder names.
    2. Define label mapping: Based on user requirements (e.g., binary classification), decide how existing class names map to target classes (coarse labels like `'malignant'` and `'benign'`).
    3. Prepare new folders: For each target class, use `make_folder("{output_data_path}/<class_name>")` to create destination folders.
    4. Move data: For each source class, use `copy_images` to move all image files to their new harmonized folder.
    """

    return system_prompt

### Phase 4: Federated Learning #################################################################################################################

def FL_algorithm_selector_prompt(algorithm_description_path):

    system_prompt = f"""
    You are acting as a server agent in Federated Learning responsible for selecting the federated learning algorithm in your client based on the human user requirement.
    I provide you with a list of algorithm descriptions: {algorithm_description_path}, which is a json file that contains a list of dictionaries.
    Every dictionary contains following entries: ["algorithm", "Full Name", "Key idea"].
    
    You have access to the tools:
    read_files: This function reads a script file (such as a Python file) so you can understand its content. 

    Here is the typical workflow you should follow:
    1. Use read_files to read {algorithm_description_path}, understand its content.
    2. Choose the algorithm that best matches the server instructions. Remember, your choice should be mainly based on "Full Name", "Key idea" entries.
    3. Return the chosen algorithm as Algorithm Name: ....
    4. Include <end> to end the conversation.
     """
    return system_prompt

# def FL_algorithm_selector_prompt(algorithm_description_path):
#     system_prompt = f"""
#     You are a server agent in a Federated Learning setup responsible for selecting the most appropriate federated learning algorithm based on the human user’s task requirement.

#     You are provided with a list of algorithm descriptions in the file {algorithm_description_path}, formatted as a JSON list of dictionaries. Each dictionary contains information about an algorithm, including its name, full name, and key idea.

#     Your objective is to analyze the algorithm descriptions and identify the method that best aligns with the user’s intent. Focus primarily on the "Full Name" and "Key idea" fields to determine relevance.

#     You have access to the following tool:
#     - read_files: This function reads a script file (such as a Python file) so you can understand its content.

#     Once you have selected the most suitable algorithm, return it in the format:
#     Algorithm Name: <selected_algorithm>

#     Conclude your response with "<end>".
#     """
#     return system_prompt


# def FL_trainer_prompt(selected_algorithm, datasets):
def FL_trainer_prompt(project_directory, selected_algorithm):
    system_prompt = f"""

    You are a trainer agent that performs federated learning with selected clients using the chosen algorithm: {selected_algorithm} 
    You have access to the tools:
    run_federated_method: Runs the specified federated learning method 

    Use run_federated_method to run the specific federated learning algorithm: {selected_algorithm} and report results.

     """
    #   You are a trainer agent that performs federated learning with selected clients: {datasets} using the chosen algorithm: {selected_algorithm} 
    return system_prompt

####################################################################################################################


class Communicator_1_State(TypedDict):
    messages: list

class SelectorState(TypedDict):
    messages: list

class Communicator_2_State(TypedDict):
    messages: list

class DatacleanerState(TypedDict):
    messages: list

class DataharmonizerState(TypedDict):
    messages: list

class AlgorithmselectorState(TypedDict):
    messages: list

class TrainerState(TypedDict):
    messages: list  

####################################################################################################################

def should_continue_communicator_1(state: Communicator_1_State) -> Literal["communicator_tools_1", END]:
    messages = state['messages']
    last_message = messages[-1]

    if "<end>" in last_message.content:
        return END

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "communicator_tools_1"
    
    return END


def should_continue_selector(state: SelectorState) -> Literal["selector_tools", END]:
    messages = state['messages']
    last_message = messages[-1]

    if "<end>" in last_message.content:
        return END

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "selector_tools"
    
    return END

def should_continue_communicator_2(state: Communicator_2_State) -> Literal["communicator_tools_2", END]:
    messages = state['messages']
    last_message = messages[-1]

    if "<end>" in last_message.content:
        return END

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "communicator_tools_2"
    
    return END

def should_continue_datacleaner(state: DatacleanerState) -> Literal["datacleaner_tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    if "<end>" in last_message.content:
        return END
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "datacleaner_tools"
    
    return END

def should_continue_dataharmonizer(state: DataharmonizerState) -> Literal["dataharmonizer_tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    if "<end>" in last_message.content:
        return END
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "dataharmonizer_tools"
    
    return END

def should_continue_algorithmselector(state: AlgorithmselectorState) -> Literal["algorithmselector_tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    if "<end>" in last_message.content:
        return END
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "algorithmselector_tools"
    
    return END

def should_continue_trainer(state: TrainerState) -> Literal["trainer_tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    if "<end>" in last_message.content:
        return END
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "trainer_tools"
    
    return END

####################################################################################################################

communicator_tools_1=[]
selector_tools = [read_files]
communicator_tools_2=[]

datacleaner_tools =[read_files, run_selfclean_on_dataset, organize_into_subfolder, copy_folder, remove_other_files]
dataharmonizer_tools=[list_folders, make_folder, copy_images]

algorithmselector_tools=[read_files]
trainer_tools = [run_federated_method]

####################################################################################################################
####################################################################################################################

communicator_tool_1_node= ToolNode(communicator_tools_1)
selector_tool_node = ToolNode(selector_tools)
communicator_tool_2_node= ToolNode(communicator_tools_2)

datacleaner_tool_node = ToolNode(datacleaner_tools)
dataharmonizer_tool_node = ToolNode(dataharmonizer_tools)

algorithmselector_tool_node = ToolNode(algorithmselector_tools)
trainer_tool_node = ToolNode(trainer_tools)

####################################################################################################################

communicator_1 = create_react_agent(model, communicator_tool_1_node)
selector = create_react_agent(model, selector_tool_node)
communicator_2 = create_react_agent(model, communicator_tool_2_node)

datacleaner = create_react_agent(model, datacleaner_tool_node)
dataharmonizer = create_react_agent(model, dataharmonizer_tool_node)

algorithmselector = create_react_agent(model, algorithmselector_tool_node)
trainer = create_react_agent(model, trainer_tool_node)

####################################################################################################################

communicator_1_workflow= StateGraph(Communicator_1_State)
selector_workflow= StateGraph(SelectorState)
communicator_2_workflow= StateGraph(Communicator_2_State)

datacleaner_workflow = StateGraph(DatacleanerState)
dataharmonizer_workflow = StateGraph(DataharmonizerState)

algorithmselector_workflow = StateGraph(AlgorithmselectorState)
trainer_workflow = StateGraph(TrainerState)

####################################################################################################################

# build the communicator_1 workflow
communicator_1_workflow = StateGraph(Communicator_1_State)

communicator_1_workflow.add_node("communicator_1", communicator_1)
communicator_1_workflow.add_node("communicator_tools_1", communicator_tool_1_node)

communicator_1_workflow.add_edge(START, "communicator_1")
communicator_1_workflow.add_conditional_edges(
    "communicator_1",
    should_continue_communicator_1,
    {
        "communicator_tools_1": "communicator_tools_1",
        END: END
    }
)
communicator_1_workflow.add_edge("communicator_tools_1", "communicator_1")
communicator_1_graph = communicator_1_workflow.compile()


# build the selector workflow
selector_workflow = StateGraph(SelectorState)

selector_workflow.add_node("selector", selector)
selector_workflow.add_node("selector_tools", selector_tool_node)

selector_workflow.add_edge(START, "selector")
selector_workflow.add_conditional_edges(
    "selector",
    should_continue_selector,
    {
        "selector_tools": "selector_tools",
        END: END
    }
)
selector_workflow.add_edge("selector_tools", "selector")
selector_graph = selector_workflow.compile()


# build the communicator_2 workflow
communicator_2_workflow = StateGraph(Communicator_2_State)

communicator_2_workflow.add_node("communicator_2", communicator_2)
communicator_2_workflow.add_node("communicator_tools_2", communicator_tool_2_node)

communicator_2_workflow.add_edge(START, "communicator_2")
communicator_2_workflow.add_conditional_edges(
    "communicator_2",
    should_continue_communicator_2,
    {
        "communicator_tools_2": "communicator_tools_2",
        END: END
    }
)
communicator_2_workflow.add_edge("communicator_tools_2", "communicator_2")
communicator_2_graph = communicator_2_workflow.compile()


# build the datacleaner workflow
datacleaner_workflow = StateGraph(DatacleanerState)

datacleaner_workflow.add_node("datacleaner", datacleaner)
datacleaner_workflow.add_node("datacleaner_tools", datacleaner_tool_node)

datacleaner_workflow.add_edge(START, "datacleaner")
datacleaner_workflow.add_conditional_edges(
    "datacleaner",
    should_continue_datacleaner,
    {
        "datacleaner_tools": "datacleaner_tools",
        END: END
    }
)
datacleaner_workflow.add_edge("datacleaner_tools", "datacleaner")

datacleaner_graph = datacleaner_workflow.compile()



# build the dataharmonizer workflow
dataharmonizer_workflow = StateGraph(DataharmonizerState)

dataharmonizer_workflow.add_node("dataharmonizer", dataharmonizer)
dataharmonizer_workflow.add_node("dataharmonizer_tools", dataharmonizer_tool_node)

dataharmonizer_workflow.add_edge(START, "dataharmonizer")
dataharmonizer_workflow.add_conditional_edges(
    "dataharmonizer",
    should_continue_dataharmonizer,
    {
        "dataharmonizer_tools": "dataharmonizer_tools",
        END: END
    }
)
dataharmonizer_workflow.add_edge("dataharmonizer_tools", "dataharmonizer")

dataharmonizer_graph = dataharmonizer_workflow.compile()



# build the algorithmselector workflow
algorithmselector_workflow = StateGraph(AlgorithmselectorState)

algorithmselector_workflow.add_node("algorithmselector", algorithmselector)
algorithmselector_workflow.add_node("algorithmselector_tools", algorithmselector_tool_node)

algorithmselector_workflow.add_edge(START, "algorithmselector")
algorithmselector_workflow.add_conditional_edges(
    "algorithmselector",
    should_continue_algorithmselector,
    {
        "algorithmselector_tools": "algorithmselector_tools",
        END: END
    }
)
algorithmselector_workflow.add_edge("algorithmselector_tools", "algorithmselector")

algorithmselector_graph = algorithmselector_workflow.compile()



# build the trainer workflow
trainer_workflow = StateGraph(TrainerState)

trainer_workflow.add_node("trainer", trainer)
trainer_workflow.add_node("trainer_tools", trainer_tool_node)

trainer_workflow.add_edge(START, "trainer")
trainer_workflow.add_conditional_edges(
    "trainer",
    should_continue_trainer,
    {
        "trainer_tools": "trainer_tools",
        END: END
    }
)
trainer_workflow.add_edge("trainer_tools", "trainer")

trainer_graph = trainer_workflow.compile()

####################################################################################################################
import tiktoken

def count_tokens(text, model_name="gpt-4.1"):
    return len(str(text).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_requirements", type=str)
    args = parser.parse_args()

    Human_Requirements = args.human_requirements
    print("\nHuman Requirements: ", Human_Requirements, "\n---------------------------------------------------------------------\n")
    
#########################################################################################################################


    algorithm="..."

    folder_path = "..."
    trainer_prompt=FL_trainer_prompt(folder_path, algorithm)
    trainer_result = trainer_graph.invoke(
        {
            "messages": [
                SystemMessage(content=trainer_prompt)
            ]
        }
    )
    trainer_content = trainer_result['messages'][-1].content.replace("<end>", "")#
    # datacleaner_content = datacleaner_result['messages']
    print("trainer_content", trainer_content)
    # total_tokens = count_tokens(datacleaner_prompt_1) + count_tokens(datacleaner_result)
    # print(f"🔢 Total datacleaner_content tokens sent to model: {total_tokens}")


if __name__=="__main__":
    main()
