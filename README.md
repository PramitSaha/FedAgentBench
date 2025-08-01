# FedAgentBench
The official codes for FedAgentBench: Towards Automating Real-world Federated Medical Image Analysis with Server–Client Agents"

Federated learning (FL) allows collaborative model training across healthcare sites without sharing sensitive patient data. However, real-world FL deployment is often hindered by complex operational challenges that demand substantial human efforts in cross-client coordination and data engineering, including: (a) selecting appropriate clients (hospitals), (b) coordinating between the central server and clients, (c) client-level data pre-processing, (d) harmonizing non-standardized data and labels across clients, and (e) selecting FL algorithms based on user instructions and cross-client data characteristics. These operational bottlenecks motivate the need for autonomous, agent-driven FL systems, where intelligent client agents at each client (hospital) and the central server agent collaboratively manage FL setup and model training with minimal human intervention. To this end, we introduce FedAgentBench, an agent-driven FL benchmark that captures key phases of real-world FL workflows from client selection to training completion, and evaluates the ability of LLM agents to autonomously coordinate healthcare FL. Our benchmark incorporates 40 widely adopted FL algorithms, each tailored to address diverse task-specific requirements and cross-client characteristics. Furthermore, we introduce a diverse set of complex tasks across 201 carefully curated datasets, simulating six modality-specific real-world healthcare environments, \textit{viz.}, Dermatoscopy, Ultrasound, Fundus, Histopathology, MRI, and X-Ray. We assessed the agentic performance of 10 open-source and 14 closed-source LLMs spanning small, medium, and large model scales. While some agents such as GPT-4.1 and DeepSeek V3 can automate various stages of the FL pipeline, our results reveal that more complex, interdependent tasks based on implicit goals remain challenging for even the strongest models. These findings underscore both the promise and current limitations of LLM agents for end-to-end automation in real-world FL healthcare systems. With FedAgentBench, we aim to unlock the full potential of cross-silo healthcare data by alleviating human-driven operational bottlenecks.

## System
Overview of our system **M3Builder**. From user’s free-text request to model delivery. The system integrates user requirements, a workspace with candidate data, tools, and code templates. A network of 4 specialized collaborative agents performs task analysis, data engineering, module assembling and training execution. A sample log tracks the Model Trainer agent’s activities during diagnosis model development.

## Setup

### Enviroment
We provide `requirements.txt` as a reference, the versions of packages are not compulsory.

### Data Preparation

* Create a folder named `ExternalDataset` locally.
* Put your custom dataset folder into `ExternalDataset`.
* We suggest you remove training/testing-irrelevant files from your dataset folder to avoid interference!

Your `ExternalDataset` folder should be like:
```
ExternalDataset
|   Dataset_1
|   |    images
|   |    masks
|   |    labels.csv
|   Dataset_2
|   |    Class1
|   |    Class2
|   |    labels.json
|   ......
```

## To Run **FedAgentBench** on Your Dataset
After environment setup and data preparation, you should first check all the files, and replace all 'path/to/sth' into your own paths.
Then, edit the `human_requirements` parameter in `run.sh` to your own requirements, and run:
```
./run.sh
```
Training logs and checkpoints will be placed under `TrainPipeline/Logout'.

## Acknowledgement
We sincerely thank all the contributors who developed relevant codes in our repository.



