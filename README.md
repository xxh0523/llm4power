# llm4power
llm4power is developed based on Azure OpenAI API, [CloudPSS SDK](https://docs.cloudpss.net/sdknew), and [Py_PSOPS](https://github.com/xxh0523/Py_PSOPS).

In this repository, 3 demos are provided. 

1. Retrieval-Augmented Generation (RAG) based on CloudPSS SDK Manual.
2. Power System Analysis Agent
3. CloudPSS SDK Code Generation

# Preparation
Before downloading this source code, you need to prepare your Azure OpenAI API. Get your terminal address, API key, and deployment name. 

At least two deployments are required. One is a text embedding deployment. The other is a chat model such as GPT-35-Turbo. 

# Requirements
Install the following pachages before running the demos. 
```
pip install cloudpss
pip install openai
pip install langchain
conda install qtwebkit
conda install streamlit
```

# 1. Retrieval-Augmented Generation (RAG) based on CloudPSS SDK Manual
RAG demo is in <u>**demo_1_rag_cloudpss_sdk.ipynb**</u>. 

You can ask questions about CloudPSS SDK and check the answers of fundamental LLM and RAG. 

# 2. Power System Analysis Agent
Running <u>**demo_2_agent_power.py**</u> with streamlit.

```
streamlit run demo_2_agent_power.py
```

Then a QA dialog will be started and you can ask questions such as "Calculate power flow and tell me the power flow results of generator 1, 5, and 9."

# 3. CloudPSS SDK Code Generation
Code generation demo is in <u>**demo_3_code_generation.ipynb**</u>.

You can ask LLM to generate the code you need. 

# Reference
[1] 丁俐夫, 陈颖, 肖谭南, 黄少伟, 沈沉. 大语言模型辅助的新型电力系统业务延展. 已投稿