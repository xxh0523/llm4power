# llm4power
llm4power is developed based on Azure OpenAI API, [CloudPSS SDK](https://docs.cloudpss.net/sdknew).

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
[1] 丁俐夫, 陈颖, **肖谭南**, 黄少伟, 沈沉. 基于大语言模型的新型电力系统生成式应用模式初探, 电力系统自动化, 审稿中. 

L. Ding, Y. Chen, **T. Xiao**, S. Huang, C. Shen, “A Novel Generative Application Mode for Power Systems Based on Large Language Models,” *Automation of Electric Power Systems*, Under Review. (in Chinese)

[2] Y. Song, Y. Chen, Z. Yu, S. Huang, and C. Shen, “CloudPSS: A high-performance power system simulator based on cloud computing,” Energy Reports, vol. 6, pp. 1611–1618, Dec. 2020, doi: [10.1016/j.egyr.2020.12.028](https://www.sciencedirect.com/science/article/pii/S2352484720317297).
