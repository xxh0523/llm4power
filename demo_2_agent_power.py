from langchain.agents import AgentType, initialize_agent

from langchain.chains.conversation.memory import ConversationBufferMemory

import streamlit as st

from src.psops.py_psops import Py_PSOPS
psops_api = Py_PSOPS(rng=42)

def cal_power_flow(word: str):
    iter = psops_api.cal_power_flow_basic_nr()
    if iter <= 0: return 'power flow calculation do not converge, please check.'
    return f'power flow calculation is successfully converged after {iter} iterations.'

def get_gen_pf_result(word: str):
    if ',' in word: word = word.replace(',', '')
    str_list = word.split(' ')
    results = []
    for s in str_list:
        if s.isdigit(): 
            gen_no = int(s)
            result = psops_api.get_generator_lf_result(generator_no=gen_no)
            results.append(f'Generator {gen_no}: P = {result[0]} p.u., Q = {result[1]} p.u.')
    return results

# get_gen_pf_result('generator 1, 2, 3, 4')
from langchain.tools import Tool
tools = [
    Tool(
        name='Power Flow Calculation',
        func=lambda word: str(cal_power_flow(word)),
        description='use when you want to calculate power flow. ' + 
        'If the power flow converges, a postive integer will be returned showing the number of iterations needed for convergence. ' +
        'If the power flow failed, a non-positive integer will be returned.',
        # return_direct=True
    ),
    Tool(
        name='Get Power Flow Results of Generator',
        func=lambda word: str(get_gen_pf_result(word)),
        description='use when you want to get power flow results of a generator. ' + 
        'A list of active power and reactive power of the generator No. mentioned in the word will be returned.',
        # return_direct=True
    )
]

memory = ConversationBufferMemory(memory_key='chat_history')

from langchain.chat_models import AzureChatOpenAI
BASE_URL = "https://yourterminal.openai.azure.com/" #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!此处设置自己的terminal
API_KEY = "your api key" #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!此处设置自己的API Key
DEPLOYMENT_NAME = "your deployment name" #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!此处设置自己的deployment name
llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-05-15",
    deployment_name=DEPLOYMENT_NAME,
    model="gpt-35-turbo",
    openai_api_key=API_KEY,
    openai_api_type="azure",
    verbose=True,
)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


st.header(':blue[Langchain chatbot with agent/tools and memory] :sunglasses:')
user_input = st.text_input('You: ')
if 'memory' not in st.session_state:
    st.session_state['memory'] = ''

if st.button('Submit'):
    st.markdown(agent.run(input=user_input))
    st.session_state['memory'] += memory.buffer
    print(st.session_state['memory'])