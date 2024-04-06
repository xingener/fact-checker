# # Claim Fact-Checker

import os
import sys
import time

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from loguru import logger

from fact_checker.agent_prompts import CHAT_MESSAGE
from fact_checker.tools import get_entailment_recognition_tool, get_query_analyse_tool

logger.add(sys.stdout, level="INFO")

class FactChecker(object):
    
    def __init__(
        self,
        model="gpt-3.5-turbo",
        temperature=0.2,
        openai_api_key=None,
        max_retries=3,
        ):
        
        llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=openai_api_key,
            max_retries=max_retries)
        qa_tool = get_query_analyse_tool(llm=llm)
        er_tool = get_entailment_recognition_tool(llm=llm)
        wp_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        tools = [
            qa_tool,
            er_tool,
            wp_tool
        ]
        prompt = ChatPromptTemplate.from_messages(CHAT_MESSAGE)
        agent = create_openai_tools_agent(llm, tools, prompt)
        self.agent = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def check(self,
              statement,
              human_prefix='Fact-check this query with tools:',
              max_attempts=3,
              delay_seconds=1):
        human_prompt = f"{human_prefix}[{statement}]"
        attempts = 0
        result = None
        while attempts < max_attempts:
            try:
                result = self.agent.invoke({'input': human_prompt})
                attempts = max_attempts
            except Exception as e:
                print(f"statement: {statement}. Attempt {attempts + 1} failed. Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            attempts += 1
        return result

