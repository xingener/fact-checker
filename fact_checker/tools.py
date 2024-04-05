from typing import List

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents import Tool, ZeroShotAgent
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

QUERY_ANALYSER_PROMPT = PromptTemplate.from_template("""You are a query analyser who is an expert at finding out implicit statements or declarations on the given search query: {query}.
There are three types of queries, and implicit statements examples under each type of queries:
1. Normal query: straightforward statement or declaration of something or someone. For example, "河北将实现缴费即参保" and "吞白糖可以下鱼刺", statements under this kind of queries is just the query itself.
2. Why-query: asking for the reason behind a certain statement or declaration. For example, "航空总医院为什么不是三甲", "少年锦衣卫破产的原因". Statements under the WHY-queries are "航空总医院不是三甲", "少年锦衣卫破产了".
3. What-query: asking about matters(what, how-many, how-long ...) related to a particular statement or declaration. For example, "胡歌刘亦菲新电影叫什么", "唯一没出过皇帝的省份". Statements under the Why-queries are "胡歌和刘亦菲合作了新电影", "存在唯一一个省份没出过皇帝".
Determine the type from the three types above for the given query.
The query may contain multiple statements, all of which need to be listed.

Do NOT generate any statement or declaration about there is someone who is asking the given query or the given query is asking something.
Do NOT try to answer the query.
Analyze strictly on the given query, and don't make things up yourself.
Generate the implicit statement in Chinese.
Use the following format:
Query type: query-type-you-determined
Statements: 给定query中包含的陈述(one or more)"""
)
QUERY_ANALYSER_DESCRIPTION = "Useful for when you need to analyse implicit statement within a given query."

def get_query_analyse_tool(llm=None):
    """
    QA TOOL
    从指定的语言模型中获取QueryAnalyser工具。
    
    Args:
        llm (optional[ncclient.manager.LLM]): ncclient库的连接管理器实例，用于获取语言模型。默认为None。
    
    Returns:
        ncclient.manager.Tool: QueryAnalyser工具类对象。
    
    """
    qa_chain = LLMChain(llm=llm, prompt=QUERY_ANALYSER_PROMPT)
    class QAInput(BaseModel):
        tool_input: str = Field()

    qa_tool = Tool(
        # args_schema=QAInput,
        name="QueryAnalyser",
        func=qa_chain.run,
        description=QUERY_ANALYSER_DESCRIPTION
    )
    return qa_tool


TODO_PREFIX = "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}. You have access to the following tools:"
TODO_FORMAT_INSTRUCTIONS = "You should come up with the first todo with QueryAnalyser tool to analyse the implicit claims if the query is a question. Use Chinese if do query fact-checking and the given query is in Chinese"
TODO_TOOL_DESCRIPTION = "Useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!"

def get_todo_tool(tools:List[Tool]):
    """
    从模板生成 FactCheckConcluder 工具对象。
    
    Args:
        llm (LanguageModel): 语言模型实例。
        
    Returns:
        Tool: 生成的 FactCheckConcluder 工具对象。
    
    """

    todo_prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=TODO_PREFIX,
        format_instructions = TODO_FORMAT_INSTRUCTIONS,
        input_variables = ["objective"]
    )
    todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
    todo_tool = Tool(
        name="TODO",
        func=todo_chain.run,
        description=TODO_TOOL_DESCRIPTION
    )
    return todo_tool


def get_struct_parser():
    """
    从给定的文件中获取ClaimFactCheckResult的解析器。
    
    Args:
        无参数。
    
    Returns:
        Parser: 获取到的解析器对象，为PydanticOutputParser类型。
    
    """
    class ClaimFactCheckResult(BaseModel):
        fact_check_result: bool = Field(default=None, description="true of false result of fact-checking to the given claim")
        explanation: str = Field(default=None, description="explanation to the fact-checking result")
        
        # You can add custom validation logic easily with Pydantic.
        # @validator('setup')
        # def question_ends_with_question_mark(cls, field):
        #     if field[-1] != '?':
        #         raise ValueError("Badly formed question!")
        #     return field

    parser = PydanticOutputParser(pydantic_object=ClaimFactCheckResult)
    return parser
    

ENTAILMENT_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["statement_and_basis"], 
    template= """You are a entailment recognition tool which determines a statement and a basis have a consistent, contradictory, or uncertain meaning.
Here are 3 options:
- ENTAILMENT: if the implicit statement is consistant with facts from 2rd step basis, give ENTAILMENT answer.
- CONTRADICTION: if the implicit statement and facts retrieved via 2rd step are highly conflictive, give CONTRADICTION answer.
- UNCERTAIN: if you have limited information about the implicit statements, give UNCERTAIN result.
---------------
Statement and basis(Separated by comma): {statement_and_basis}
Determine the entailment label =>
"""
)
ENTAILMENT_RECOGNITION_DESCRIPTION = """Useful for when you need to determine if a statement and a basis have a consistent meaning. The input to this tool should be like this:
the-statement,the-basis-content
The input should be a string, consisting of two parts connected by a comma."""

def get_entailment_recognition_tool(llm=None):
    entailment_chain = LLMChain(llm=llm, prompt=ENTAILMENT_RECOGNITION_PROMPT)
    er_tool = Tool(
        name="EntialmentRecognitionTool",
        func=entailment_chain.run,
        description=ENTAILMENT_RECOGNITION_DESCRIPTION
    )

    return er_tool
