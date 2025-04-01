from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ColaState(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    target: str
    statement: str
    role: str
    # output
    linguist_analysis: str
    expert_analysis: str
    user_analysis: str
    pro_arguments: str
    anti_arguments: str
    neutral_arguments: str
    judgement: str


class ColaInput(TypedDict):
    target: str
    statement: str
    role: str


class ColaOutput(TypedDict):
    linguist_analysis: str
    expert_analysis: str
    user_analysis: str
    pro_arguments: str
    anti_arguments: str
    neutral_arguments: str
    judgement: str


class Cola:
    """
    A class which implements the 'Collaborative rOle-infused LLM-based Agents' (COLA)
    stance detection method from arXiv:2310.10467.

    """

    def __init__(self,
                 sys_linguist,
                 sys_expert,
                 sys_user,
                 hu_linguist,
                 hu_expert,
                 hu_user,
                 opinion_template,
                 judge_template,
                 agent_llm,
                 analysis_llm,
                 judge_llm):

        self.argument_to_stance = {
            'pro_arguments': 'Support',
            'anti_arguments': 'Against',
            'neutral_arguments': 'Neutral'
        }

        # Initialize message templates
        self.sys_message_linguist = SystemMessagePromptTemplate.from_template(sys_linguist)
        self.sys_message_expert = SystemMessagePromptTemplate.from_template(sys_expert)
        self.sys_message_user = SystemMessagePromptTemplate.from_template(sys_user)
        self.human_message_linguist = HumanMessagePromptTemplate.from_template(hu_linguist)
        self.human_message_expert = HumanMessagePromptTemplate.from_template(hu_expert)
        self.human_message_user = HumanMessagePromptTemplate.from_template(hu_user)
        self.message_opinion = HumanMessagePromptTemplate.from_template(opinion_template)
        self.message_judge = HumanMessagePromptTemplate.from_template(judge_template)

        # Initialize LLMs
        self.agent_llm = agent_llm
        self.analysis_llm = analysis_llm
        self.judge_llm = judge_llm

    def agent_node(self, state: ColaState,
                   sys_msg_template: SystemMessagePromptTemplate,
                   human_msg_template: HumanMessagePromptTemplate,
                   agent_name: str) -> ColaState:

        if sys_msg_template.input_variables:
            sys_message = sys_msg_template.format(**{k: state[k] for k in sys_msg_template.input_variables})
        else:
            sys_message = sys_msg_template.format()

        if human_msg_template.input_variables:
            human_message = human_msg_template.format(**{k: state[k] for k in human_msg_template.input_variables})
        else:
            human_message = human_msg_template.format()

        res = self.agent_llm.invoke([sys_message, human_message])

        return {f'{agent_name}_analysis': res.content if isinstance(res, BaseMessage) else res,
                'messages': [human_message, res]}

    def linguist_node(self, state: ColaState) -> ColaState:
        return self.agent_node(state, self.sys_message_linguist, self.human_message_linguist, 'linguist')

    def expert_node(self, state: ColaState) -> ColaState:
        return self.agent_node(state, self.sys_message_expert, self.human_message_expert, 'expert')

    def user_node(self, state: ColaState) -> ColaState:
        return self.agent_node(state, self.sys_message_user, self.human_message_user, 'user')

    def analysis_node(self, state: ColaState, argument_type: str) -> ColaState:
        message = self.message_opinion.format(
            stance=self.argument_to_stance[argument_type],
            target=state['target'],
            statement=state['statement'],
            role=state['role'],
            linguist_analysis=state['linguist_analysis'],
            expert_analysis=state['expert_analysis'],
            user_analysis=state['user_analysis']
        )
        res = self.analysis_llm.invoke([message])
        return {argument_type: res.content if isinstance(res, BaseMessage) else res,
                'messages': [message, res]}

    def pro_analysis(self, state: ColaState) -> ColaState:
        return self.analysis_node(state, 'pro_arguments')

    def anti_analysis(self, state: ColaState) -> ColaState:
        return self.analysis_node(state, 'anti_arguments')

    def neutral_analysis(self, state: ColaState) -> ColaState:
        return self.analysis_node(state, 'neutral_arguments')

    def judge(self, state: ColaState) -> ColaState:
        message = self.message_judge.format(
            target=state['target'],
            statement=state['statement'],
            pro_arguments=state['pro_arguments'],
            anti_arguments=state['anti_arguments'],
            neutral_arguments=state['neutral_arguments']
        )
        res = self.judge_llm.invoke([message])
        return {'judgement': res.content if isinstance(res, BaseMessage) else res,
                'messages': res}

    def initialize_graph(self):
        builder = StateGraph(ColaState, input=ColaInput, output=ColaOutput)

        # Add nodes
        builder.add_node('Linguist', self.linguist_node)
        builder.add_node('Expert', self.expert_node)
        builder.add_node('User', self.user_node)
        builder.add_node('ProAnalysis', self.pro_analysis)
        builder.add_node('AntiAnalysis', self.anti_analysis)
        builder.add_node('NeutralAnalysis', self.neutral_analysis)
        builder.add_node('Judgement', self.judge)

        # Add edges
        builder.add_edge(START, 'Linguist')
        builder.add_edge('Linguist', 'Expert')
        builder.add_edge('Expert', 'User')
        builder.add_edge('User', 'ProAnalysis')
        builder.add_edge('ProAnalysis', 'AntiAnalysis')
        builder.add_edge('AntiAnalysis', 'NeutralAnalysis')
        builder.add_edge('NeutralAnalysis', 'Judgement')
        builder.add_edge('Judgement', END)

        return builder.compile()


class CosState(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    target: str
    statement: str
    # output
    step_one_analysis: str
    step_two_analysis: str
    step_three_analysis: str
    step_four_analysis: str
    step_five_analysis: str
    step_six_analysis: str


class CosInput(TypedDict):
    target: str
    statement: str


class CosOutput(TypedDict):
    step_one_analysis: str
    step_two_analysis: str
    step_three_analysis: str
    step_four_analysis: str
    step_five_analysis: str
    step_six_analysis: str


class Cos:
    """
    A class which implements the 'Chain of Stance' stance detection
    method from arXiv:2408.04649.

    """
    def __init__(self,
                 step_one_template,
                 step_two_template,
                 step_three_template,
                 step_four_template,
                 step_five_template,
                 step_six_template,
                 llm):

        # Initialize message templates
        self.step_one_prompt = HumanMessagePromptTemplate.from_template(step_one_template)
        self.step_two_prompt = HumanMessagePromptTemplate.from_template(step_two_template)
        self.step_three_prompt = HumanMessagePromptTemplate.from_template(step_three_template)
        self.step_four_prompt = HumanMessagePromptTemplate.from_template(step_four_template)
        self.step_five_prompt = HumanMessagePromptTemplate.from_template(step_five_template)
        self.step_six_prompt = HumanMessagePromptTemplate.from_template(step_six_template)

        # Initialize LLMs
        self.agent_llm = llm

    def agent_node(self, state: CosState,
                   human_msg_template: HumanMessagePromptTemplate,
                   agent_name: str) -> CosState:

        if human_msg_template.input_variables:
            human_message = human_msg_template.format(**{k: state[k] for k in human_msg_template.input_variables})
        else:
            human_message = human_msg_template.format()

        res = self.agent_llm.invoke([*state['messages'], human_message])

        return {f'{agent_name}_analysis': res.content if isinstance(res, BaseMessage) else res,
                'messages': [human_message, res]}

    def first_node(self, state: CosState) -> CosState:
        return self.agent_node(state, self.step_one_prompt, 'step_one')

    def second_node(self, state: CosState) -> CosState:
        return self.agent_node(state, self.step_two_prompt, 'step_two')

    def third_node(self, state: CosState) -> CosState:
        return self.agent_node(state, self.step_three_prompt, 'step_three')

    def fourth_node(self, state: CosState) -> CosState:
        return self.agent_node(state, self.step_four_prompt, 'step_four')

    def fifth_node(self, state: CosState) -> CosState:
        return self.agent_node(state, self.step_five_prompt, 'step_five')

    def sixth_node(self, state: CosState) -> CosState:
        return self.agent_node(state, self.step_six_prompt, 'step_six')

    def initialize_graph(self):
        builder = StateGraph(CosState, input=CosInput, output=CosOutput)

        # Add nodes
        builder.add_node('Context', self.first_node)
        builder.add_node('MainIdea', self.second_node)
        builder.add_node('EmotionalAnalysis', self.third_node)
        builder.add_node('StanceReinforcement', self.fourth_node)
        builder.add_node('LogicalInference', self.fifth_node)
        builder.add_node('StanceDeterminism', self.sixth_node)

        # Add edges
        builder.add_edge(START, 'Context')
        builder.add_edge('Context', 'MainIdea')
        builder.add_edge('MainIdea', 'EmotionalAnalysis')
        builder.add_edge('EmotionalAnalysis', 'StanceReinforcement')
        builder.add_edge('StanceReinforcement', 'LogicalInference')
        builder.add_edge('LogicalInference', 'StanceDeterminism')
        builder.add_edge('StanceDeterminism', END)

        return builder.compile()
