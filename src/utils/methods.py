"""
StanceDetector - Methods Module
==============================

This module contains LangGraph implementation of classes for stance detection using different approaches:
1. Cola (Collaborative role-infused LLM-based agents) - A multi-agent approach for detailed stance analysis
2. CoS (Chain of Stance) - A step-by-step approach for structured stance analysis

Each approach provides different state management, input/output structures, and processing methods.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class ColaState(TypedDict):
    """
    Represents the state of a Cola (Collaborative role-infused LLM-based agents) process.

    This class maintains the complete state including input messages, targets, statements,
    analysis from different perspectives, arguments and final judgements.

    This structure is particularly useful for agents that maintain context over multiple interactions.

    Attributes:
        messages (list): Collection of messages in the conversation.
        target (str): The target entity being analyzed.
        statement (str): The statement being evaluated for stance.
        role (str): The role perspective for analysis.
        linguist_analysis (str): Analysis from linguistic perspective.
        expert_analysis (str): Analysis from domain expert perspective.
        user_analysis (str): Analysis from user perspective.
        pro_arguments (str): Arguments supporting the statement.
        anti_arguments (str): Arguments against the statement.
        neutral_arguments (str): Neutral arguments related to the statement.
        judgement (str): Final stance determination.
    """

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
    """
    Input structure for the Cola (Collaborative role-infused LLM-based agents) process.

    Contains the essential input parameters needed to initiate a stance analysis.

    Attributes:
        target (str): The target entity to analyze stance towards.
        statement (str): The statement to evaluate for stance.
        role (str): The role or perspective to use for analysis.
    """
    target: str
    statement: str
    role: str


class ColaOutput(TypedDict):
    """
    Output structure for the Cola (Collaborative role-infused LLM-based agents) process.

    Contains the results of the stance analysis process including different perspectives
    and the final judgement.

    Attributes:
        linguist_analysis (str): Analysis from a linguistic perspective.
        expert_analysis (str): Analysis from a domain expert perspective.
        user_analysis (str): Analysis from a user perspective.
        pro_arguments (list): Arguments supporting the statement.
        anti_arguments (list): Arguments against the statement.
        neutral_arguments (list): Neutral arguments related to the statement.
        judgement (str): Final stance determination e.g. Support/Against/Neutral.
    """
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

    Implements a multi-agent approach where different perspectives (linguist expert,
    domain specialist, social media veteran) are analyzed and combined to determine the
    stance towards a target entity.

    The analysis process flows through multiple specialized nodes in a graph structure,
    with each node contributing different aspects of analysis.
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

    # Multidimensional Text Analysis Stage
    def agent_node(self, state: ColaState,
                   sys_msg_template: SystemMessagePromptTemplate,
                   human_msg_template: HumanMessagePromptTemplate,
                   agent_name: str) -> ColaState:
        """
        Creates a node for general agent processing in the analysis graph.

        Returns:
            ColaState: An updated ColaState object with the new agent analysis.
        """
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
        """
        Creates a specialized node associated with the linguistic expert agent.

        Returns:
            Node: A configured linguist node for the analysis graph.
        """
        return self.agent_node(state, self.sys_message_linguist, self.human_message_linguist, 'linguist')

    def expert_node(self, state: ColaState) -> ColaState:
        """
        Creates a specialized node associated with the domain specialist agent.

        Returns:
            Node: A configured expert node for the analysis graph.
        """
        return self.agent_node(state, self.sys_message_expert, self.human_message_expert, 'expert')

    def user_node(self, state: ColaState) -> ColaState:
        """
        Creates a specialized node associated with the social media veteran agent.

        Returns:
            Node: A configured user node for the analysis graph.
        """
        return self.agent_node(state, self.sys_message_user, self.human_message_user, 'user')

    # Reasoning-Enhanced Debating Stage
    def analysis_node(self, state: ColaState, argument_type: str) -> ColaState:
        """
        Creates a node for comprehensive argument analysis.

        Analyzes the input from multiple perspectives and categorizes arguments
        into pro, anti, and neutral stances.

        Returns:
            Node: A configured analysis node for the analysis graph.
        """
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

    # Stance Conclusion Stage
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
        """
        Initializes and configures the complete analysis graph.

        Sets up all nodes and their connections to create the full analysis pipeline.

        Returns:
            Graph: The configured analysis graph ready for execution.
        """
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
    """
    Represents the state of a CoS (Chain of Stance) process.

    Maintains the complete state including input messages, targets, statements,
    and the analysis from each step in the process.

    This structure is particularly useful for agents that maintain context over multiple interactions.

    Attributes:
        messages (list): Collection of messages in the conversation.
        target (str): The target entity being analyzed.
        statement (str): The statement being evaluated for stance.
        step_one_analysis (str): Analysis from the first step.
        step_two_analysis (str): Analysis from the second step.
        step_three_analysis (str): Analysis from the third step.
        step_four_analysis (str): Analysis from the fourth step.
        step_five_analysis (str): Analysis from the fifth step.
        step_six_analysis (str): Analysis from the sixth step.
    """
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
    """
    Input structure for the CoS (Chain of Stance) process.

    Contains the essential input parameters needed to initiate a step-by-step stance analysis.

    Attributes:
        target (str): The target entity to analyze stance towards.
        statement (str): The statement to evaluate for stance.
    """
    target: str
    statement: str


class CosOutput(TypedDict):
    """
    Output structure for the CoS (Chain of Stance) process.

    Contains the results from each step of the analysis process.

    Attributes:
        step_one_analysis (str): Results from the first analysis step.
        step_two_analysis (str): Results from the second analysis step.
        step_three_analysis (str): Results from the third analysis step.
        step_four_analysis (str): Results from the fourth analysis step.
        step_five_analysis (str): Results from the fifth analysis step.
        step_six_analysis (str): Results from the sixth analysis step (final determination).
    """
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

    Implements a structured step-by-step approach to stance analysis, where each step
    builds on the previous ones to create a comprehensive understanding of the stance.

    The six steps involve (see Figure 1 of arXiv:2408.04649):
    1. Recognizing context
    2. Analyzing the main idea
    3. Emotional analysis
    4. Stance reinforcement
    5. Logical imference
    6. Stance determination
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
