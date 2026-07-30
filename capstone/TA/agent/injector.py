import TA.agent.ollama_patch

from langchain.agents import create_agent
from langchain.tools import BaseTool
from TA.agent.middleware import NodeMiddle
from TA.retrieval.middleware import RetrievalPolicyMiddleware
from core.schema.retrieval import RetrievalRunContext
from TA.tools.factory import ToolFactory
from langchain.agents.structured_output import ToolStrategy
from core.llm.llm_engine import CoreLLMEngine
from TA.agent.base import AGENT_SPECS


class AgentInjector:
    @staticmethod
    def initialize_all_agents(llm_engine: CoreLLMEngine, tools_factory: ToolFactory, config):
        """
        RAG agent: global tool-calling agent with NodeSchemaMiddleware.
        TA agent:  tool-calling agent with PDF + context tools + NodeSchemaMiddleware.
                   TA tools are re-bound per-request via session_context (see ta_module.py).
        Generator: raw LLM (no tool loop).
        """
        agents = {}

        for agent_name, agent_prompt, _schema in config: # schema deprecated 
            llm_instance = llm_engine._get_llm(agent_name)

            if agent_name in AGENT_SPECS:
                spec = AGENT_SPECS[agent_name]
                DEBUG = spec["debug"]
                agent_tools = spec["tools"](tools_factory)
                middleware = [NodeMiddle()]
                context_schema = None
                if agent_name == "RAG":
                    middleware.append(RetrievalPolicyMiddleware())
                    context_schema = RetrievalRunContext
                agent = create_agent(
                    model=llm_instance,
                    tools=agent_tools,
                    system_prompt=agent_prompt,
                    response_format=ToolStrategy(schema=spec["schema"]),
                    middleware=middleware,
                    context_schema=context_schema,
                    debug=DEBUG,
                    name=agent_name,
                )
                agent.model = llm_instance
                agent.system_prompt_text = agent_prompt  ## finish nodes stream raw model, agent sys prompt bypassed
                agents[agent_name] = agent
            else:
                # Generator and any future raw-LLM agents
                agents[agent_name] = llm_instance

        return agents
