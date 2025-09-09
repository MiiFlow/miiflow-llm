"""Tests for the unified Agent architecture - Complete LlamaIndex replacement."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from miiflow_llm.core.agent import (
    Agent, RunContext, RunResult, AgentType,
    DatabaseService, VectorStoreService, KnowledgeService
)
from miiflow_llm.core.client import LLMClient, ChatResponse
from miiflow_llm.core.message import Message, MessageRole
from miiflow_llm.core.metrics import TokenCount
from dataclasses import dataclass
from typing import Optional


# Test-specific dependency container (flexible implementation)
@dataclass
class MockDeps:
    """Test dependency container that implements the protocols."""
    
    db: DatabaseService
    vector_store: VectorStoreService
    knowledge: KnowledgeService
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None


class TestUnifiedAgentArchitecture:
    """Test the unified agent system that replaces LlamaIndex."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock LLMClient for testing."""
        client = MagicMock()
        client.provider_name = "openai"
        client.achat = AsyncMock()
        return client
    
    @pytest.fixture
    def llm_client(self, mock_client):
        """LLMClient instance with mock."""
        return LLMClient(mock_client)
    
    @pytest.fixture
    def mock_services(self):
        """Mock miiflow-web services."""
        db_service = MagicMock()
        db_service.query = AsyncMock(return_value=[{"result": "test_data"}])
        db_service.get_user_context = AsyncMock(return_value={"user": "test_user"})
        
        vector_service = MagicMock() 
        vector_service.similarity_search = AsyncMock(return_value=[
            {"content": "relevant doc 1", "title": "Doc 1"},
            {"content": "relevant doc 2", "title": "Doc 2"}
        ])
        vector_service.add_documents = AsyncMock()
        
        knowledge_service = MagicMock()
        knowledge_service.get_relevant_docs = AsyncMock(return_value=[{"doc": "relevant"}])
        knowledge_service.get_thread_context = AsyncMock(return_value={"thread": "context"})
        
        return {
            "db": db_service,
            "vector_store": vector_service,
            "knowledge": knowledge_service
        }
    
    @pytest.fixture
    def test_deps(self, mock_services):
        """MockDeps instance with mock services."""
        return MockDeps(
            db=mock_services["db"],
            vector_store=mock_services["vector_store"],
            knowledge=mock_services["knowledge"],
            user_id="test_user_123",
            thread_id="thread_456"
        )
    
    def test_agent_type_configuration(self, llm_client):
        """Test that different agent types are configured properly."""
        rag_agent = Agent(llm_client, agent_type=AgentType.RAG)
        assert rag_agent.max_iterations == 3
        
        workflow_agent = Agent(llm_client, agent_type=AgentType.WORKFLOW)
        assert workflow_agent.max_iterations == 20
        
        analysis_agent = Agent(llm_client, agent_type=AgentType.ANALYSIS)
        assert analysis_agent.temperature == 0.1
    
    def test_run_context_flexible_integration(self, test_deps):
        """Test RunContext with flexible dependency injection."""
        context = RunContext(
            deps=test_deps,
            user_id="user123", 
            thread_id="thread456",
            session_id="session789",
            metadata={"custom": "data"}
        )
        
        assert context.deps.user_id == "test_user_123"
        assert context.user_id == "user123"
        assert context.thread_id == "thread456"
        assert context.has_context("custom")
        assert not context.has_context("nonexistent")
    
    def test_flexible_agent_creation(self, llm_client):
        """Test flexible agent creation for different types."""
        chat_agent = Agent(llm_client, agent_type=AgentType.CHAT, deps_type=MockDeps)
        assert chat_agent.agent_type == AgentType.CHAT
        assert chat_agent.deps_type == MockDeps
        assert len(chat_agent._tools) == 0
        
        rag_agent = Agent(llm_client, agent_type=AgentType.RAG, deps_type=MockDeps)
        assert rag_agent.agent_type == AgentType.RAG
        assert rag_agent.max_iterations == 3
        
        search_agent = Agent(llm_client, agent_type=AgentType.SEARCH, deps_type=MockDeps)
        assert search_agent.agent_type == AgentType.SEARCH
    
    @pytest.mark.asyncio
    async def test_agent_with_dependency_injection(self, llm_client, mock_client, test_deps):
        """Test agent with flexible dependency injection."""
        agent = Agent(llm_client, agent_type=AgentType.CHAT, deps_type=MockDeps)
        
        @agent.tool(name="get_user_context", description="Get user context information")
        async def get_user_context(context: RunContext[MockDeps]) -> str:
            """Get user context from the database."""
            try:
                if context.deps and context.deps.user_id:
                    user_data = await context.deps.db.get_user_context(context.deps.user_id)
                    return f"User context: {user_data}"
                return "No user context available"
            except Exception as e:
                return f"Error getting user context: {e}"
        
        tool_call_response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{
                    "id": "call_123",
                    "function": {
                        "name": "get_user_context",
                        "arguments": '{}'
                    }
                }]
            ),
            usage=TokenCount(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            model="gpt-4",
            provider="openai"
        )
        
        final_response = ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content="Hello! I can see your user context."),
            usage=TokenCount(prompt_tokens=25, completion_tokens=8, total_tokens=33),
            model="gpt-4",
            provider="openai"
        )
        
        mock_client.achat.side_effect = [tool_call_response, final_response]
        
        result = await agent.run("Hello", deps=test_deps)
        
        assert isinstance(result, RunResult)
        assert result.data == "Hello! I can see your user context."
        
        test_deps.db.get_user_context.assert_called_once_with("test_user_123")
    
    @pytest.mark.asyncio
    async def test_rag_agent_knowledge_search(self, llm_client, mock_client, test_deps):
        """Test RAG agent using flexible vector store."""
        agent = Agent(llm_client, agent_type=AgentType.RAG, deps_type=MockDeps)
        
        @agent.tool(name="search_knowledge", description="Search the knowledge base")
        async def search_knowledge(context: RunContext[MockDeps], query: str) -> str:
            """Search the vector store for relevant documents."""
            try:
                if context.deps and context.deps.vector_store:
                    results = await context.deps.vector_store.similarity_search(query, k=5)
                    return f"Found {len(results)} relevant documents: {results}"
                return "Vector store not available"
            except Exception as e:
                return f"Error searching knowledge: {e}"
        
        tool_call_response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{
                    "id": "call_456", 
                    "function": {
                        "name": "search_knowledge",
                        "arguments": '{"query": "What is AI?"}'
                    }
                }]
            ),
            usage=TokenCount(prompt_tokens=30, completion_tokens=15, total_tokens=45),
            model="gpt-4",
            provider="openai"
        )
        
        final_response = ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content="Based on the knowledge search, AI is..."),
            usage=TokenCount(prompt_tokens=40, completion_tokens=12, total_tokens=52),
            model="gpt-4", 
            provider="openai"
        )
        
        mock_client.achat.side_effect = [tool_call_response, final_response]
        
        result = await agent.run("What is AI?", deps=test_deps)
        
        assert result.data == "Based on the knowledge search, AI is..."
        
        test_deps.vector_store.similarity_search.assert_called_once_with("What is AI?", k=5)
    
    def test_flexible_agent_with_custom_prompt(self, llm_client):
        """Test flexible agent creation with custom configuration."""
        chat_agent = Agent(llm_client, agent_type=AgentType.CHAT, deps_type=MockDeps)
        assert chat_agent.agent_type == AgentType.CHAT
        
        rag_agent = Agent(llm_client, agent_type=AgentType.RAG, deps_type=MockDeps)
        assert rag_agent.agent_type == AgentType.RAG
        
        custom_agent = Agent(
            llm_client, 
            agent_type=AgentType.ANALYSIS,
            deps_type=MockDeps,
            system_prompt="Custom analysis prompt"
        )
        assert custom_agent.system_prompt == "Custom analysis prompt"
    
    def test_agent_protocol_compatibility(self):
        """Test that miiflow-web services implement the required protocols."""
        class MockDBService:
            async def query(self, sql: str) -> list:
                return [{"test": "data"}]
            
            async def get_user_context(self, user_id: str) -> dict:
                return {"user": user_id}
        
        class MockVectorService:
            async def similarity_search(self, query: str, k: int = 5) -> list:
                return [{"content": "test"}]
            
            async def add_documents(self, documents: list) -> None:
                pass
        
        class MockKnowledgeService:
            async def get_relevant_docs(self, query: str) -> list:
                return [{"doc": "test"}]
            
            async def get_thread_context(self, thread_id: str) -> dict:
                return {"thread": thread_id}
        
        db_service: DatabaseService = MockDBService()
        vector_service: VectorStoreService = MockVectorService()
        knowledge_service: KnowledgeService = MockKnowledgeService()
        
        deps = MockDeps(
            db=db_service,
            vector_store=vector_service,
            knowledge=knowledge_service,
            user_id="test"
        )
        
        assert deps.user_id == "test"


class TestLlamaIndexReplacementPatterns:
    """Test patterns that specifically replace LlamaIndex functionality."""
    
    def test_rag_pattern_replacement(self):
        """Test that RAG agent replaces LlamaIndex RAG patterns."""
        from miiflow_llm.core.client import LLMClient
        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        llm_client = LLMClient(mock_provider)
        
        agent = Agent(llm_client, agent_type=AgentType.RAG)
        
        @agent.tool("search_knowledge")
        async def search_knowledge(query: str, context) -> str:
            return f"Searching for: {query}"
            
        @agent.tool("get_thread_context")  
        async def get_thread_context(context) -> str:
            return "Thread context"
        
        assert "search_knowledge" in agent.tool_registry.tools
        assert "get_thread_context" in agent.tool_registry.tools
    
    def test_search_pattern_replacement(self):
        """Test that search agent replaces LlamaIndex search patterns.""" 
        from miiflow_llm.core.client import LLMClient
        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        llm_client = LLMClient(mock_provider)
        
        agent = Agent(llm_client, agent_type=AgentType.SEARCH)
        
        @agent.tool("semantic_search")
        async def semantic_search(query: str, context) -> str:
            return f"Search results for: {query}"
        
        assert "semantic_search" in agent.tool_registry.tools
    
    def test_workflow_pattern_replacement(self):
        """Test that workflow agent replaces LlamaIndex workflow patterns."""
        from miiflow_llm.core.client import LLMClient
        mock_provider = MagicMock()
        mock_provider.provider_name = "test"
        llm_client = LLMClient(mock_provider)
        
        agent = Agent(llm_client, agent_type=AgentType.WORKFLOW)
        
        @agent.tool("execute_step")
        async def execute_step(step: str, context) -> str:
            return f"Executed step: {step}"
        
        assert agent.max_iterations == 20
        assert "execute_step" in agent.tool_registry.tools


class TestImportPatterns:
    """Test that imports work correctly for miiflow-web."""
    
    def test_core_imports(self):
        """Test that miiflow-web can import everything it needs."""
        from miiflow_llm.core import Agent, RunContext, RunResult, AgentType
        from miiflow_llm.core import DatabaseService, VectorStoreService, KnowledgeService
        
        assert Agent is not None
        assert RunContext is not None
        assert AgentType.CHAT is not None
    
    def test_agent_typing_support(self):
        """Test that proper type annotations work.""" 
        from miiflow_llm.core import Agent
        
        def create_typed_agent(client) -> Agent[MockDeps, str]:
            return Agent(client, deps_type=MockDeps, result_type=str)
        
        agent_type = Agent[MockDeps, str]
        assert agent_type is not None
