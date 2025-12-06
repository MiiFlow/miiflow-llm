# Changelog

All notable changes to miiflow-llm will be documented here.

## [0.2.0] - 2025-12-05

### Added
- Enhanced system prompts for native tool calling with improved guidance

### Changed
- ReAct orchestrator now exclusively uses native tool calling (removed legacy XML-only path)
- Simplified orchestrator architecture by removing `use_native_tools` parameter
- Enhanced provider implementations (Gemini, OpenAI, Anthropic) for better compatibility
- Improved response handling with XML tag sanitization for cleaner outputs
- Agent.stream() now automatically injects user queries into message context

### Fixed
- Improved null safety in classification logic for edge cases
- Better error handling for unexpected classification responses
- Enhanced schema generation for array types and complex parameters

## [0.1.0] - 2025-11-30

### Added
- Unified interface for 9 LLM providers (OpenAI, Anthropic, Google Gemini, Groq, OpenRouter, Mistral, xAI, Ollama, Bedrock)
- Support for latest models (GPT-4o, Claude 3.5 Sonnet, Gemini 2.0)
- Streaming with unified StreamChunk format
- ReAct agents with native tool calling
- Plan & Execute orchestrator for complex multi-step tasks
- Tool system with @tool decorator and automatic schema generation
- Context injection patterns (Pydantic AI compatible)
- Multi-modal support (text + images)
- Async and sync APIs
- Full type hints with generics
- Comprehensive error handling with retry logic
- Token usage tracking and metrics
- Observability support (OpenTelemetry, Prometheus, Arize Phoenix)

### Documentation
- Quickstart guide
- Complete API reference
- Tool tutorial
- Agent tutorial (ReAct + Plan & Execute)
- Provider-specific documentation
- Contributing guidelines
- Code of conduct
