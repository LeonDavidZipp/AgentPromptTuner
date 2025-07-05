from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Literal, Any
from .anthropic_callback_handler import AnthropicCallbackHandler


currently_supported_providers = [
	"openai",
]


def assign_handler(
	model_provider: Literal[
		"openai",
		"anthropic",
		"azure_openai",
		"azure_ai",
		"google_vertexai",
		"google_genai",
		"bedrock",
		"bedrock_converse",
		"cohere",
		"fireworks",
		"together",
		"mistralai",
		"huggingface",
		"groq",
		"ollama",
		"google_anthropic_vertex",
		"deepseek",
		"ibm",
		"nvidia",
		"xai",
		"perplexity",
	],
) -> BaseCallbackHandler:
	match model_provider:
		case "openai":
			return OpenAICallbackHandler()
		case "anthropic":
			return AnthropicCallbackHandler()
		case "azure_openai":
			if "azure_openai" not in currently_supported_providers:
				raise NotImplementedError(
					"AzureOpenAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "azure_ai":
			if "azure_ai" not in currently_supported_providers:
				raise NotImplementedError(
					"AzureAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "google_vertexai":
			if "google_vertexai" not in currently_supported_providers:
				raise NotImplementedError(
					"GoogleVertexAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "google_genai":
			if "google_genai" not in currently_supported_providers:
				raise NotImplementedError(
					"GoogleGenAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "bedrock":
			if "bedrock" not in currently_supported_providers:
				raise NotImplementedError(
					"BedrockCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "bedrock_converse":
			if "bedrock_converse" not in currently_supported_providers:
				raise NotImplementedError(
					"BedrockConverseCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "cohere":
			if "cohere" not in currently_supported_providers:
				raise NotImplementedError(
					"CohereCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "fireworks":
			if "fireworks" not in currently_supported_providers:
				raise NotImplementedError(
					"FireworksCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "together":
			if "together" not in currently_supported_providers:
				raise NotImplementedError(
					"TogetherCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "mistralai":
			if "mistralai" not in currently_supported_providers:
				raise NotImplementedError(
					"MistralAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "huggingface":
			if "huggingface" not in currently_supported_providers:
				raise NotImplementedError(
					"HuggingfaceCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "groq":
			if "groq" not in currently_supported_providers:
				raise NotImplementedError(
					"GroqCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "ollama":
			if "ollama" not in currently_supported_providers:
				raise NotImplementedError(
					"OllamaCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "google_anthropic_vertex":
			if "google_anthropic_vertex" not in currently_supported_providers:
				raise NotImplementedError(
					"GoogleAnthropicVertexCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "deepseek":
			if "deepseek" not in currently_supported_providers:
				raise NotImplementedError(
					"DeepseekCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "ibm":
			if "ibm" not in currently_supported_providers:
				raise NotImplementedError(
					"IBMCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "nvidia":
			if "nvidia" not in currently_supported_providers:
				raise NotImplementedError(
					"NVIDIACallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "xai":
			if "xai" not in currently_supported_providers:
				raise NotImplementedError(
					"XAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case "perplexity":
			if "perplexity" not in currently_supported_providers:
				raise NotImplementedError(
					"PerplexityCallbackHandler is not implemented yet. Please use OpenAICallbackHandler."
				)
		case _:
			raise ValueError(
				f"Unsupported model provider: {model_provider}. Supported providers are: {', '.join(currently_supported_providers)}"
			)
	raise ValueError(
		f"Unsupported model provider: {model_provider}. Supported providers are: {', '.join(currently_supported_providers)}"
	)


class CallbackHandler(BaseCallbackHandler):
	def __init__(
		self,
		model_provider: Literal[
			"openai",
			"anthropic",
			"azure_openai",
			"azure_ai",
			"google_vertexai",
			"google_genai",
			"bedrock",
			"bedrock_converse",
			"cohere",
			"fireworks",
			"together",
			"mistralai",
			"huggingface",
			"groq",
			"ollama",
			"google_anthropic_vertex",
			"deepseek",
			"ibm",
			"nvidia",
			"xai",
			"perplexity",
		],
	):
		super().__init__()
		self.handler = assign_handler(model_provider)

	def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
		return self.handler.on_llm_end(response, **kwargs)

	@property
	def total_cost(self) -> float:
		"""Get total cost from the underlying handler."""
		return self.handler.total_cost  # type: ignore[return-value]

	@property
	def total_tokens(self) -> int:
		"""Get total tokens from the underlying handler."""
		return self.handler.total_tokens  # type: ignore[return-value]

	@property
	def prompt_tokens(self) -> int:
		"""Get prompt tokens from the underlying handler."""
		return self.handler.prompt_tokens  # type: ignore[return-value]

	@property
	def prompt_tokens_cached(self) -> int:
		"""Get cached prompt tokens from the underlying handler."""
		return getattr(self.handler, "prompt_tokens_cached", 0)

	@property
	def completion_tokens(self) -> int:
		"""Get completion tokens from the underlying handler."""
		return self.handler.completion_tokens  # type: ignore[return-value]

	@property
	def reasoning_tokens(self) -> int:
		"""Get reasoning tokens from the underlying handler (Anthropic-specific)."""
		return getattr(self.handler, "reasoning_tokens", 0)

	@property
	def successful_requests(self) -> int:
		"""Get successful requests count from the underlying handler."""
		return self.handler.successful_requests  # type: ignore[return-value]


class AzureOpenAICallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"AzureOpenAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class AzureAICallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"AzureAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class GoogleVertexAICallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"GoogleVertexAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class GoogleGenAICallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"GoogleGenAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class BedrockCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"BedrockCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class BedrockConverseCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"BedrockConverseCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class CohereCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"CohereCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class FireworksCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"FireworksCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class TogetherCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"TogetherCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class MistralAICallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"MistralAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class HuggingfaceCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"HuggingfaceCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class GroqCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"GroqCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class OllamaCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"OllamaCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class GoogleAnthropicVertexCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"GoogleAnthropicVertexCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class DeepseekCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"DeepseekCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class IBMCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"IBMCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class NVIDIACallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"NVIDIACallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class XAICallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"XAICallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)


class PerplexityCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"PerplexityCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)
