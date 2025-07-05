from langchain_community.callbacks.openai_info import OpenAICallbackHandler, TokenType
from typing import override, Any, Self

from langchain_core.outputs import LLMResult
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core._api import warn_deprecated


MODEL_COST_PER_1K_TOKENS: dict[str, float] = {}


# TODO: implement
def standardize_model_name(
	model_name: str,
	is_completion: bool = False,
	*,
	token_type: TokenType = TokenType.PROMPT,
) -> str:
	"""
	Standardize the model name to a format that can be used in the OpenAI API.

	Args:
		model_name: Model name to standardize.
		is_completion: Whether the model is used for completion or not.
			Defaults to False. Deprecated in favor of ``token_type``.
		token_type: Token type. Defaults to ``TokenType.PROMPT``.

	Returns:
		Standardized model name.

	"""
	if is_completion:
		warn_deprecated(
			since="0.3.13",
			message=(
				"is_completion is deprecated. Use token_type instead. Example:\n\n"
				"from langchain_community.callbacks.openai_info import TokenType\n\n"
				"standardize_model_name('gpt-4o', token_type=TokenType.COMPLETION)\n"
			),
			removal="1.0",
		)
		token_type = TokenType.COMPLETION
	model_name = model_name.lower()
	if ".ft-" in model_name:
		model_name = model_name.split(".ft-")[0] + "-azure-finetuned"
	if ":ft-" in model_name:
		model_name = model_name.split(":")[0] + "-finetuned-legacy"
	if "ft:" in model_name:
		model_name = model_name.split(":")[1] + "-finetuned"
	if token_type == TokenType.COMPLETION and (
		model_name.startswith("gpt-4")
		or model_name.startswith("gpt-3.5")
		or model_name.startswith("gpt-35")
		or model_name.startswith("o1-")
		or model_name.startswith("o3-")
		or model_name.startswith("o4-")
		or ("finetuned" in model_name and "legacy" not in model_name)
	):
		return model_name + "-completion"
	if (
		token_type == TokenType.PROMPT_CACHED
		and (
			model_name.startswith("gpt-4o")
			or model_name.startswith("gpt-4.1")
			or model_name.startswith("o1")
			or model_name.startswith("o3")
			or model_name.startswith("o4")
		)
		and not (model_name.startswith("gpt-4o-2024-05-13"))
	):
		return model_name + "-cached"
	else:
		return model_name


# TODO: implement
def get_anthropic_token_cost_for_model(
	model_name: str,
	num_tokens: int,
	is_completion: bool = False,
	*,
	token_type: TokenType = TokenType.PROMPT,
) -> float:
	"""
	Get the cost in USD for a given model and number of tokens.

	Args:
		model_name: Name of the model
		num_tokens: Number of tokens.
		is_completion: Whether the model is used for completion or not.
			Defaults to False. Deprecated in favor of ``token_type``.
		token_type: Token type. Defaults to ``TokenType.PROMPT``.

	Returns:
		Cost in USD.
	"""
	if is_completion:
		warn_deprecated(
			since="0.3.13",
			message=(
				"is_completion is deprecated. Use token_type instead. Example:\n\n"
				"from langchain_community.callbacks.openai_info import TokenType\n\n"
				"get_anthropic_token_cost_for_model('gpt-4o', 10, token_type=TokenType.COMPLETION)\n"  # noqa: E501
			),
			removal="1.0",
		)
		token_type = TokenType.COMPLETION
	model_name = standardize_model_name(model_name, token_type=token_type)
	if model_name not in MODEL_COST_PER_1K_TOKENS:
		raise ValueError(
			f"Unknown model: {model_name}. Please provide a valid OpenAI model name."
			"Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
		)
	return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)


class AnthropicCallbackHandler(OpenAICallbackHandler):
	def __init__(self):
		super().__init__()
		raise NotImplementedError(
			"AnthropicCallbackHandler is not implemented yet. Please use OpenAICallbackHandler or another provider's callback handler."
		)

	@override
	def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
		"""Collect token usage."""
		# Check for usage_metadata (langchain-core >= 0.2.2)
		try:
			generation = response.generations[0][0]
		except IndexError:
			generation = None
		if isinstance(generation, ChatGeneration):
			try:
				message = generation.message
				if isinstance(message, AIMessage):
					usage_metadata = message.usage_metadata
					response_metadata = message.response_metadata
				else:
					usage_metadata = None
					response_metadata = None
			except AttributeError:
				usage_metadata = None
				response_metadata = None
		else:
			usage_metadata = None
			response_metadata = None

		prompt_tokens_cached = 0
		reasoning_tokens = 0

		if usage_metadata:
			token_usage = {"total_tokens": usage_metadata["total_tokens"]}
			completion_tokens = usage_metadata["output_tokens"]
			prompt_tokens = usage_metadata["input_tokens"]
			if response_model_name := (response_metadata or {}).get("model_name"):
				model_name = standardize_model_name(response_model_name)
			elif response.llm_output is None:
				model_name = ""
			else:
				model_name = standardize_model_name(
					response.llm_output.get("model_name", "")
				)
			if "cache_read" in usage_metadata.get("input_token_details", {}):
				prompt_tokens_cached = usage_metadata["input_token_details"][
					"cache_read"
				]
			if "reasoning" in usage_metadata.get("output_token_details", {}):
				reasoning_tokens = usage_metadata["output_token_details"]["reasoning"]
		else:
			if response.llm_output is None:
				return None

			if "token_usage" not in response.llm_output:
				with self._lock:
					self.successful_requests += 1
				return None

			# compute tokens and cost for this request
			token_usage = response.llm_output["token_usage"]
			completion_tokens = token_usage.get("completion_tokens", 0)
			prompt_tokens = token_usage.get("prompt_tokens", 0)
			model_name = standardize_model_name(
				response.llm_output.get("model_name", "")
			)

		if model_name in MODEL_COST_PER_1K_TOKENS:
			uncached_prompt_tokens = prompt_tokens - prompt_tokens_cached
			uncached_prompt_cost = get_anthropic_token_cost_for_model(
				model_name, uncached_prompt_tokens, token_type=TokenType.PROMPT
			)
			cached_prompt_cost = get_anthropic_token_cost_for_model(
				model_name, prompt_tokens_cached, token_type=TokenType.PROMPT_CACHED
			)
			prompt_cost = uncached_prompt_cost + cached_prompt_cost
			completion_cost = get_anthropic_token_cost_for_model(
				model_name, completion_tokens, token_type=TokenType.COMPLETION
			)
		else:
			completion_cost = 0
			prompt_cost = 0

		with self._lock:
			self.total_cost += prompt_cost + completion_cost
			self.total_tokens += token_usage.get("total_tokens", 0)
			self.prompt_tokens += prompt_tokens
			self.prompt_tokens_cached += prompt_tokens_cached
			self.completion_tokens += completion_tokens
			self.reasoning_tokens += reasoning_tokens
			self.successful_requests += 1

	@override
	def __copy__(self) -> Self:
		"""Return a copy of the callback handler."""
		return self

	@override
	def __deepcopy__(self, memo: Any) -> Self:
		"""Return a deep copy of the callback handler."""
		return self
