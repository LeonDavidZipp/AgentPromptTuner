from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate


class UnstructuredLLMPromptTuner:
	"""
	A class for evaluating LLMs on unstructured tasks with various prompts and expected outputs.
	Also supports prompt improvement and output validation using LLMs.

	Attributes:
		repeat (int): Number of times to repeat each scenario.
		prompt_improvement_llm (BaseChatModel | None): LLM used to suggest prompt improvements.
		initial_prompt_improvement_prompt (ChatPromptTemplate | None): Prompt template for generating prompt improvements.
		improvement_retries (int | None): Number of times to retry prompt improvement if needed.
		validation_llm (BaseChatModel | None): LLM used to validate unstructured outputs.
		validation_prompt (ChatPromptTemplate | None): Prompt template for output validation.
		dotenv_path (str | None): Path to a .env file for environment variables.

	Methods:
		arun_structured: Asynchronously runs all scenarios on all model/prompt/temperature combinations.
	"""

	def __init__(
		self,
		repeat: int = 1,
		prompt_improvement_llm: BaseChatModel | None = None,
		initial_prompt_improvement_prompt: str | None = None,
		improvement_retries: int | None = None,
		validation_llm: BaseChatModel | None = None,
		validation_prompt: str | None = None,
	):
		"""
		Initial
		"""
		self.repeat: int = repeat
		self.prompt_improvement_llm: BaseChatModel | None = prompt_improvement_llm
		self.initial_prompt_improvement_prompt: ChatPromptTemplate | None = (
			ChatPromptTemplate.from_messages(  # type: ignore[arg-type]
				messages=[("system", initial_prompt_improvement_prompt)],
				template_format="f-string",
			)
			if initial_prompt_improvement_prompt
			else None
		)
		self.improvement_retries: int | None = improvement_retries
		self.validation_llm_: BaseChatModel | None = validation_llm
		self.validation_prompt_: ChatPromptTemplate | None = (
			ChatPromptTemplate.from_messages(  # type: ignore[arg-type]
				messages=[("system", validation_prompt)], template_format="f-string"
			)
			if validation_prompt
			else None
		)
