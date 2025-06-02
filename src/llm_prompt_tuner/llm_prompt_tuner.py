from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from typing import TypedDict, Any, Annotated, Literal
from collections.abc import Coroutine
from pydantic import BaseModel
import asyncio


"""
Notes:
- scenarios:
	- every scenario is like a test the model has to pass
	- across all scenarios, individual scores for llm-prompt-temperature are calculated
	- the combination with the overall highest target value gets chosen as the result
"""


class TuningScenario(TypedDict):
	input: Annotated[Any, "Input data for the scenario, can be any type"]
	expected_output: Annotated[Any, "Expected structured output for this scenario"]


class TuneInput(TypedDict):
	model: Annotated[Runnable[LanguageModelInput, dict[str, Any] | BaseModel], "The model in the list of structured_llms"]
	prompt: Annotated[ChatPromptTemplate, "The prompt"]
	scenario: Annotated[TuningScenario, "The scenario to fit the combination to."]


class SingleTuneResult(TypedDict):
	model: Annotated[Runnable[LanguageModelInput, dict[str, Any] | BaseModel], "The model in the list of structured_llms"]
	prompt: Annotated[ChatPromptTemplate, "The prompt"]
	score: Annotated[float, "Score of the model-prompt(-temperature)-combination, between 0 and 1"]
	cost_per_value: Annotated[
		float | None, "Cost per correct value for the scenario. None if not applicable"
	]


class TuneResult(TypedDict):
	model: Annotated[Runnable[LanguageModelInput, dict[str, Any] | BaseModel], "Index of the model in the list of structured_llms"]
	prompt: Annotated[ChatPromptTemplate, "Index of the prompt in the list of prompts"]
	individual_scores: Annotated[
		dict[int, float], "Scores for each individual tuning run, between 0 and 1"
	]
	median_score: Annotated[float, "Median score of all iterations"]
	mean_score: Annotated[float, "Mean score of all iterations"]
	best_score: Annotated[float, "Best score of all iterations"]
	worst_score: Annotated[float, "Worst score of all iterations"]
	best_cost_per_value: Annotated[
		float | None, "Best cost per correct value for all individual tuning runs. None if not applicable",
	]


class BaseLLMPromptTuner:
	def __init__(self):
		self.model_providers_ = [
			'openai',
			'anthropic',
			'azure_openai',
			'azure_ai',
			'google_vertexai',
			'google_genai',
			'bedrock',
			'bedrock_converse',
			'cohere',
			'fireworks',
			'together',
			'mistralai',
			'huggingface',
			'groq',
			'ollama',
			'google_anthropic_vertex',
			'deepseek',
			'ibm',
			'nvidia',
			'xai',
			'perplexity',
		]


class StructuredLLMPromptTuner(BaseLLMPromptTuner):
	"""
	A class for evaluating and tuning combinations of LLMs, prompts, and temperatures on structured tasks.
	Optionally, it can improve prompts using another LLM.

	Attributes:
		repeat (int): Number of times to repeat each scenario for statistical robustness.
		target (Literal["median_score", "mean_score", "best_case", "worst_case", "cost_per_value"]):
			The metric used to select the best model-prompt(-temperature) combination. If multiple combinations are tied,
			all are returned.
			Possible values:
				- "median_score": Selects combinations with the highest median score across all scenarios.
				- "mean_score": Selects combinations with the highest average score across all scenarios.
				- "best_case": Selects combinations with the highest single score (best-case performance).
				- "worst_case": Selects combinations with the highest minimum score (best worst-case performance).
				- "cost_per_value": Selects combinations with the lowest cost per correct value.
		prompt_improvement_llm (BaseChatModel | None): LLM used to suggest prompt improvements.
		initial_prompt_improvement_prompt (ChatPromptTemplate | None): Prompt template for generating prompt improvements.
		improvement_retries (int | None): Number of times to retry prompt improvement if needed.

	Methods:
		find: Synchronously finds the best model-prompt(-temperature) combination.
		afind: Asynchronously finds the best model-prompt(-temperature) combination.
		arun_structured: Asynchronously runs all scenarios on all model/prompt/temperature combinations.
	"""

	def __init__(
		self,
		target: Literal["median_score", "mean_score", "best_case", "worst_case", "cost_per_value"] = "median_score",
		repeat: int = 1,
		prompt_improvement_llm: BaseChatModel | None = None,
		initial_prompt_improvement_prompt: str | None = None,
		improvement_retries: int | None = None,
	):
		self.target = target
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

	def find(
		self,
		structured_llms: list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]],
		prompts: list[ChatPromptTemplate],
		temperature: int | float | list[float | float],
		scenario: list[TuningScenario],
	) -> TuneResult:
		"""
		Runs scenario asynchronously on structured LLMs with various prompts and expected outputs.

		Args:
			structured_llms (list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]]):
				List of structured LLMs to try.
			prompts (list[ChatPromptTemplate]): List of prompts to try.
			scenario (list[TuningScenario]): List of scenarios to run, each containing input and expected output.

		Returns:
			list[SingleTuneResult]: List of tuning results, each containing the model, prompt,
				individual scores, overall score, and optional improvement suggestion.
		"""
		raise NotImplementedError(
			"This method is not implemented yet."
		)

	async def afind(
		self,
		structured_llms: list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]],
		prompts: list[ChatPromptTemplate],
		temperature: int | float | list[float | float],
		scenarios: list[TuningScenario],
	) -> TuneResult:
		"""
		Runs scenario asynchronously on structured LLMs with various prompts and expected outputs.

		Args:
			structured_llms (list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]]):
				List of structured LLMs to test.
			prompts (list[ChatPromptTemplate]): List of prompts to use for testing.
			scenario (list[TuningScenario]): List of scenario to run, each containing input and expected output.

		Returns:
			list[SingleTuneResult]: List of test results, each containing model index, prompt index,
				individual scores, overall score, and optional improvement suggestion.
		"""

		if isinstance(temperature, float):
			if temperature < 0.0 or temperature > 1.0:
				raise ValueError("Temperature must be between 0.0 and 1.0.")
			temperature = [temperature]
		elif isinstance(temperature, int):
			if temperature < 0 or temperature > 1:
				raise ValueError("Temperature must be between 0 and 1.")
			temperature = [float(temperature)]
		elif isinstance(temperature, list): # type: ignore[assignment]
			if not all(isinstance(t, (float, int)) and 0 <= t <= 1 for t in temperature): # type: ignore[assignment]
				raise ValueError("All temperatures must be between 0.0 and 1.0.")

		tasks: list[Coroutine[Any, Any, tuple[int, int, dict[int, float], float]]] = []
		# TODO: add support for repetitions and temperature variations
		# for every scenario
		result: TuneResult
		results: dict[int, list[SingleTuneResult]] = {}
		# for every scenario
		for scenario_id, scenario in enumerate(scenarios):
			results[scenario_id] = []
			# for every llm
			for llm in structured_llms:
				# with every temperature
				for temp in temperature:
					new_llm = llm.with_config(
						temperature=temp,
					) if temp else llm
					# combine with all prompts
					for prompt in prompts:
						# repeat every prompt test 3 times
						for _ in range(self.repeat):
							# test the prompt-llm-temp-combination
							tasks.append(
								self.arun_scenario_structured_(
									new_llm, temp, temp, temp_llm, prompt, scenario
								)
							)
							# TODO: tests
							# Assign res
							"""
							res = {
								"model": new_llm,
								"prompt": prompt,
							}
							"""
							if self.improvement_retries is not None:
								for _ in range(self.improvement_retries):
									pass
							# then 
			# all_results = await asyncio.gather(*tasks)
			# for model_idx, prompt_idx, temp, individual_scores, overall_score in all_results:
			# 	results.append(
			# 		SingleTuneResult(
			# 			model_idx=model_idx,
			# 			prompt_idx=prompt_idx,
			# 			temperature=temp,
			# 			individual_scores=individual_scores,
			# 			overall_score=overall_score,
			# 		)
			# 	)
			return result
		
	def improve_prompt(prompt: ChatPromptTemplate) -> ChatPromptTemplate:
		message = prompt.messages[0][1] if prompt.messages else ""

	
	def calc_score_(self):
		pass

	def calc_costs_per_value_(self):
		pass

	@staticmethod
	async def arun_scenario_structured_(
		model: Runnable[LanguageModelInput, dict[str, Any] | BaseModel],
		temperature: float | int,
		llm: Runnable[LanguageModelInput, dict[str, Any] | BaseModel],
		prompt: ChatPromptTemplate,
		scenario: list[TuningScenario],
	) -> tuple[int, int, dict[int, float], float]:
		individual_scores: dict[int, float] = {}
		for scenario_idx, scenario in enumerate(scenario):
			input_data = prompt.format(**scenario["input"])
			output = await llm.ainvoke(input_data)
			# TODO: run statistics
			score = float(output == scenario["expected_output"])
			individual_scores[scenario_idx] = score
		overall_score = sum(individual_scores.values()) / len(individual_scores)
		return (model_idx, prompt_idx, temperature, individual_scores, overall_score)


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
