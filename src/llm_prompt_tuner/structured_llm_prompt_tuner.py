from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from typing import Any, Literal
from pydantic import BaseModel
from enum import IntEnum
import numpy as np
from numpy.typing import NDArray
import asyncio
import itertools
import warnings
from .config import SUPPORTED_PROVIDERS


class TuneOutcome(IntEnum):
	SUCCESS = 0
	UNIDENTIFIED_FAILURE = 1
	ZERO_FIELDS_FAILURE = 2
	# [...]


class ImprovePromptOutput(BaseModel):
	"""
	Output of the prompt improvement LLM, containing the improved prompt and an optional suggestion.
	- improved_prompt: The improved prompt suggested by the LLM.
	- suggestion: Optional suggestion for further improvements.
	"""

	improved_prompt: str
	suggestion: str | None


class Scenario(BaseModel):
	"""
	A scenario consist of the following:
	- input: the input to predict an output for
	- expected_output: the expected output to compare against
	"""

	input: dict[str, Any]
	expected_output: Any

def make_scenarios(inputs: list[tuple[dict[str, Any], Any]]) -> list[Scenario]:
	"""
	Utility function to create a list of scenarios from a list of input-output pairs.

	Args:
		inputs (list[tuple[dict[str, Any], Any]]): List of tuples containing input and expected output.

	Returns:
		list[Scenario]: List of Scenario objects created from the input-output pairs.
	"""

	return [
		Scenario(input=input_data, expected_output=expected_output)
		for input_data, expected_output in inputs
	]


class TuneInput(BaseModel):
	"""
	A tune input contains all input parameters for a single model-prompt-scenario combination
	- model: the model to try
	- prompt: the prompt to pass to the model
	- scenario: the scenario to run the model-prompt-combination against
	"""

	model: Runnable[LanguageModelInput, dict[str, Any] | BaseModel]
	prompt: ChatPromptTemplate
	scenario: Scenario


class SingleTuneResult(BaseModel):
	"""
	A single tune result contains the score of a single model-prompt-scenario combination and the state of the tuning run.
	- target_score: the score of the model-prompt-combination, between 0 and 1 if "median_score", "mean_score", "best_case", "worst_case" or >= 0 if "cost_per_value"
	- state: the state of the tuning run, e.g. SUCCESS or FAILURE
	"""

	target_score: float
	state: TuneOutcome


class TuneResult(BaseModel):
	"""
	A tune result contains the overall best model-prompt-combination along with the scores. Used for both single scenario & overall tuning.
	- model: the model that was used
	- prompt: the prompt that was used
	- individual_scores: the scores for each individual tuning run, between 0 and 1
	- target_score: the overall score of the model-prompt-combination, between 0 and 1 if "median_score", "mean_score", "best_case", "worst_case" or >= 0 if "cost_per_value"
	"""

	model: Runnable[LanguageModelInput, dict[str, Any] | BaseModel]
	prompt: ChatPromptTemplate
	individual_scores: NDArray[np.float128]
	target_score: float


class BaseLLMPromptTuner:
	def __init__(self, logs_path: str | None = None):
		self.logs_path = logs_path
		self.model_providers_ = SUPPORTED_PROVIDERS


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
		target: Literal[
			"median_score", "mean_score", "best_case", "worst_case", "cost_per_value"
		] = "median_score",
		repeat: int = 1,
		prompt_improvement_llm: BaseChatModel | None = None,
		initial_prompt_improvement_prompt: str | None = None,
		improvement_retries: int | None = None,
		logs_path: str | None = None,
	):
		super().__init__(logs_path=logs_path)
		self.target = target
		self.repeat = repeat
		self.prompt_improvement_llm = prompt_improvement_llm
		self.prompt_improvement_prompt = initial_prompt_improvement_prompt
		self.improvement_retries = improvement_retries
		# TODO: add logging

	def find(
		self,
		structured_llms: list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]],
		prompts: list[str],
		temperature: int | float | list[float | float],
		scenario: list[Scenario],
	) -> TuneResult:
		"""
		Runs scenario asynchronously on structured LLMs with various prompts and expected outputs.

		Args:
			structured_llms (list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]]):
				List of structured LLMs to try.
			prompts (list[ChatPromptTemplate]): List of prompts to try.
			scenario (list[Scenario]): List of scenarios to run, each containing input and expected output.

		Returns:
			list[SingleTuneResult]: List of tuning results, each containing the model, prompt,
				individual scores, overall score, and optional improvement suggestion.
		"""
		raise NotImplementedError("This method is not implemented yet.")

	async def afind(
		self,
		structured_llms: list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]],
		prompts: list[str],
		temperature: int | float | list[int | float],
		scenarios: list[Scenario],
	) -> TuneResult:
		"""
		Runs scenario asynchronously on structured LLMs with various prompts and expected outputs.

		Args:
			structured_llms (list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]]):
				List of structured LLMs to test.
			prompts (list[str]): List of prompts to use for testing.
			scenario (list[Scenario]): List of scenario to run, each containing input and expected output.

		Returns:
			list[SingleTuneResult]: List of test results, each containing model index, prompt index,
				individual scores, overall score, and optional improvement suggestion.
		"""

		if isinstance(temperature, int):
			temperature = float(temperature)
		if isinstance(temperature, float):
			if temperature < 0.0 or temperature > 1.0:
				raise ValueError("Temperature must be between 0.0 and 1.0.")
			temperature = [temperature]
		elif isinstance(temperature, list):  # type: ignore[assignment]
			if not all(
				isinstance(t, float) and 0 <= t <= 1
				for t in temperature  # type: ignore[assignment]
			):
				raise ValueError("All temperatures must be between 0.0 and 1.0.")
		else:
			raise TypeError(
				"Temperature must be a float, int, or list of floats/ints between 0 and 1 inclusive."
			)

		combinations = itertools.product(
			scenarios, structured_llms, temperature, prompts
		)
		# TODO: add support for repetitions and temperature variations for every scenario
		async with asyncio.TaskGroup() as tg:
			tasks: list[asyncio.Task[TuneResult]] = [
				tg.create_task(
					coro=self.atune_scenario(
						scenario,
						llm.with_config(temperature=temp) if temp else llm,
						prompt,
					)
				)
				for scenario, llm, temp, prompt in combinations
			]
		results: list[TuneResult] = [task.result() for task in tasks]
		scores = np.array(
			[result.target_score for result in results],
			dtype=float,
		)
		target_score: float = self.calc_target_score_(scores)

	async def atune_scenario(
		self,
		scenario: Scenario,
		llm: Runnable[LanguageModelInput, dict[str, Any] | BaseModel],
		prompt: str,
	) -> TuneResult:
		"""
		Tunes a single scenario using the provided LLM and prompt for self.repeat times.

		Args:
			scenario (Scenario): The scenario to tune, containing input and expected output.
			llm (Runnable[LanguageModelInput, dict[str, Any] | BaseModel]):
				The LLM to use for tuning the scenario.
			prompt (str): The prompt to use for the LLM.
		"""
		async with asyncio.TaskGroup() as tg:
			tasks: list[asyncio.Task[SingleTuneResult]] = [
				tg.create_task(
					coro=self.ascore_single_scenario(scenario, llm, prompt), name=str(i)
				)
				for i in range(self.repeat)
			]
		results: list[SingleTuneResult] = [task.result() for task in tasks]
		scores = np.array(
			[result.target_score for result in results if result.state == TuneOutcome.SUCCESS],
			dtype=float,
		)
		if scores.size == 0:
			raise ValueError(
				"No successful tuning runs found. All runs failed or had zero fields."
			)

		target_score: float = self.calc_target_score_(scores)

		return TuneResult(
			model=llm,
			prompt=ChatPromptTemplate.from_messages(  # type: ignore[arg-type]
				[("system", prompt)]
			),
			individual_scores=scores,
			target_score=target_score,
		)

	async def ascore_single_scenario(
		self,
		scenario: Scenario,
		llm: Runnable[LanguageModelInput, dict[str, Any] | BaseModel],
		prompt: str,
	) -> SingleTuneResult:
		"""
		Scores a single scenario using the provided LLM and prompt.

		Args:
			scenario (Scenario): The scenario to score, containing input and expected output.
			llm (Runnable[LanguageModelInput, dict[str, Any] | BaseModel]): The LLM to use for scoring the scenario.
			prompt (str): The prompt to use for the LLM.

		Returns:
			SingleTuneResult: The result of the scoring, containing the score and cost per value score.
		"""
		# retrieve output and expected output to compare against
		formatted_prompt = ChatPromptTemplate.from_messages(  # type: ignore[arg-type]
			[("system", prompt)]
		).format(**scenario.input)
		output = llm.invoke(formatted_prompt)
		expected = scenario.expected_output

		if not isinstance(output, expected.__class__):
			raise ValueError(
				f"Output type {type(output)} does not match expected type {type(expected)}"
			)

		fields = set(
			el
			for el in dir(output)
			if not el.startswith("__") and not el.endswith("__")
		)
		expected_fields = set(
			el
			for el in dir(expected)
			if not el.startswith("__") and not el.endswith("__")
		)
		common_fields = fields & expected_fields

		correct = sum(
			getattr(output, field, None) == getattr(expected, field, None)
			for field in common_fields
		)
		total = len(common_fields)

		if total == 0:
			warnings.warn(
				"No comparable fields found in output and expected output.",
				UserWarning,
			)
			return SingleTuneResult(
				target_score=0.0,
				state=TuneOutcome.ZERO_FIELDS_FAILURE,
			)

		# TODO: implement cost calculation logic, this is a placeholder
		return SingleTuneResult(
			target_score=correct / total,
			state=TuneOutcome.SUCCESS,
		)

	def improve_prompt(self, prompt: str) -> ImprovePromptOutput:
		if not self.prompt_improvement_llm:
			raise ValueError(
				f"{self.__class__.__name__} was initialized without a prompt improvement LLM"
			)
		elif not self.prompt_improvement_prompt:
			raise ValueError(
				f"{self.__class__.__name__} was initialized without an initial prompt improvement prompt"
			)

		# self.prompt_improvement_llm.invoke()

		raise NotImplementedError("unimplemented")

	def calc_target_score_(
		self, scores: NDArray[np.float128]
	) -> float:
		"""
		Calculates the overall score from individual results.

		Args:
			results (list[SingleTuneResult]): List of individual tuning results.

		Returns:
			TuneResult: The overall tuning result.
		"""

		match self.target:
			case "median_score":
				return float(np.median(scores))
			case "mean_score":
				return scores.mean()
			case "best_case":
				return scores.max()
			case "worst_case":
				return scores.min()
			case "cost_per_value":
				return self.calc_cost_per_value_score_(scores)
			case _:
				raise ValueError(f"Unsupported target: {self.target}")

	# TODO: implement cost calculation logic, this is a placeholder
	def calc_cost_per_value_score_(self, scores: NDArray[np.float128]) -> float:
		"""
		Calculates the cost per correct value score from individual results.

		Args:
			scores (NDArray[np.float128]): Array of individual scores.

		Returns:
			float | None: The cost per correct value score, or None if not applicable.
		"""
		return scores.mean()
