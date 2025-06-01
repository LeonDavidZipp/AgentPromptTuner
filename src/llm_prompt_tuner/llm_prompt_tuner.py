from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from typing import TypedDict, Any, Annotated, NotRequired
from collections.abc import Coroutine
from pydantic import BaseModel
import asyncio


class Test(TypedDict):
	input: Annotated[Any, "Input data for the test, can be any type"]
	expected_output: Annotated[Any, "Expected output for the test, can be any type"]


class Improvement(TypedDict):
	"""
	A class to represent an improvement suggestion for a prompt.
	Attributes:
		prompt (ChatPromptTemplate): The improved prompt template.
		scores (dict[int, float]): Scores for each individual test, between 0 and 1.
		overall_score (float): Overall score of the test, between 0 and 1.
	"""

	prompt: Annotated[str, "The improved prompt after testing"]
	individual_scores: Annotated[
			dict[int, float],
			"Improved individual scores for each test, between 0 and 1",
		]
	overall_score: Annotated[float, "Improved overall score of the test, between 0 and 1"]
	retries: Annotated[int, "Number of retries for the test"]
	best_temperature: Annotated[float, "Best temperature for the model based on test results"]


class TestResult(TypedDict):
	model_idx: Annotated[int, "Index of the model in the list of structured_llms"]
	prompt_idx: Annotated[int, "Index of the prompt in the list of prompts"]
	individual_scores: Annotated[
		dict[int, float], "Scores for each individual test, between 0 and 1"
	]
	overall_score: Annotated[float, "Overall score of the test, between 0 and 1"]
	retries: Annotated[int, "Number of retries for the test"]
	best_temperature: Annotated[float, "Best temperature for the model based on test results"]
	improvement: NotRequired[
		Annotated[
			Improvement,
			"Optional improvement suggestion for the prompt based on test results",
		]
	]


class LLMPromptTester:
	"""
	A class to test structured LLMs with various prompts and expected outputs.

	Attributes:
		repeat_ (int): Number of times to repeat the tests.
		prompt_improvement_llm_ (BaseChatModel | None): LLM used for prompt improvement.
		prompt_improvement_prompt_ (ChatPromptTemplate | None): Prompt template for prompt improvement.
		improvement_retries_ (int | None): Number of retries for prompt improvement.
		validation_llm_ (BaseChatModel | None): LLM used for validation.
		validation_prompt_ (ChatPromptTemplate | None): Prompt template for validation.
		dotenv_path_ (str | None): Path to a .env file for environment variables.

	Methods:
		arun_structured: Asynchronously runs tests on structured LLMs with different temperatures, prompts and expected outputs.
	"""

	def __init__(
		self,
		repeat: int = 1,
		prompt_improvement_llm: BaseChatModel | None = None,
		prompt_improvement_prompt: str | None = None,
		improvement_retries: int | None = None,
		validation_llm: BaseChatModel | None = None,
		validation_prompt: str | None = None,
	):
		"""
		Initial
		"""
		self.repeat_: int = repeat
		self.prompt_improvement_llm_: BaseChatModel | None = prompt_improvement_llm
		self.prompt_improvement_prompt_: ChatPromptTemplate | None = (
			ChatPromptTemplate.from_messages(  # type: ignore[arg-type]
				messages=[("system", prompt_improvement_prompt)],
				template_format="f-string",
			)
			if prompt_improvement_prompt
			else None
		)
		self.improvement_retries_: int | None = improvement_retries
		self.validation_llm_: BaseChatModel | None = validation_llm
		self.validation_prompt_: ChatPromptTemplate | None = (
			ChatPromptTemplate.from_messages(  # type: ignore[arg-type]
				messages=[("system", validation_prompt)], template_format="f-string"
			)
			if validation_prompt
			else None
		)

	async def arun_structured(
		self,
		structured_llms: list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]],
		prompts: list[ChatPromptTemplate],
		tests: list[Test],
		temperature: float | list[float] = 0.0,
	) -> list[TestResult]:
		"""
		Runs tests asynchronously on structured LLMs with various prompts and expected outputs.

		Args:
			structured_llms (list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]]):
				List of structured LLMs to test.
			prompts (list[ChatPromptTemplate]): List of prompts to use for testing.
			tests (list[Test]): List of tests to run, each containing input and expected output.

		Returns:
			list[TestResult]: List of test results, each containing model index, prompt index,
				individual scores, overall score, and optional improvement suggestion.
		"""

		if isinstance(temperature, float):
			temperature = [temperature]

		results: list[TestResult] = []
		tasks: list[Coroutine[Any, Any, tuple[int, int, dict[int, float], float]]] = []
		# TODO: add support for repetitions and temperature variations
		for _ in range(self.repeat_):
			for temp in temperature:
				for model_idx, llm in enumerate(structured_llms):
					for prompt_idx, prompt in enumerate(prompts):
						tasks.append(
							self.arun_tests_structured_(
								model_idx, prompt_idx, llm, prompt, tests
							)
						)
				all_results = await asyncio.gather(*tasks)
				for model_idx, prompt_idx, individual_scores, overall_score in all_results:
					results.append(
						TestResult(
							model_idx=model_idx,
							prompt_idx=prompt_idx,
							individual_scores=individual_scores,
							overall_score=overall_score,
						)
					)
		return results

	@staticmethod
	async def arun_tests_structured_(
		model_idx: int,
		prompt_idx: int,
		llm: Runnable[LanguageModelInput, dict[str, Any] | BaseModel],
		prompt: ChatPromptTemplate,
		tests: list[Test],
	) -> tuple[int, int, dict[int, float], float]:
		individual_scores: dict[int, float] = {}
		for test_idx, test in enumerate(tests):
			input_data = prompt.format(**test["input"])
			output = await llm.ainvoke(input_data)
			score = float(output == test["expected_output"])
			individual_scores[test_idx] = score
		overall_score = sum(individual_scores.values()) / len(individual_scores)
		return (model_idx, prompt_idx, individual_scores, overall_score)
