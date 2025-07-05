from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from typing import Any
from pydantic import BaseModel
import unittest
import warnings
import asyncio
import numpy as np
from unittest.mock import patch, Mock, AsyncMock
from src.llm_prompt_tuner import (
	StructuredLLMPromptTuner,
	Scenario,
	TuneOutcome,
	TuneResult,
	IntermediateTuneResult,
	SingleTuneResult,
)


class MockLLM(Runnable[LanguageModelInput, dict[str, Any] | BaseModel]):
	"""Mock LLM that always returns the expected output from the scenario"""

	def __init__(self, expected_output: BaseModel):
		self.expected_output = expected_output

	def invoke(self, input: str, config=None) -> BaseModel:  # type: ignore[override]
		# Always return the expected output (perfect score)
		return self.expected_output

	async def ainvoke(self, input: str, config=None) -> BaseModel:  # type: ignore[override]
		pass

	def with_config(self, **kwargs) -> "MockLLM":  # type: ignore
		new_mock = MockLLM(self.expected_output)
		return new_mock


class MultiCallMockLLM(Runnable[LanguageModelInput, dict[str, Any] | BaseModel]):
	def __init__(self, expected_output: list[BaseModel], repeat: int = 1):
		self.expected_output: list[BaseModel] = expected_output
		self.count: int = 0
		self.repeat: int = repeat
		self.max: int = repeat * len(expected_output)

	def invoke(self, input: str, config=None) -> BaseModel:  # type: ignore[override]
		# Always return the expected output (perfect score)
		if self.count >= self.max:
			raise ValueError(f"Cannot call this more than {self.max} times")
		res = self.expected_output[int(self.count / self.repeat)]
		self.count += 1
		return res

	async def ainvoke(self, input: str, config=None) -> BaseModel:  # type: ignore[override]
		pass

	def with_config(self, **kwargs) -> "MultiCallMockLLM":  # type: ignore
		new_mock = MultiCallMockLLM(self.expected_output, self.repeat)
		return new_mock


class FailingMockLLM(Runnable[LanguageModelInput, dict[str, Any] | BaseModel]):
	def invoke(self, input: str, config=None) -> BaseModel:  # type: ignore[override]
		raise RuntimeError("LLM failed")

	async def ainvoke(self, input, config=None) -> BaseModel:  # type: ignore[override]
		pass

	def with_config(self, **kwargs):  # type: ignore[override]
		return FailingMockLLM()


class PersonOutput(BaseModel):
	name: str
	age: int
	weight: float


class AlternatePersonOutput(BaseModel):
	name: str
	age: int
	weight: float


class ComplexOutput(BaseModel):
	tags: list[str]
	score: float
	metadata: dict[str, Any]


class DifferentOutput(BaseModel):
	title: str
	description: str


class NullableOutput(BaseModel):
	name: str | None
	age: int | None
	weight: float | None


class EmptyOutput(BaseModel):
	pass


class AScoreSingleScenarioTest(unittest.IsolatedAsyncioTestCase):
	def setUp(self):
		self.tuner = StructuredLLMPromptTuner(repeat=1)

		self.scenario = Scenario(
			input={"input": "Hannah is 30 years old and weighs 60kg."},
			expected_output=PersonOutput(name="Hannah", age=30, weight=60.0),
		)

	async def test_perfect_score(self):
		"""Test when output perfectly matches expected"""
		result = await self.tuner.ascore_single_scenario(
			scenario=self.scenario,
			llm=MockLLM(PersonOutput(name="Hannah", age=30, weight=60.0)),
			prompt="Extract person data: {input}",
		)

		self.assertEqual(result.target_score, 1.0)
		self.assertEqual(result.state, TuneOutcome.SUCCESS)

	async def test_partial_score(self):
		"""Test when output partially matches expected"""
		result = await self.tuner.ascore_single_scenario(
			scenario=self.scenario,
			llm=MockLLM(PersonOutput(name="Hannah", age=25, weight=60.0)),  # age wrong
			prompt="Extract person data: {input}",
		)

		self.assertAlmostEqual(result.target_score, 2 / 3, delta=0.01)
		self.assertEqual(result.state, TuneOutcome.SUCCESS)

	async def test_zero_score(self):
		"""Test when output doesn't match expected at all"""
		result = await self.tuner.ascore_single_scenario(
			scenario=self.scenario,
			llm=MockLLM(PersonOutput(name="John", age=25, weight=70.0)),  # all wrong
			prompt="Extract person data: {input}",
		)

		self.assertEqual(result.target_score, 0.0)
		self.assertEqual(result.state, TuneOutcome.SUCCESS)

	async def test_output_not_basemodel_raises_error(self):
		"""Test that non-BaseModel output raises ValueError"""
		with self.assertRaises(ValueError) as context:
			await self.tuner.ascore_single_scenario(
				scenario=self.scenario,
				llm=MockLLM({"name": "Hannah", "age": 30}),  # type: ignore[call-arg]
				prompt="Extract person data: {input}",
			)

		self.assertIn(
			"Output and expected output must be instances of pydantic BaseModel",
			str(context.exception),
		)

	async def test_expected_not_basemodel_raises_error(self):
		"""Test that non-BaseModel expected output raises ValueError"""
		scenario_with_dict = Scenario(
			input={"input": "test"},
			expected_output={"name": "Hannah"},  # dict instead of BaseModel
		)

		with self.assertRaises(ValueError) as context:
			await self.tuner.ascore_single_scenario(
				scenario=scenario_with_dict,
				llm=MockLLM(PersonOutput(name="Hannah", age=30, weight=60.0)),
				prompt="Extract person data: {input}",
			)

		self.assertIn(
			"Output and expected output must be instances of pydantic BaseModel",
			str(context.exception),
		)

	async def test_no_common_fields_warning(self):
		"""Test warning when no common fields exist"""
		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("always")

			result = await self.tuner.ascore_single_scenario(
				scenario=self.scenario,
				llm=MockLLM(
					DifferentOutput(title="Test", description="Desc")
				),  # Different fields
				prompt="Extract person data: {input}",
			)

			# Check warning was raised
			self.assertEqual(len(w), 1)
			self.assertIn("No comparable fields found", str(w[0].message))

			# Check result
			self.assertEqual(result.target_score, 0.0)
			self.assertEqual(result.state, TuneOutcome.ZERO_FIELDS_FAILURE)

	async def test_empty_models_no_common_fields(self):
		"""Test with empty models that have no fields"""
		empty_scenario = Scenario(
			input={"input": "test"}, expected_output=EmptyOutput()
		)

		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("always")

			result = await self.tuner.ascore_single_scenario(
				scenario=empty_scenario,
				llm=MockLLM(EmptyOutput()),
				prompt="Extract data: {input}",
			)

			self.assertEqual(len(w), 1)
			self.assertEqual(result.target_score, 0.0)
			self.assertEqual(result.state, TuneOutcome.ZERO_FIELDS_FAILURE)

	async def test_complex_field_types(self):
		"""Test with complex field types (nested objects, lists, etc.)"""

		expected = ComplexOutput(
			tags=["person", "data"],
			score=0.95,
			metadata={"source": "text", "confidence": 0.9},
		)

		complex_scenario = Scenario(input={"input": "test"}, expected_output=expected)

		# Exact match
		result = await self.tuner.ascore_single_scenario(
			scenario=complex_scenario,
			llm=MockLLM(
				ComplexOutput(
					tags=["person", "data"],
					score=0.95,
					metadata={"source": "text", "confidence": 0.9},
				)
			),
			prompt="Extract: {input}",
		)

		self.assertEqual(result.target_score, 1.0)

	async def test_none_values(self):
		"""Test handling of None values in fields"""

		expected = NullableOutput(name="Hannah", age=None, weight=60.0)
		nullable_scenario = Scenario(input={"input": "test"}, expected_output=expected)

		result = await self.tuner.ascore_single_scenario(
			scenario=nullable_scenario,
			llm=MockLLM(
				NullableOutput(name="Hannah", age=None, weight=70.0)
			),  # weight differs
			prompt="Extract: {input}",
		)

		self.assertAlmostEqual(result.target_score, 2 / 3, delta=0.01)

	async def test_llm_exception_propagated(self):
		"""Test that LLM exceptions are properly propagated"""

		with self.assertRaises(RuntimeError) as context:
			await self.tuner.ascore_single_scenario(
				scenario=self.scenario,
				llm=FailingMockLLM(),
				prompt="Extract: {input}",
			)

		self.assertIn("LLM failed", str(context.exception))

	async def test_scenario_input_missing_key(self):
		"""Test behavior when prompt references missing input key"""
		# This should raise a KeyError during prompt formatting
		with self.assertRaises(KeyError):
			await self.tuner.ascore_single_scenario(
				scenario=self.scenario,
				llm=MockLLM(PersonOutput(name="Hannah", age=30, weight=60.0)),
				prompt="Extract from: {missing_key}",  # Key doesn't exist in scenario.input
			)

	async def test_different_model_same_fields(self):
		"""Test with different model classes but same field names"""

		# Same field names and values but different class
		result = await self.tuner.ascore_single_scenario(
			scenario=self.scenario,
			llm=MockLLM(AlternatePersonOutput(name="Hannah", age=30, weight=60.0)),
			prompt="Extract: {input}",
		)

		self.assertEqual(result.target_score, 1.0)
		self.assertEqual(result.state, TuneOutcome.SUCCESS)

	async def test_floating_point_comparison(self):
		"""Test that floating point values are compared correctly"""
		result = await self.tuner.ascore_single_scenario(
			scenario=self.scenario,
			llm=MockLLM(
				PersonOutput(name="Hannah", age=30, weight=60.000001)
			),  # Tiny difference
			prompt="Extract: {input}",
		)

		# Should be exact comparison, not approximate
		self.assertAlmostEqual(result.target_score, 2 / 3, delta=0.01)


class ATuneScenarioTest(unittest.IsolatedAsyncioTestCase):
	def setUp(self):
		"""Set up tuner instances with different repeat values"""
		self.tuner1 = StructuredLLMPromptTuner(repeat=1)
		self.tuner3 = StructuredLLMPromptTuner(repeat=3)
		self.tuner5 = StructuredLLMPromptTuner(repeat=5)

		self.scenario = Scenario(
			input={"input": "test data"},
			expected_output=PersonOutput(name="Hannah", age=30, weight=60.0),
		)

		self.mock_llm = MockLLM(PersonOutput(name="Hannah", age=30, weight=60.0))

	async def test_single_repeat_calls_ascore_once(self):
		"""Test that repeat=1 calls ascore_single_scenario exactly once"""
		with patch.object(
			self.tuner1, "ascore_single_scenario", new_callable=AsyncMock
		) as mock_ascore:
			mock_ascore.return_value = SingleTuneResult(
				target_score=1.0, state=TuneOutcome.SUCCESS
			)

			result = await self.tuner1.atune_scenario(
				self.scenario, self.mock_llm, "test prompt"
			)

			# Verify ascore was called exactly once
			self.assertEqual(mock_ascore.call_count, 1)
			mock_ascore.assert_called_with(self.scenario, self.mock_llm, "test prompt")

			# Verify result
			self.assertEqual(len(result.individual_scores), 1)
			self.assertEqual(result.individual_scores[0], 1.0)

	async def test_multiple_repeats_calls_ascore_multiple_times(self):
		"""Test that repeat=3 calls ascore_single_scenario exactly 3 times"""
		with patch.object(
			self.tuner3, "ascore_single_scenario", new_callable=AsyncMock
		) as mock_ascore:
			mock_ascore.return_value = SingleTuneResult(
				target_score=0.8, state=TuneOutcome.SUCCESS
			)

			result = await self.tuner3.atune_scenario(
				self.scenario, self.mock_llm, "test prompt"
			)

			# Verify ascore was called exactly 3 times
			self.assertEqual(mock_ascore.call_count, 3)

			# Verify all calls had same arguments
			for call in mock_ascore.call_args_list:
				self.assertEqual(call[0], (self.scenario, self.mock_llm, "test prompt"))

			# Verify result aggregation
			self.assertEqual(len(result.individual_scores), 3)
			np.testing.assert_array_equal(result.individual_scores, [0.8, 0.8, 0.8])

	async def test_concurrent_execution_timing(self):
		"""Test that multiple repeats run concurrently, not sequentially"""
		call_times: list[float] = []

		async def slow_ascore_mock(
			scenario: Scenario,
			llm: Runnable[LanguageModelInput, dict[str, Any] | BaseModel],
			prompt: str,
		):
			import time

			call_times.append(time.time())
			await asyncio.sleep(0.1)  # Simulate slow operation
			return SingleTuneResult(target_score=1.0, state=TuneOutcome.SUCCESS)

		with patch.object(
			self.tuner3, "ascore_single_scenario", side_effect=slow_ascore_mock
		):
			import time

			start_time = time.time()

			await self.tuner3.atune_scenario(
				self.scenario, self.mock_llm, "test prompt"
			)

			elapsed_time = time.time() - start_time

			# Should take ~0.1s (concurrent) not ~0.3s (sequential)
			self.assertLess(elapsed_time, 0.2)

			# All calls should start around the same time
			time_diffs = [t - call_times[0] for t in call_times]
			self.assertLess(max(time_diffs), 0.05)  # All start within 50ms

	async def test_filters_only_successful_results(self):
		"""Test that only SUCCESS state results are included in scores"""
		results = [
			SingleTuneResult(target_score=1.0, state=TuneOutcome.SUCCESS),
			SingleTuneResult(target_score=0.0, state=TuneOutcome.ZERO_FIELDS_FAILURE),
			SingleTuneResult(target_score=0.8, state=TuneOutcome.SUCCESS),
		]

		with patch.object(
			self.tuner3, "ascore_single_scenario", new_callable=AsyncMock
		) as mock_ascore:
			mock_ascore.side_effect = results

			result = await self.tuner3.atune_scenario(
				self.scenario, self.mock_llm, "test prompt"
			)

			# Should only include the 2 successful results
			self.assertEqual(len(result.individual_scores), 2)
			np.testing.assert_array_equal(result.individual_scores, [1.0, 0.8])

	async def test_raises_error_when_no_successful_runs(self):
		"""Test ValueError is raised when all runs fail or have zero fields"""
		failed_results = [
			SingleTuneResult(target_score=0.0, state=TuneOutcome.ZERO_FIELDS_FAILURE),
			SingleTuneResult(target_score=0.0, state=TuneOutcome.ZERO_FIELDS_FAILURE),
			SingleTuneResult(target_score=0.0, state=TuneOutcome.ZERO_FIELDS_FAILURE),
		]

		with patch.object(
			self.tuner3, "ascore_single_scenario", new_callable=AsyncMock
		) as mock_ascore:
			mock_ascore.side_effect = failed_results

			with self.assertRaises(ValueError) as context:
				await self.tuner3.atune_scenario(
					self.scenario, self.mock_llm, "test prompt"
				)

			self.assertIn("No successful tuning runs found", str(context.exception))

	async def test_task_group_exception_handling(self):
		"""Test that exceptions from ascore_single_scenario propagate correctly"""
		with patch.object(
			self.tuner3, "ascore_single_scenario", new_callable=AsyncMock
		) as mock_ascore:
			mock_ascore.side_effect = RuntimeError("Mocked error")

			with self.assertRaises(ExceptionGroup) as context:
				await self.tuner3.atune_scenario(
					self.scenario, self.mock_llm, "test prompt"
				)

			# Verify it's an ExceptionGroup with 3 RuntimeErrors
			exception_group = context.exception
			self.assertEqual(len(exception_group.exceptions), 3)

			# Check that all sub-exceptions are RuntimeErrors with correct message
			for exc in exception_group.exceptions:
				self.assertIsInstance(exc, RuntimeError)
				self.assertIn("Mocked error", str(exc))

	async def test_numpy_array_creation(self):
		"""Test that individual_scores is properly created as numpy array"""
		scores = [0.1, 0.5, 0.9]
		results = [
			SingleTuneResult(target_score=score, state=TuneOutcome.SUCCESS)
			for score in scores
		]

		with patch.object(
			self.tuner3, "ascore_single_scenario", new_callable=AsyncMock
		) as mock_ascore:
			mock_ascore.side_effect = results

			result = await self.tuner3.atune_scenario(
				self.scenario, self.mock_llm, "test prompt"
			)

			# Verify numpy array properties
			self.assertIsInstance(result.individual_scores, np.ndarray)
			self.assertEqual(result.individual_scores.dtype, np.float64)
			np.testing.assert_array_equal(result.individual_scores, scores)

	async def test_result_structure_and_references(self):
		"""Test that IntermediateTuneResult contains correct references"""
		with patch.object(
			self.tuner1, "ascore_single_scenario", new_callable=AsyncMock
		) as mock_ascore:
			mock_ascore.return_value = SingleTuneResult(
				target_score=0.7, state=TuneOutcome.SUCCESS
			)

			prompt = "custom test prompt"
			result = await self.tuner1.atune_scenario(
				self.scenario, self.mock_llm, prompt
			)

			# Verify structure
			self.assertIsInstance(result, IntermediateTuneResult)
			self.assertIs(result.model, self.mock_llm)
			self.assertEqual(result.prompt, prompt)
			self.assertIsInstance(result.individual_scores, np.ndarray)

	async def test_task_names_for_debugging(self):
		"""Test that tasks are created with proper names for debugging"""
		task_names = []

		original_create_task = asyncio.TaskGroup.create_task

		def mock_create_task(self, coro, name=None):  # type: ignore[no-untyped-def]
			task_names.append(name)  # type: ignore[no-untyped-def]
			return original_create_task(self, coro, name=name)  # type: ignore[no-untyped-def]

		with patch.object(
			self.tuner3, "ascore_single_scenario", new_callable=AsyncMock
		) as mock_ascore:
			mock_ascore.return_value = SingleTuneResult(
				target_score=1.0, state=TuneOutcome.SUCCESS
			)

			with patch.object(asyncio.TaskGroup, "create_task", mock_create_task):  # type: ignore[no-untyped-call]
				await self.tuner3.atune_scenario(
					self.scenario, self.mock_llm, "test prompt"
				)

			# Verify task names are sequential strings
			self.assertEqual(task_names, ["0", "1", "2"])

	async def test_large_number_of_repeats(self):
		"""Test with large repeat count"""
		large_tuner = StructuredLLMPromptTuner(repeat=100)

		with patch.object(
			large_tuner, "ascore_single_scenario", new_callable=AsyncMock
		) as mock_ascore:
			mock_ascore.return_value = SingleTuneResult(
				target_score=0.5, state=TuneOutcome.SUCCESS
			)

			result = await large_tuner.atune_scenario(
				self.scenario, self.mock_llm, "test prompt"
			)

			# Verify all 100 calls were made
			self.assertEqual(mock_ascore.call_count, 100)
			self.assertEqual(len(result.individual_scores), 100)
			np.testing.assert_array_equal(result.individual_scores, np.full(100, 0.5))

	async def test_mixed_success_failure_filtering(self):
		"""Test complex scenario with various result states"""
		mixed_results = [
			SingleTuneResult(target_score=1.0, state=TuneOutcome.SUCCESS),
			SingleTuneResult(target_score=0.0, state=TuneOutcome.ZERO_FIELDS_FAILURE),
			SingleTuneResult(target_score=0.3, state=TuneOutcome.SUCCESS),
			SingleTuneResult(target_score=0.0, state=TuneOutcome.ZERO_FIELDS_FAILURE),
			SingleTuneResult(target_score=0.9, state=TuneOutcome.SUCCESS),
		]

		tuner5 = StructuredLLMPromptTuner(repeat=5)

		with patch.object(
			tuner5, "ascore_single_scenario", new_callable=AsyncMock
		) as mock_ascore:
			mock_ascore.side_effect = mixed_results

			result = await tuner5.atune_scenario(
				self.scenario, self.mock_llm, "test prompt"
			)

			# Should only include the 3 successful results
			self.assertEqual(len(result.individual_scores), 3)
			np.testing.assert_array_equal(result.individual_scores, [1.0, 0.3, 0.9])


class AFindTest(unittest.IsolatedAsyncioTestCase):
	def setUp(self):
		"""Set up tuner instances and test data"""
		self.tuner = StructuredLLMPromptTuner(repeat=1)
		self.tuner_multi = StructuredLLMPromptTuner(repeat=3)

		# Test scenarios
		self.scenario1 = Scenario(
			input={"input": "Hannah is 30 years old and weighs 60kg."},
			expected_output=PersonOutput(name="Hannah", age=30, weight=60.0),
		)
		self.scenario2 = Scenario(
			input={"input": "John is 25 years old and weighs 70kg."},
			expected_output=PersonOutput(name="John", age=25, weight=70.0),
		)

		# Test LLMs
		self.perfect_llm: Runnable[LanguageModelInput, dict[str, Any] | BaseModel] = (
			MockLLM(PersonOutput(name="Hannah", age=30, weight=60.0))
		)
		self.partial_llm: Runnable[LanguageModelInput, dict[str, Any] | BaseModel] = (
			MockLLM(PersonOutput(name="Hannah", age=25, weight=60.0))
		)  # age wrong
		self.zero_llm: Runnable[LanguageModelInput, dict[str, Any] | BaseModel] = (
			MockLLM(PersonOutput(name="Wrong", age=99, weight=99.0))
		)  # all wrong

	# Temperature validation tests
	async def test_temperature_int_conversion(self):
		"""Test that integer temperature is converted to float"""
		result = await self.tuner.afind(
			structured_llms=[self.perfect_llm],
			prompts=["Extract: {input}"],
			scenarios=[self.scenario1],
			temperature=0,  # int
		)

		self.assertIsInstance(result, TuneResult)
		self.assertEqual(result.temperature, 0.0)

	async def test_temperature_float_validation(self):
		"""Test that valid float temperature works"""
		result = await self.tuner.afind(
			structured_llms=[self.perfect_llm],
			prompts=["Extract: {input}"],
			scenarios=[self.scenario1],
			temperature=0.5,
		)

		self.assertEqual(result.temperature, 0.5)

	async def test_temperature_list_validation(self):
		"""Test that list of temperatures works"""
		result = await self.tuner.afind(
			structured_llms=[self.perfect_llm],
			prompts=["Extract: {input}"],
			scenarios=[self.scenario1],
			temperature=[0.0, 0.3, 0.7],
		)

		# Should return best result from the temperature list
		self.assertIn(result.temperature, [0.0, 0.3, 0.7])

	async def test_temperature_out_of_range_raises_error(self):
		"""Test that temperature outside 0-1 range raises ValueError"""
		with self.assertRaises(ValueError) as context:
			await self.tuner.afind(
				structured_llms=[self.perfect_llm],
				prompts=["Extract: {input}"],
				scenarios=[self.scenario1],
				temperature=1.5,  # > 1.0
			)

		self.assertIn("Temperature must be between 0.0 and 1.0", str(context.exception))

	async def test_temperature_negative_raises_error(self):
		"""Test that negative temperature raises ValueError"""
		with self.assertRaises(ValueError) as context:
			await self.tuner.afind(
				structured_llms=[self.perfect_llm],
				prompts=["Extract: {input}"],
				scenarios=[self.scenario1],
				temperature=-0.1,
			)

		self.assertIn("Temperature must be between 0.0 and 1.0", str(context.exception))

	async def test_temperature_list_invalid_values_raises_error(self):
		"""Test that list with invalid temperature values raises ValueError"""
		with self.assertRaises(ValueError) as context:
			await self.tuner.afind(
				structured_llms=[self.perfect_llm],
				prompts=["Extract: {input}"],
				scenarios=[self.scenario1],
				temperature=[0.0, 0.5, 1.5],  # 1.5 > 1.0
			)

		self.assertIn(
			"All temperatures must be between 0.0 and 1.0", str(context.exception)
		)

	async def test_temperature_wrong_type_raises_error(self):
		"""Test that invalid temperature type raises TypeError"""
		with self.assertRaises(TypeError) as context:
			await self.tuner.afind(
				structured_llms=[self.perfect_llm],
				prompts=["Extract: {input}"],
				scenarios=[self.scenario1],
				temperature="invalid",  # type: ignore
			)

		self.assertIn(
			"Temperature must be a float, int, or list", str(context.exception)
		)

	# Basic functionality tests
	async def test_single_combination_perfect_score(self):
		"""Test single model, prompt, scenario combination with perfect score"""
		result = await self.tuner.afind(
			structured_llms=[self.perfect_llm],
			prompts=["Extract person data: {input}"],
			scenarios=[self.scenario1],
			temperature=0.0,
		)

		self.assertIsInstance(result, TuneResult)
		self.assertEqual(result.target_score, 1.0)
		self.assertEqual(result.temperature, 0.0)
		self.assertEqual(result.prompt, "Extract person data: {input}")
		self.assertIs(result.model, self.perfect_llm)

	async def test_multiple_scenarios_aggregation(self):
		"""Test that multiple scenarios are properly aggregated"""
		# Both scenarios should get perfect scores
		perfect_llm_multi = MockLLM(PersonOutput(name="Perfect", age=99, weight=99.0))

		result = await self.tuner.afind(
			structured_llms=[perfect_llm_multi],
			prompts=["Extract: {input}"],
			scenarios=[self.scenario1, self.scenario2],
			temperature=0.0,
		)

		# Should have scores from both scenarios
		self.assertEqual(len(result.individual_scores), 2)

	async def test_multiple_models_comparison(self):
		"""Test that multiple models are compared and best is returned"""
		models = [self.perfect_llm, self.partial_llm, self.zero_llm]

		result = await self.tuner.afind(
			structured_llms=models,
			prompts=["Extract: {input}"],
			scenarios=[self.scenario1],
			temperature=0.0,
		)

		# Should return the perfect LLM as best
		self.assertEqual(result.target_score, 1.0)
		self.assertIs(result.model, self.perfect_llm)

	async def test_multiple_prompts_comparison(self):
		"""Test that multiple prompts are compared"""
		prompts = [
			"Extract person data: {input}",
			"Find person info: {input}",
			"Parse person details: {input}",
		]

		result = await self.tuner.afind(
			structured_llms=[self.perfect_llm],
			prompts=prompts,
			scenarios=[self.scenario1],
			temperature=0.0,
		)

		# Should return one of the prompts (all should perform equally with perfect LLM)
		self.assertIn(result.prompt, prompts)
		self.assertEqual(result.target_score, 1.0)

	async def test_multiple_temperatures_comparison(self):
		"""Test that multiple temperatures are compared"""
		temperatures = [0.0, 0.3, 0.7]

		result = await self.tuner.afind(
			structured_llms=[self.perfect_llm],
			prompts=["Extract: {input}"],
			scenarios=[self.scenario1],
			temperature=temperatures,
		)

		# Should return one of the temperatures
		self.assertIn(result.temperature, temperatures)

	async def test_combination_explosion_handling(self):
		"""Test handling of many combinations (2 models × 3 prompts × 2 temps × 2 scenarios = 24 combinations)"""
		models = [self.perfect_llm, self.partial_llm]
		prompts = ["Prompt 1: {input}", "Prompt 2: {input}", "Prompt 3: {input}"]
		temperatures = [0.0, 0.5]
		scenarios = [self.scenario1, self.scenario2]

		result = await self.tuner.afind(
			structured_llms=models,
			prompts=prompts,
			scenarios=scenarios,
			temperature=temperatures,
		)

		# Should complete successfully and return best result
		self.assertIsInstance(result, TuneResult)
		self.assertIn(result.model, models)
		self.assertIn(result.prompt, prompts)
		self.assertIn(result.temperature, temperatures)

	# Error handling tests
	async def test_all_combinations_fail_raises_error(self):
		"""Test that ValueError is raised when all combinations fail"""
		failing_llm = FailingMockLLM()

		with self.assertRaises(ExceptionGroup):  # TaskGroup wraps exceptions
			await self.tuner.afind(
				structured_llms=[failing_llm],
				prompts=["Extract: {input}"],
				scenarios=[self.scenario1],
				temperature=0.0,
			)

	async def test_no_successful_runs_raises_error(self):
		"""Test error when all runs have zero fields (no common fields)"""
		different_output_llm = MockLLM(
			DifferentOutput(title="Test", description="Desc")
		)

		with self.assertRaises(ExceptionGroup) as context:
			await self.tuner.afind(
				structured_llms=[different_output_llm],
				prompts=["Extract: {input}"],
				scenarios=[self.scenario1],
				temperature=0.0,
			)

		# Verify the ExceptionGroup contains the expected ValueError
		exception_group = context.exception
		self.assertEqual(len(exception_group.exceptions), 1)

		inner_exception = exception_group.exceptions[0]
		self.assertIsInstance(inner_exception, ValueError)
		self.assertIn("No successful tuning runs found", str(inner_exception))

	# Result structure tests
	async def test_result_structure_completeness(self):
		"""Test that TuneResult contains all expected fields"""
		result = await self.tuner.afind(
			structured_llms=[self.perfect_llm],
			prompts=["Extract: {input}"],
			scenarios=[self.scenario1],
			temperature=0.0,
		)

		# Check all required fields are present
		self.assertIsNotNone(result.model)
		self.assertIsNotNone(result.prompt)
		self.assertIsNotNone(result.temperature)
		self.assertIsNotNone(result.individual_scores)
		self.assertIsNotNone(result.target_score)

		# Check types
		self.assertIsInstance(result.individual_scores, np.ndarray)
		self.assertIsInstance(result.target_score, float)

	async def test_individual_scores_aggregation(self):
		"""Test that individual scores are properly aggregated across scenarios"""
		llm = MultiCallMockLLM(
			[self.scenario1.expected_output, self.scenario2.expected_output], 3
		)
		result = await self.tuner_multi.afind(  # repeat=3
			structured_llms=[llm],
			prompts=["Extract: {input}"],
			scenarios=[self.scenario1, self.scenario2],  # 2 scenarios
			temperature=0.0,
		)

		# Should have 3 repeats × 2 scenarios = 6 scores
		self.assertEqual(len(result.individual_scores), 6)
		# All should be perfect scores
		np.testing.assert_array_equal(result.individual_scores, np.ones(6))

	# Concurrency and performance tests
	async def test_concurrent_execution(self):
		"""Test that combinations are executed concurrently"""
		import time

		# Use a slow mock to test concurrency
		class SlowMockLLM(MockLLM):
			def invoke(self, input, config=None):  # type: ignore
				import time

				time.sleep(0.1)  # Simulate slow LLM
				return self.expected_output

		slow_llm = SlowMockLLM(PersonOutput(name="Hannah", age=30, weight=60.0))

		start_time = time.time()

		await self.tuner.afind(
			structured_llms=[slow_llm, slow_llm],  # 2 models
			prompts=["Prompt 1: {input}", "Prompt 2: {input}"],  # 2 prompts
			scenarios=[self.scenario1],
			temperature=[0.0, 0.5],  # 2 temperatures
		)

		elapsed_time = time.time() - start_time

		# Should take ~0.1s (concurrent) not ~0.8s (sequential for 8 combinations)
		self.assertLess(elapsed_time, 0.5)

	# with_config method testing
	async def test_temperature_with_config_called(self):
		"""Test that with_config is called with temperature when temperature > 0"""
		mock_llm = Mock(spec=MockLLM)
		mock_llm.with_config.return_value = self.perfect_llm
		mock_configured_llm = mock_llm.with_config.return_value

		with patch.object(
			self.tuner,
			"atune_scenario",
			return_value=IntermediateTuneResult(
				model=mock_configured_llm,
				prompt="test",
				individual_scores=np.array([1.0]),
			),
		) as _:
			await self.tuner.afind(
				structured_llms=[mock_llm],
				prompts=["Extract: {input}"],
				scenarios=[self.scenario1],
				temperature=0.5,
			)

			# Verify with_config was called with temperature
			mock_llm.with_config.assert_called_with(temperature=0.5)

	async def test_no_with_config_when_temp_zero(self):
		"""Test that with_config is not called when temperature is 0"""
		mock_llm = Mock(spec=MockLLM)

		with patch.object(
			self.tuner,
			"atune_scenario",
			return_value=IntermediateTuneResult(
				model=mock_llm, prompt="test", individual_scores=np.array([1.0])
			),
		) as _:
			await self.tuner.afind(
				structured_llms=[mock_llm],
				prompts=["Extract: {input}"],
				scenarios=[self.scenario1],
				temperature=0.0,
			)

			# Verify with_config was not called
			mock_llm.with_config.assert_not_called()

	# Logging tests
	async def test_logging_output(self):
		"""Test that appropriate logging occurs"""
		with patch.object(self.tuner, "logger") as mock_logger:
			await self.tuner.afind(
				structured_llms=[self.perfect_llm],
				prompts=["Extract: {input}"],
				scenarios=[self.scenario1],
				temperature=0.0,
			)

			# Check that logging calls were made
			self.assertGreater(mock_logger.info.call_count, 0)

			# Check specific log messages
			log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
			self.assertTrue(
				any("Starting LLM tuning session" in call for call in log_calls)
			)
			self.assertTrue(
				any("Finished LLM tuning session" in call for call in log_calls)
			)

	# Edge cases
	async def test_empty_lists_raises_error(self):
		"""Test behavior with empty input lists"""
		# This should fail during iteration - empty combinations
		with self.assertRaises(Exception):
			await self.tuner.afind(
				structured_llms=[],  # Empty
				prompts=["Extract: {input}"],
				scenarios=[self.scenario1],
				temperature=0.0,
			)

	async def test_best_selection_logic(self):
		"""Test that the correct 'best' result is selected"""
		# Create LLMs with different performance levels
		excellent_llm = MockLLM(
			PersonOutput(name="Hannah", age=30, weight=60.0)
		)  # 3/3 = 1.0
		good_llm = MockLLM(
			PersonOutput(name="Hannah", age=25, weight=60.0)
		)  # 2/3 = 0.67
		poor_llm = MockLLM(PersonOutput(name="Wrong", age=99, weight=99.0))  # 0/3 = 0.0

		result = await self.tuner.afind(
			structured_llms=[
				poor_llm,
				good_llm,
				excellent_llm,
			],  # Order shouldn't matter
			prompts=["Extract: {input}"],
			scenarios=[self.scenario1],
			temperature=0.0,
		)

		# Should select the excellent LLM
		self.assertEqual(result.target_score, 1.0)
		self.assertIs(result.model, excellent_llm)


if __name__ == "__main__":
	unittest.main()
