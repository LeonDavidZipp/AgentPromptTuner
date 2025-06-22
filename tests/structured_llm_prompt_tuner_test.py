from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from typing import Any
from pydantic import BaseModel
import unittest
import warnings
import asyncio
import numpy as np
from unittest.mock import patch, AsyncMock
from src.llm_prompt_tuner import (
	StructuredLLMPromptTuner,
	Scenario,
	TuneOutcome,
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

	def with_config(self, **kwargs):  # type: ignore[override]
		pass


class FailingMockLLM(Runnable[LanguageModelInput, dict[str, Any] | BaseModel]):
	def invoke(self, input: str, config=None) -> BaseModel:  # type: ignore[override]
		raise RuntimeError("LLM failed")

	async def ainvoke(self, input, config=None) -> BaseModel:  # type: ignore[override]
		pass

	def with_config(self, **kwargs):  # type: ignore[override]
		pass


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


if __name__ == "__main__":
	unittest.main()
