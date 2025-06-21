from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from typing import Any
from pydantic import BaseModel
import unittest
from src.llm_prompt_tuner import StructuredLLMPromptTuner, Scenario, TuneOutcome


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


class PersonOutput(BaseModel):
	name: str
	age: int
	weight: float


class AScoreSingleScenarioTest(unittest.IsolatedAsyncioTestCase):
	def setUp(self):
		"""Set up the LLMPromptTester instance for testing."""
		self.tuner1 = StructuredLLMPromptTuner(repeat=1)
		self.tuner2 = StructuredLLMPromptTuner(repeat=2)
		self.tuner3 = StructuredLLMPromptTuner(repeat=3)

		self.scenario1 = Scenario(
			input={"input": "Hannah is 30 years old and weighs 60kg."},
			expected_output=PersonOutput(name="Hannah", age=30, weight=60.0),
		)
		self.scenario2 = Scenario(
			input={"input": "John is 25 years old and weighs 70kg."},
			expected_output=PersonOutput(name="John", age=25, weight=70.0),
		)
		self.scenario3 = Scenario(
			input={"input": "Alice is 28 years old and weighs 55kg."},
			expected_output=PersonOutput(name="Alice", age=28, weight=55.0),
		)

	async def test_ascore_single_scenario(self):
		# 1.0
		result1 = await self.tuner1.ascore_single_scenario(
			scenario=self.scenario1,
			llm=MockLLM(expected_output=self.scenario1.expected_output),
			prompt="Extract the person's name, age, and weight from the input: {input}",
		)

		self.assertAlmostEqual(result1.target_score, 1.0, delta=0.01)
		self.assertEqual(result1.state, TuneOutcome.SUCCESS)

		# 0.666...
		result2 = await self.tuner2.ascore_single_scenario(
			scenario=self.scenario2,
			llm=MockLLM(
				expected_output=PersonOutput(
					name="John",
					age=25,
					weight=69,  # incorrect
				)
			),
			prompt="Extract the person's name, age, and weight from the input: {input}",
		)
		self.assertAlmostEqual(result2.target_score, 0.6666666666666666, delta=0.01)
		self.assertEqual(result2.state, TuneOutcome.SUCCESS)

		# 0.333...
		result3 = await self.tuner3.ascore_single_scenario(
			scenario=self.scenario3,
			llm=MockLLM(
				expected_output=PersonOutput(
					name="Alice",
					age=27,  # incorrect
					weight=120.0,  # incorrect
				)
			),
			prompt="Extract the person's name, age, and weight from the input: {input}",
		)

		self.assertAlmostEqual(result3.target_score, 0.3333333333333333, delta=0.01)
		self.assertEqual(result3.state, TuneOutcome.SUCCESS)
