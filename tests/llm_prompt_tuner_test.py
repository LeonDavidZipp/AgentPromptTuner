from llm_prompt_tuner import Test, LLMPromptTester  # type: ignore[import]
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from typing import Any
from pydantic import BaseModel
import unittest


class LLMPromptTesterTest(unittest.IsolatedAsyncioTestCase):
	def setUp(self):
		"""Set up the LLMPromptTester instance for testing."""
		self.tester = LLMPromptTester(
			prompt_improvement_llm=None,
			prompt_improvement_prompt=None,
			improvement_retries=None,
			validation_llm=None,
			validation_prompt=None,
		)

	async def test_arun_structured(self):
		"""Test the arun_structured method of LLMPromptTester."""
		structured_llms: list[
			Runnable[LanguageModelInput, dict[str, Any] | BaseModel]
		] = []
		prompts: list[ChatPromptTemplate] = []
		tests: list[Test] = []

		results = await self.tester.arun_structured(structured_llms, prompts, tests)
		self.assertIsInstance(results, list)
