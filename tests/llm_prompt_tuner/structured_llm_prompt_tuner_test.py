from langchain.chat_models import init_chat_model
from pydantic import BaseModel
import unittest
import warnings
from src.llm_prompt_tuner import (
	StructuredLLMPromptTuner,
	Scenario,
	TuneResult,
)
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")

if not OPENAI_API_KEY:
	warnings.warn("OPENAI_API_KEY is not set. Tests may fail due to missing API keys.")
elif not OPENROUTER_API_KEY:
	warnings.warn(
		"OPENROUTER_API_KEY is not set. Tests may fail due to missing API keys."
	)


class PersonInfo(BaseModel):
	name: str
	age: int
	city: str


class StructuredLLMPromptTunerTest(unittest.IsolatedAsyncioTestCase):
	def setUp(self):
		self.gpt35_turbo = init_chat_model(  # type: ignore
			model="gpt-3.5-turbo",
			model_provider="openai",
		).with_structured_output(PersonInfo)

		self.gpt_4o = init_chat_model(  # type: ignore
			model="gpt-4o",
			model_provider="openai",
		).with_structured_output(PersonInfo)

		self.gpt_3o = init_chat_model(  # type: ignore
			model="gpt-3.5-turbo",
			model_provider="openai",
		).with_structured_output(PersonInfo)

		self.person_text = {
			"text": "John Doe is a 30-year-old software engineer living in San Francisco. "
			"He enjoys hiking and playing video games during his free time."
		}

		self.person_prompts = [
			"Find the age, name, and city of the person in the text: {text}",
			"Extract the person's age, full name, and city from the following text: {text}",
			"Please identify and return the age, name, and city mentioned for the person in this text: {text}",
			"From the given text, locate the person's age (in years), their complete name, and the city they are associated with: {text}",
			"Parse the following text and extract three key pieces of information about the person: their age, their name, and their city: {text}",
			"Analyze this text and find the person's demographic information - specifically their age, full name, and city of residence: {text}",
			"Review the text below and identify: 1) the person's age, 2) their complete name, and 3) the city mentioned: {text}",
			"Looking at this text, what is the person's age, what is their name, and which city are they from? Text: {text}",
			"Extract personal details from the text: age (numeric), name (first and last), and city location: {text}",
			"Find and return the following information about the person mentioned in the text - age, full name, city: {text}",
		]

		# Complex multi-input example
		self.job_analysis_data = {
			"text": "Sarah Williams is applying for a senior developer position. She has 8 years of experience in Python and JavaScript. Her current salary is $95,000 and she's looking for a 20% increase.",
			"position": "Senior Software Developer",
			"required_experience": 5,
			"salary_budget": 120000,
			"location": "Remote",
		}

		self.job_analysis_prompts = [
			"Analyze the candidate profile in {text} for the {position} role. Required experience: {required_experience} years, Budget: ${salary_budget}, Location: {location}. Determine if they're a good fit.",
			"Given candidate info: {text}\nPosition: {position}\nMin Experience: {required_experience} years\nBudget: ${salary_budget}\nLocation: {location}\n\nEvaluate their suitability and salary expectations.",
			"Review this job application:\nCandidate: {text}\nRole: {position}\nExperience requirement: {required_experience}+ years\nSalary range: up to ${salary_budget}\nWork style: {location}\n\nProvide hiring recommendation.",
			"As an HR manager, assess this candidate: {text}\nFor position: {position}\nMinimum experience needed: {required_experience} years\nBudget limit: ${salary_budget}\nLocation type: {location}\n\nMake a hiring decision.",
			"Compare candidate profile against job requirements:\n• Candidate: {text}\n• Position: {position}\n• Experience needed: {required_experience} years\n• Max salary: ${salary_budget}\n• Work arrangement: {location}\n\nProvide match assessment.",
			"Job matching analysis:\nProfile: {text}\nTarget role: {position}\nExperience threshold: {required_experience} years\nBudget ceiling: ${salary_budget}\nLocation preference: {location}\n\nGenerate compatibility score and recommendation.",
		]

		# Additional complex scenario - product analysis
		self.product_review_data = {
			"review_text": "This laptop is amazing! Great performance, battery lasts 8 hours, screen is crystal clear. Only downside is it's a bit heavy at 4.2 lbs.",
			"product_category": "Laptop",
			"price_range": "800-1200",
			"target_audience": "Students",
			"key_features": [
				"battery life",
				"display quality",
				"performance",
				"portability",
			],
		}

		self.product_analysis_prompts = [
			"Analyze this {product_category} review: {review_text}\nPrice range: ${price_range}\nTarget: {target_audience}\nKey features to evaluate: {key_features}\nProvide feature rating and recommendation.",
			"Product assessment for {target_audience}:\nReview: {review_text}\nCategory: {product_category}\nPrice: ${price_range}\nImportant features: {key_features}\nRate each feature and overall suitability.",
			"Review analysis:\n• Product: {product_category}\n• Review: {review_text}\n• Price bracket: ${price_range}\n• Audience: {target_audience}\n• Focus areas: {key_features}\n\nGenerate feature scores and buying recommendation.",
		]

	async def test_tune(self):
		expected_output = PersonInfo(name="John Doe", age=30, city="San Francisco")
		scenario = Scenario(input=self.person_text, expected_output=expected_output)
		tuner = StructuredLLMPromptTuner(target="mean_score", repeat=1)

		# Test with single prompt
		result = await tuner.afind(
			structured_llms=[self.gpt_3o],  # type: ignore
			prompts=[self.person_prompts[0]],
			scenarios=[scenario],
			temperature=0.0,
		)

		# Basic assertions
		self.assertIsInstance(result, TuneResult)
		self.assertEqual(result.temperature, 0.0)
		self.assertGreaterEqual(result.target_score, 0.0)
		self.assertLessEqual(result.target_score, 1.0)

	async def test_tune_multiple_temperatures(self):
		expected_output = PersonInfo(name="John Doe", age=30, city="San Francisco")
		scenario = Scenario(input=self.person_text, expected_output=expected_output)
		tuner = StructuredLLMPromptTuner(target="median_score", repeat=1)

		result = await tuner.afind(
			structured_llms=[self.gpt_4o],  # type: ignore
			prompts=[self.person_prompts[0]],
			scenarios=[scenario],
			temperature=[0.0, 0.3, 0.7],
		)

		self.assertIsInstance(result, TuneResult)
		self.assertIn(result.temperature, [0.0, 0.3, 0.7])

	async def test_tune_multiple_prompts(self):
		expected_output = PersonInfo(name="John Doe", age=30, city="San Francisco")
		scenario = Scenario(input=self.person_text, expected_output=expected_output)
		tuner = StructuredLLMPromptTuner(target="best_case", repeat=1)

		result = await tuner.afind(
			structured_llms=[self.gpt35_turbo],  # type: ignore
			prompts=self.person_prompts[:3],
			scenarios=[scenario],
			temperature=0.0,
		)

		self.assertIsInstance(result, TuneResult)
		self.assertIn(result.prompt, self.person_prompts[:3])

	async def test_tune_invalid_temperature(self):
		expected_output = PersonInfo(name="John Doe", age=30, city="San Francisco")
		scenario = Scenario(input=self.person_text, expected_output=expected_output)
		tuner = StructuredLLMPromptTuner()

		with self.assertRaises(ValueError):
			await tuner.afind(
				structured_llms=[self.gpt35_turbo],  # type: ignore
				prompts=[self.person_prompts[0]],
				scenarios=[scenario],
				temperature=1.5,  # Invalid temperature > 1.0
			)

	async def test_tune_different_target_metrics(self):
		expected_output = PersonInfo(name="John Doe", age=30, city="San Francisco")
		scenario = Scenario(input=self.person_text, expected_output=expected_output)

		for target in ["worst_case", "cost_per_value"]:
			tuner = StructuredLLMPromptTuner(target=target, repeat=1)  # type: ignore
			result = await tuner.afind(
				structured_llms=[self.gpt35_turbo],  # type: ignore
				prompts=[self.person_prompts[0]],
				scenarios=[scenario],
				temperature=0.0,
			)
			self.assertIsInstance(result, TuneResult)
