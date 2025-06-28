import json
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from typing import Any
from pydantic import BaseModel
from ..shared import Scenario
from .model_generator import create_nested_model


@dataclass(frozen=True)
class Config:
	structured_llms: list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]]
	prompts: list[str]
	scenarios: list[Scenario]
	temperature: list[int | float]

	@classmethod
	def from_file(cls, path: str) -> "Config":
		"""
		Loads a configuration from a JSON file.
		Args:
			path (str): The path to the JSON configuration file.
		Returns:
			Config: An instance of Config containing the loaded configuration.
		Raises:
			ValueError: If the JSON file does not contain the required keys or if any key is
			missing or has an unexpected type.
		"""

		with open(path, "r") as f:
			config: dict[str, Any] = json.load(f)

		required_keys = ["models", "prompts", "temperature", "scenarios"]
		if not all(key in config for key in required_keys):
			missing_keys = [key for key in required_keys if key not in config]
			raise ValueError(f"Missing required keys: {missing_keys}")

		models_json: list[dict[str, str]] | None = config.get("models", [])
		if models_json is None:
			raise ValueError("'models' field not found in config")
		models: list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]] = [
			init_chat_model(
				model=model.get("model", None), provider=model.get("provider", None)
			)
			for model in models_json
		]

		if (prompts := config.get("prompts")) is None:
			raise ValueError("'prompts' field not found in config")

		if (temperature := config.get("temperature")) is None:
			raise ValueError("'temperature' field not found in config")

		scenarios_json: (
			list[dict[str, dict[str, list[int | float | str] | int | float | str]]]
			| None
		) = config.get("scenarios")
		if scenarios_json is None:
			raise ValueError("'scenarios' field not found in config")

		base_model = create_nested_model(
			scenarios_json
		)

		# if len()

		scenarios: list[Scenario] = []
		for s in scenarios_json:
			if (input_json := s.get("input")) is None:
				raise ValueError("'input' field not found in scenario")
			elif not isinstance(input_json, dict):  # type: ignore
				raise ValueError("'input' field must be a dictionary")
			elif expected_json := s.get("expected_output") is None:
				raise ValueError("'expected_json' field not found in scenario")

			scenarios.append(
				Scenario(
					input=input_json,
					expected_output=expected_json,
				)
			)

		return cls(
			structured_llms=models,
			prompts=prompts,
			scenarios=scenarios,
			temperature=temperature,
		)
