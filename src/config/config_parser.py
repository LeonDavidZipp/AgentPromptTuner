import json
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from typing import Any, Literal, TypeAlias
from pydantic import BaseModel, ConfigDict
from collections import defaultdict
from enum import IntEnum
import numpy as np
from numpy.typing import NDArray
import asyncio
import itertools
import warnings
import logging
import time
from ..shared import Scenario


@dataclass(frozen=True)
class Config:
	structured_llms: list[Runnable[LanguageModelInput, dict[str, Any] | BaseModel]]
	prompts: list[str]
	scenarios: list[Scenario]
	temperature: list[int | float]

	@classmethod
	def from_file(cls, path: str) -> "Config":
		with open(path, "r") as f:
			config = json.load(f)

		required_keys = ["models", "prompts", "scenarios"]
		if not all(key in config for key in required_keys):
			missing_keys = [key for key in required_keys if key not in config]
			raise ValueError(f"Missing required keys: {missing_keys}")
