from pydantic import BaseModel
from typing import Any


class Scenario(BaseModel):
	"""
	A scenario consist of the following:
	- input: the input to predict an output for
	- expected_output: the expected output to compare against
	"""

	input: dict[str, Any]
	expected_output: Any
