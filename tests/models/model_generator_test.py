from pydantic import BaseModel
import unittest
from src.models import create_nested_model

class ExpectedSimpleOutput(BaseModel):
	person: str
	age: int

class ExpectedNestedOutput(BaseModel):
	people: list[str]
	ages: list[int]

class CreateNestedModelTester(unittest.TestCase):
	def setUp(self):
		self.simple_input = {
			"person": "Nana",
			"age": 24
		}
		self.expected_simple = ExpectedSimpleOutput(person="Nana", age=24)
		self.nested_input = {
			"people": ["Earl", "Nana", "Pedro"],
			"ages": [33, 24, 45]
		}

	def test_simple_input(self):
		expected_attributes = [
			"person", "age"
		]
		simple_output = create_nested_model(self.simple_input, 0)
		for attr in expected_attributes:
			self.assertTrue(hasattr(simple_output, attr))


if __name__ == "__main__":
	unittest.main()
