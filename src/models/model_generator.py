from typing import Any, TypeAlias
from pydantic import BaseModel, create_model

NestedDict: TypeAlias = dict[str, "NestedDict"] | list["NestedDict"] | str | int | float | bool

def create_nested_model(input_dict: NestedDict, level: int = 0) -> BaseModel:
	"""
	Creates a Pydantic BaseModel instance from a nested dictionary.

	Args:
		input_dict: The input dictionary to create the model from.
		level: Current recursion level (for model naming).
		model_name: Base name for the generated model.

	Returns:
		BaseModel: A Pydantic BaseModel instance created from the input dictionary.
	"""

	# Handle the case where input_dict is not actually a dict
	if not isinstance(input_dict, dict):
		raise ValueError("Input must be a dictionary")

	fields: dict[str, Any] = {}

	for key, val in input_dict.items():
		if isinstance(val, dict):
			# Create nested model instance
			nested_instance = create_nested_model(val, level + 1)
			fields[key] = (type(nested_instance), nested_instance)
			
		elif isinstance(val, list):
			if not val:
				# Empty list
				fields[key] = (list[Any], [])
			elif all(isinstance(item, dict) for item in val):
				# list of dictionaries - create model instances
				nested_instances: list[BaseModel] = [
					create_nested_model(item, level + 1) 
					for item in val
				]
				if nested_instances:
					# Use the type of the first instance for the list type
					item_type = type(nested_instances[0])
					fields[key] = (list[item_type], nested_instances)
				else:
					fields[key] = (list[Any], [])
			else:
				# list of primitives or mixed types
				if val and all(isinstance(item, type(val[0])) for item in val): # type: ignore
					# All items are the same type
					fields[key] = (list[type(val[0])], val)
				else:
					# Mixed types
					fields[key] = (list[Any], val)
		else:
			# Primitive value
			fields[key] = (type(val), val)

	# Create the model class and return an instance
	ModelClass = create_model(f"auto_generated{level}", field_definitions=fields)
	return ModelClass()
