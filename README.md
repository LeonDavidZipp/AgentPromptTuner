# LLM Prompt Tuner
A lightweight, open source framework for selecting the correct model-prompt-combination for various tasks.
Langchain-exclusive.
## Features
- structured & unstructured output
- LLM tuning (find the best LLM for a task)
- prompt tuning (find the best prompt for a task)
- temperature tuning
- optional AI-generated prompt improvements
- optional AI-enhanced output validation for unstructured output

## Usage
### Structured example:
```python
from langchain.chat_models import init_chat_model
from llm_prompt_tuner import StructuredLLMPromptTuner
from typing import Annotated
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv("path/to/.env")

improver_llm = init_chat_model(
    model="improver_model",
    model_provider="improver_provider"
)

class ExampleOutput(BaseModel):
    field1: Annotated[int, "Field 1"]
    field2: Annotated[int, "Field 2"]

models = [
    init_chat_model(model="example_model1", model_provider="example_provider1").with_structured_Output(ExampleOutput),
    init_chat_model(model="example_model2", model_provider="example_provider2").with_structured_Output(ExampleOutput)
]

tuner = StructuredLLMPromptTuner(
    target="median_score",
    repeat=3,
    prompt_improvement_llm=improver_llm,
    initial_prompt_improvement_prompt="Improve this prompt for better results.",
    improvement_retries=2,
)
```

### Unstructured example:
```python
print("Hello, world!")
```