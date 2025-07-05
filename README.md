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
A simple example of how to use the `StructuredLLMPromptTuner`. Remember to replace the model and provider names with your own.
```python
from langchain.chat_models import init_chat_model
from llm_prompt_tuner import StructuredLLMPromptTuner, Scenario, TuneResult
from typing import Annotated
from pydantic import BaseModel
from dotenv import load_dotenv

async def main():
    load_dotenv("path/to/.env")

    improver_llm = init_chat_model(
        model="improver_model",
        model_provider="improver_provider"
    )

    class ExampleOutput(BaseModel):
        field1: Annotated[int, "Field 1"]
        field2: Annotated[int, "Field 2"]

    structured_models = [
        init_chat_model(model="example_model1", model_provider="example_provider1").with_structured_output(ExampleOutput),
        init_chat_model(model="example_model2", model_provider="example_provider2").with_structured_output(ExampleOutput)
    ]

    prompts = [
        "Extract field1 and field2: {some_text}",
        "Please provide field1 and field2 from the following text: {some_text}"
    ]

    inputs = [
        {"some_text": "This is an example text with field1=10 and field2=20."},
        {"some_text": "Another example with field1=30 and field2=40."}
    ]

    expected_outputs = [
        ExampleOutput(field1=10, field2=20),
        ExampleOutput(field1=30, field2=40)
    ]

    scenarios = [
        Scenario(
            input=inputs[0],
            expected_output=expected_outputs[0],
        ),
        Scenario(
            input=inputs[1],
            expected_output=expected_outputs[1],
        )
    ]

    tuner = StructuredLLMPromptTuner(
        target="median_score",
        repeat=3,
        prompt_improvement_llm=improver_llm,
        initial_prompt_improvement_prompt="Improve this prompt for better results.",
        improvement_retries=2,
    )

    temperatures = [0.0, 0.1, 0.2]

    result = await tuner.afind(
        structured_llms=structured_models,  # type: ignore
        prompts=prompts,
        scenarios=scenarios,
        temperature=temperatures,
    )

    print("Best model:", result.model)
    print("Best prompt:", result.prompt)
    print("Best temperature:", result.temperature)
    print("Best target score:", result.target_score)
    print("Individual scores:", result.individual_scores)

if __name__ == "__main__":
    asyncio.run(main())
```
