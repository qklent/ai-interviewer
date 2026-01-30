Using Structured Outputs with OpenAI Python SDK

A Comprehensive Guide for Coding Agents

What Are Structured Outputs?

Structured Outputs is a feature in the OpenAI API that guarantees the
model will always generate responses that match a JSON Schema you
provide. Instead of getting unpredictable free-form text that you have
to parse and validate yourself, you define the exact structure you want,
and the API ensures the response follows it perfectly.

Think of it like filling out a form versus writing on a blank piece of
paper. With structured outputs, you provide the form template (schema),
and the AI fills it out correctly every time.

Why Use Structured Outputs?

-   **Reliability:** The model\'s output will always match your
    schema---no missing fields, no invalid data types, no hallucinated
    keys.

-   **Type Safety:** When using Pydantic models, you get automatic type
    checking and validation in your Python code.

-   **Easier Integration:** Output can be directly used in databases,
    APIs, or UI components without extra parsing or error handling.

-   **Better Performance:** On complex schemas, models with structured
    outputs (like gpt-4o-2024-08-06) achieve 100% reliability compared
    to under 40% for older models.

How Structured Outputs Work

Structured outputs use a technique called constrained decoding with
context-free grammars. The model is forced to only generate tokens that
match your provided JSON Schema, making it impossible for it to produce
invalid output.

There are two main ways to use structured outputs:

1.  **Function Calling (via tools):** When you want the model to call
    functions with structured parameters. Set strict: true in your tool
    definition.

2.  **Response Format:** When you want the model\'s direct response to
    follow a specific structure. Use the response_format parameter with
    a JSON schema or Pydantic model.

Installation & Setup

First, install the OpenAI Python SDK and Pydantic:

pip install openai pydantic \--break-system-packages

Set up your API key:

import os from openai import OpenAI os.environ\[\'OPENAI_API_KEY\'\] =
\'your-api-key-here\' client = OpenAI()

Basic Example: Using Pydantic Models

The easiest way to use structured outputs is with Pydantic models.
Here\'s a simple example that extracts user information:

from pydantic import BaseModel from openai import OpenAI client =
OpenAI() \# Define your output structure class User(BaseModel): name:
str age: int email: str \# Make the API call response =
client.beta.chat.completions.parse( model=\"gpt-4o-2024-08-06\",
messages=\[ {\"role\": \"system\", \"content\": \"Extract user
information from the text.\"}, {\"role\": \"user\", \"content\": \"John
Doe is 28 years old. Email: john@example.com\"} \], response_format=User
) \# Access the parsed output user =
response.choices\[0\].message.parsed print(user.name) \# \"John Doe\"
print(user.age) \# 28 print(user.email) \# \"john@example.com\"

Key Points:

-   Use client.beta.chat.completions.parse() for structured outputs with
    Pydantic

-   Pass your Pydantic model to response_format

-   Access parsed output via response.choices\[0\].message.parsed

-   The model used must support structured outputs (e.g.,
    gpt-4o-2024-08-06)

Complex Example: Nested Structures

You can create complex nested structures with lists and nested objects:

from pydantic import BaseModel, Field from typing import List class
Step(BaseModel): explanation: str output: str class
MathResponse(BaseModel): steps: List\[Step\] final_answer: str response
= client.beta.chat.completions.parse( model=\"gpt-4o-2024-08-06\",
messages=\[ {\"role\": \"system\", \"content\": \"Solve math problems
step by step.\"}, {\"role\": \"user\", \"content\": \"Solve: 8x + 31 =
2\"} \], response_format=MathResponse ) result =
response.choices\[0\].message.parsed for i, step in
enumerate(result.steps, 1): print(f\"Step {i}: {step.explanation}\")
print(f\"Output: {step.output}\") print(f\"Final Answer:
{result.final_answer}\")

Using the New Responses API

OpenAI now recommends using the Responses API for new projects. Here\'s
how to use structured outputs with it:

from pydantic import BaseModel from openai import OpenAI client =
OpenAI() class CalendarEvent(BaseModel): name: str date: str
participants: list\[str\] response = client.responses.parse(
model=\"gpt-4o-2024-08-06\", input=\[ {\"role\": \"system\",
\"content\": \"Extract event information.\"}, {\"role\": \"user\",
\"content\": \"Alice and Bob are going to a science fair on Friday.\"}
\], text_format=CalendarEvent ) event = response.output_parsed
print(event.name) \# \"Science Fair\" print(event.date) \# \"Friday\"
print(event.participants) \# \[\"Alice\", \"Bob\"\]

Handling Refusals

Sometimes the model may refuse to respond (for safety reasons). Always
check for refusals:

response = client.beta.chat.completions.parse(
model=\"gpt-4o-2024-08-06\", messages=\[\...\], response_format=MySchema
) message = response.choices\[0\].message if message.refusal: \# Handle
the refusal print(f\"Model refused: {message.refusal}\") else: \# Use
the parsed output data = message.parsed \# Process data\...

Streaming Structured Outputs

You can stream structured outputs to get results as they\'re generated:

from typing import List from openai import OpenAI from pydantic import
BaseModel class EntitiesModel(BaseModel): attributes: List\[str\]
colors: List\[str\] animals: List\[str\] client = OpenAI() with
client.responses.stream( model=\"gpt-4.1\", input=\[ {\"role\":
\"system\", \"content\": \"Extract entities from the text\"}, {\"role\":
\"user\", \"content\": \"The quick brown fox jumps over the lazy dog\"}
\], text_format=EntitiesModel ) as stream: for event in stream: if
event.type == \"response.output_text.delta\": print(event.delta,
end=\"\") elif event.type == \"response.completed\":
print(\"\\nCompleted\") final_response = stream.get_final_response()
entities = final_response.output_parsed

Best Practices

-   **Use Pydantic Models:** Always prefer Pydantic models over raw JSON
    schemas. They provide better type safety and validation.

-   **Add Field Descriptions:** Use Field(\..., description=\"\...\") to
    help the model understand what each field should contain.

-   **Check for Refusals:** Always check message.refusal before
    accessing parsed data.

-   **Use Appropriate Models:** Structured outputs work best with
    gpt-4o-2024-08-06 or newer models.

-   **Handle Validation Errors:** Even with structured outputs, validate
    the semantic correctness of the data.

-   **Provide Clear Instructions:** Use detailed system prompts to guide
    the model on how to fill the schema.

Common Use Cases

-   **Data Extraction:** Extract structured information from
    unstructured text (emails, documents, web pages)

-   **Form Filling:** Automatically populate forms or databases from
    natural language input

-   **Agentic Workflows:** Build multi-step AI agents that take actions
    based on structured decisions

-   **API Integration:** Generate API payloads in the exact format
    required by external services

-   **Content Classification:** Categorize and tag content with
    consistent, structured metadata

Limitations & Considerations

-   **Model Support:** Only newer models support structured outputs
    (gpt-4o-2024-08-06 and later)

-   **Schema Complexity:** Very complex schemas may impact performance.
    Keep schemas as simple as possible.

-   **Not a Validation Replacement:** Structured outputs guarantee
    format, not semantic correctness. Always validate the content.

-   **Safety Features Still Apply:** The model can still refuse unsafe
    requests even with structured outputs enabled.

Additional Resources

-   Official Documentation:
    https://platform.openai.com/docs/guides/structured-outputs

-   Pydantic Documentation: https://docs.pydantic.dev/

-   OpenAI Python SDK: https://github.com/openai/openai-python

-   Instructor Library (Advanced): https://python.useinstructor.com/