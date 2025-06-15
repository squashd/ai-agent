import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentResponse


DEFAULT_PROMPT = "Why is Boot.dev such a great place to learn backend development? Use one paragraph maximum."


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = get_user_prompt()
    response = generate_response(client, prompt)

    print(response.text)
    report_response_tokens(response)


def generate_response(c: genai.Client, prompt: str):
    DEFAULT_MODEL = "gemini-2.0-flash-001"

    response = c.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
    )
    if not response:
        raise Exception("Failed to generate response")
    return response


def report_response_tokens(resp: GenerateContentResponse) -> None:
    metadata = resp.usage_metadata
    if not metadata:
        print("No response metadata")
        return
    print(f"Prompt tokens: {metadata.prompt_token_count}")
    print(f"Response tokens: {metadata.candidates_token_count}")


def get_user_prompt() -> str:
    args = sys.argv[1:]
    if not args:
        raise Exception("No prompt provided")
    return args[0]


if __name__ == "__main__":
    main()

