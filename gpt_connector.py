import openai
import tenacity
import os
from dotenv import load_dotenv
import asyncio

from pydantic import BaseModel, Field
from typing import AsyncGenerator
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
load_dotenv()

# Get the OpenAI API key from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the key is loaded correctly
if not groq_api_key:  
    raise ValueError("Groq API key not found. Make sure it's set in the .env file.")


# Pydantic model for the response
class GPTResponse(BaseModel):
    answer: str = Field(description=""" AI Assistant Answer Only """)
    complete: bool = False


class GPTConnector:

    def __init__(self, model: str = 'llama-3.2-11b-vision-preview', temperature: float = 0.0,
                 system_message: str = "You are an expert AI Tutor"):
        self.model = model
        self.temperature = temperature
        self.system_message = system_message

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(min=5, max=60),
        stop=tenacity.stop_after_attempt(5),
    )
    async def get_gpt_response_stream(self, question: str) -> AsyncGenerator[GPTResponse, None]:

        chat_model = ChatGroq(
            model_name=self.model,
            groq_api_key=groq_api_key,
            streaming=True,
            temperature=self.temperature,
        )

        prompt_messages: list[BaseMessage] = [
            SystemMessage(content=self.system_message),
            HumanMessage(content=question)
        ]

        try:
            async for chunk in chat_model.astream(prompt_messages):
                yield GPTResponse(answer=chunk.content)
        except openai.OpenAIError as e:
            yield GPTResponse(answer=f"Error: {str(e)}", complete=True)

        # Mark the completion of the response
        yield GPTResponse(answer="", complete=True)


async def test():
    gpt_connector = GPTConnector()
    answer = ""
    async for res in gpt_connector.get_gpt_response_stream(question="What is polymorphism"):
        if res.answer:
            answer += res.answer
    print(f"{answer=}")


if __name__ == "__main__":
    asyncio.run(test())
