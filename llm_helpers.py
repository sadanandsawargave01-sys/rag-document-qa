import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY
from language_utils import detect_query_language, language_name
from prompts import SUMMARISE_PROMPT

GEMINI_MODEL = "gemini-2.5-flash"

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GOOGLE_API_KEY,
    temperature=0,
    top_p=0.5,
)


def detect_language(text: str) -> str:
    return detect_query_language(text)


def build_prompt(question: str, lang: str) -> str:
    target_language_name = language_name(lang)
    return f"""
Answer the question in {target_language_name} in one sentence only.
Provide a direct textbook-style definition.
No extra explanation.

Question: {question}
"""


async def get_llm_general_answers(questions, target_language=None):
    async def generate_answer(question):
        lang = target_language or detect_language(question)
        prompt = build_prompt(question, lang)

        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, lambda: llm.invoke(prompt))

        return answer.content if hasattr(answer, "content") else str(answer)

    tasks = [generate_answer(q) for q in questions]
    return await asyncio.gather(*tasks)


async def get_llm_summary(content: str, target_language=None):
    lang = target_language or detect_language(content)

    prompt = SUMMARISE_PROMPT.format(content=content)
    prompt += f"\nProvide the summary in {language_name(lang)} language."

    answer = await get_llm_general_answers([prompt], target_language=lang)
    return answer[0] if answer else ""
