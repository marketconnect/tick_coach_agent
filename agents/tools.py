from langchain_core.tools import tool
@tool
def finalize_answer(draft: str) -> str:
    """Вызови ПЕРЕД финальным ответом, чтобы показать пользователю черновик и получить подтверждение/правки."""
    return draft
