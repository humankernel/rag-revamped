from typing import TypedDict, Final


class Prompt(TypedDict):
    query_plan: str
    generate_answer: str
    validate_answer: str


# TODO: add deepseek recomendation if math
# Please reason step by step, and put your final answer within \boxed{}.
# we recommend enforcing the model to initiate its response with "<think>\n" at the beginning of every output

PROMPT: Final[Prompt] = {
    "query_plan": (
        "Break the query into comprehensive sub-queries for research:\n"
        "**Query**:\n {query} \n\n"
        "Consider:\n"
        "1. Different aspects of the question\n"
        "2. Possible interpretations\n"
        "3. Related concepts\n"
    ),
    "generate_answer": (
        "Generate a clear, concise, and well-supported answer to the given query using the provided document context. Cite sources explicitly using [Doc X] notation.\n\n"
        "**Guidelines**:\n"
        "1. The answer must be factually grounded in the provided documents.\n"
        "2. Use citations where applicable (e.g., 'According to e.g: [Doc 1], ...').\n"
        "3. Do not include information not present in the provided context.\n"
        "4. Maintain a professional and neutral tone.\n"
        "5. If the context lacks sufficient information, acknowledge it clearly.\n\n"
        "**Examples**:\n"
        "1. Good Answer (With Supporting Evidence):\n"
        "   Query: 'What are the benefits of X technology?'\n"
        "   Expected Output:\n"
        '   "X technology provides improved efficiency and scalability. According to [Doc 2], it reduces processing time by 40%, making it ideal for large-scale applications. Furthermore, [Doc 4] highlights its cost-effectiveness compared to traditional methods."\n\n'
        "2. Good Answer (Acknowledging Insufficient Data):\n"
        "   Query: 'How does X compare to Y in energy consumption?'\n"
        "   Expected Output:\n"
        '   "The available documents discuss the efficiency of X but do not provide a direct comparison with Y. Further sources would be needed for a comprehensive analysis."\n\n'
        "3. Bad Answer (Unsubstantiated Claims):\n"
        "   Query: 'What are the benefits of X technology?'\n"
        "   Expected Output:\n"
        '   "X technology is the best and will revolutionize the industry." (Lacks citations and specifics)\n\n'
        "**Final Output**:\n"
        "Provide a well-structured answer following the above guidelines."
        "**Context**:\n {context}\n\n"
        "**Query**:\n {query}\n\n"
    ),
    "validate_answer": (
        "Verify if this answer fully addresses the question: \n"
        "Question: {query}\n"
        "Answer: {answer}\n"
        "Identify missing information or uncertainties. List up to 3 key gaps:"
    ),
}
