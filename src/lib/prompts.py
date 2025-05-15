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
        "Descompón la consulta en sub-preguntas para investigación:\n"
        "**Consulta**:\n {query} \n\n"
        "Considera:\n"
        "1. Diferentes aspectos de la pregunta\n"
        "2. Posibles interpretaciones\n"
        "3. Conceptos relacionados\n"
    ),
    "generate_answer": (
        "Genera una respuesta con citaciones numéricas EN EL IDIOMA DEL USUARIO usando este formato:\n"
        "1. Cada afirmación relevante lleva [n] al final\n"
        "2. Lista de referencias al final con texto completo\n\n"
        "**Instrucciones**:\n"
        "1. Usar formato: Frase [1]. Otra frase [2].\n"
        "2. Numeración consecutiva en toda la respuesta\n"
        "3. Al final:\n"
        "   [1] Texto completo del fragmento citado\n"
        "   [2] Siguiente fragmento citado\n"
        "4. Conservar el idioma original de la consulta\n"
        "5. Máximo 5 citaciones por respuesta\n\n"
        "**Ejemplo Español**:\n"
        "Consulta: '¿Qué beneficios tiene X?'\n"
        "Respuesta:\n"
        "'X mejora la eficiencia operativa [1]. Además, reduce costos según estudios recientes [2].\n\n"
        "[1] 'La tecnología X aumenta un 40% la productividad...'\n"
        "[2] 'Estudio de 2023 muestra ahorros promedio de $2M...'\n\n"
        "**English Example**:\n"
        "Query: 'Technical advantages of Y'\n"
        "Answer:\n"
        "'Y demonstrates superior thermal resistance [1]. Its modular design allows quick deployment [2].\n\n"
        "[1] 'Testing results: Y withstands 500°C for...'\n"
        "[2] 'Assembly manual section 3.2: modular components...'\n\n"
        "**Contexto**:\n {context}\n\n"
        "**Consulta**:\n {query}\n\n"
    ),
    "validate_answer": (
        "Verificar:\n"
        "1. Si la respuesta responde completamente la consulta.\n"
        "2. Coincidencia numérica entre citaciones y referencias.\n"
        "3. Formato bilingüe correcto.\n\n"
        "Consulta: {query}\n"
        "Respuesta: {answer}\n\n"
        "Identifica la informacion faltante o inseguridad de la respuesta. Lista hasta 3 fallas si las hay."
    ),
}
