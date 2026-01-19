from operator import itemgetter 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from vectorstore import get_vectorstore
from config import LLM_MODEL,GROQ_API_KEY


def bulider_rag_chain():
    vectorstore = get_vectorstore()
    retiver = vectorstore.as_retriever(search_kwargs={"k":8})

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1
    )
    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the chat history and the latest user question, "
            "rewrite the question so it is fully standalone. "
            "Do NOT answer the question. Just rewrite it if needed."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    Standalone = (
        contextualize_prompt
        |llm
        |StrOutputParser()

    )
    retiver_chain = (
        Standalone
        |retiver
    )
    qa_system_prompt = """
    You are a precise, factual, and clear information assistant. Your primary goal is to answer the user's question based **exclusively** on the provided context. Follow these rules strictly:

    1. **Source-Driven Answers:** Ground every part of your answer in the provided context. Do not use prior knowledge or speculate.
    2. **Clarity & Conciseness:** Provide a direct, well-structured answer. Use clear language and avoid unnecessary elaboration, fluff, or tangential information.
    3. **Factual Tone:** With a low temperature setting, your responses should be deterministic and factual. Avoid creative language, opinion, or hedging phrases like "I think" or "it might be."
    4. **Explicit Citation:** For each key point or claim in your answer, cite the relevant document(s) by name or ID (e.g., [Doc A], [Report_2023]) to show transparency.
    5. **Handling Uncertainty:**
       - If the context contains a complete answer, deliver it clearly and cite the sources.
       - If the context is **partial or unclear**, state the limitations explicitly (e.g., "Based on the provided documents, the evidence on X is incomplete...").
       - If the context is **irrelevant or absent**, do not invent an answer. Clearly state: "The provided documents do not contain information necessary to answer this question."
    6. **Structure (if complex):** For multi-part questions or complex topics, use bullet points or numbered lists to enhance clarity, but keep explanations tight.
    7. **Engaging Format:** Use relevant emojis (like 📊, 🤖, ✅, 🚀) to make the answer visually appealing and easier to read, but keep the tone professional.
    8. **Comparison:** If the answer requires comparing two things, synthesize the answer by combining facts found in different sections of the text.

    **Your response style should mirror a technical summary: objective, efficient, and rooted solely in the provided evidence.**

    Context:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
    )

    rag_chain = (
        {
            "context":retiver_chain | format_docs,
            "input":itemgetter("input"),
            "chat_history":itemgetter("chat_history")
        }
        |qa_prompt
        |llm
        |StrOutputParser()
    )
    return rag_chain

def format_docs(docs):
    """
    Turns the list of documents into a clean string for the LLM.
    Replaces ugly UUIDs with actual filenames.
    """
    formatted_context = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        if "/" in source:
            source = source.split("/")[-1]
        if "\\" in source:
            source = source.split("\\")[-1]
            
        page = doc.metadata.get("page", "")
        citation_label = f"{source} (Page {page})" if page else source
       
        formatted_context.append(f"Source: [{citation_label}]\nContent: {doc.page_content}")
        
    return "\n\n---\n\n".join(formatted_context)