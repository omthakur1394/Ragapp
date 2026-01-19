from rag_chain import bulider_rag_chain
from langchain_core.messages import HumanMessage, AIMessage


def main():
    # Build LCEL RAG chain
    rag = bulider_rag_chain()

    # In-memory chat history (session-level)
    chat_history = []

    print("🤖 LCEL RAG Bot with History Ready (type 'exit' to quit)")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("👋 Bye")
            break

        if not user_input:
            continue

        # Invoke LCEL chain
        response = rag.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        print(f"Bot: {response}")

        # Update chat history AFTER response
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))


if __name__ == "__main__":
    main()
