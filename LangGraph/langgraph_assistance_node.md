# Chatbot Assistant Node
def chatbot(state: State):
    response = llm.invoke(state['messages'])
    return {"messages": [response]}


# Document Summarizer Assistant Node
def summarize_pdf(state: State):
    document_text = state.get("document_text", "")
    summary = llm.invoke([{"role": "system", "content": "Summarize the following document."}, {"role": "user", "content": document_text}])
    return {"summary": summary}


# Multi-Tool Agent Assistant Node
def use_tool(state: State):
    tool_input = state.get("tool_input", "")
    # Decision logic (simplified)
    if "calculate" in tool_input:
        result = calculator_tool(tool_input)  # hypothetical tool call
    elif "search" in tool_input:
        result = search_tool(tool_input)  # hypothetical tool call
    else:
        result = llm.invoke([{"role": "user", "content": tool_input}])
    return {"tool_result": result}


# RAG System Assistant Node
def retrieve_and_answer(state: State):
    query = state.get("query", "")
    retrieved_docs = retriever.retrieve(query)  # hypothetical retriever call
    input_for_llm = [{"role": "system", "content": "Answer based on the following documents."}] + [{"role": "user", "content": doc} for doc in retrieved_docs] + [{"role": "user", "content": query}]
    answer = llm.invoke(input_for_llm)
    return {"answer": answer}
