#instead of openai api we are using local ollama, so we need to change the imports and the llm initialization

import os
import re
from typing import Dict, List, Tuple

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import fitz
import faiss
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from htmlTemplates import bot_template, css, user_template
from langchain_core.stores import InMemoryStore
from langchain_community.chat_models import ChatOllama

custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

INSURANCE_FEATURE_PATTERNS = {
    "Room Rent": [
        r"room\s+rent[^\n\.]{0,120}",
        r"single\s+private\s+room[^\n\.]{0,120}",
        r"icu\s+rent[^\n\.]{0,120}",
    ],
    "Zero Dep": [
        r"zero\s*dep(?:reciation)?[^\n\.]{0,120}",
        r"bumper\s+to\s+bumper[^\n\.]{0,120}",
        r"depreciation\s+cover[^\n\.]{0,120}",
    ],
    "Inclusions": [
        r"inclusion[s]?[:\-]?[^\n]{0,200}",
        r"covered\s+expenses?[^\n\.]{0,120}",
        r"benefits?\s+covered[^\n\.]{0,120}",
    ],
    "Exclusions": [
        r"exclusion[s]?[:\-]?[^\n]{0,200}",
        r"not\s+covered[^\n\.]{0,120}",
        r"permanent\s+exclusion[s]?[^\n\.]{0,120}",
    ],
    "Waiting Period": [
        r"waiting\s+period[^\n\.]{0,120}",
        r"initial\s+waiting[^\n\.]{0,120}",
        r"pre[-\s]?existing\s+disease[^\n\.]{0,120}",
    ],
    "Co-pay": [
        r"co[-\s]?pay(?:ment)?[^\n\.]{0,120}",
        r"co[-\s]?insurance[^\n\.]{0,120}",
    ],
    "Deductible": [
        r"deductible[^\n\.]{0,120}",
        r"voluntary\s+deductible[^\n\.]{0,120}",
    ],
    "Sub-limits": [
        r"sub[-\s]?limit[s]?[^\n\.]{0,120}",
        r"capped\s+at[^\n\.]{0,120}",
        r"limit\s+of\s+liability[^\n\.]{0,120}",
    ],
}


def get_llm():
    return ChatOllama(model="llama3", temperature=0.1)


def validate_config() -> Tuple[bool, str]:
    if not os.getenv("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY is missing. Add it in .env before using the app."
    return True, "OpenAI configuration looks valid."


def get_pdf_pages(docs):
    pages = []
    for pdf in docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text("text") # Preserves layout much better
            if text.strip():
                pages.append({"source": pdf.name, "page": i + 1, "text": text})
    return pages


def get_chunks(pages: List[Dict[str, str]]) -> Tuple[List[str], List[Dict[str, str]]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunk_texts: List[str] = []
    chunk_metadata: List[Dict[str, str]] = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for c in chunks:
            chunk_texts.append(c)
            chunk_metadata.append({"source": page["source"], "page": page["page"]})
    return chunk_texts, chunk_metadata


def get_vectorstore(chunks, metadata):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadata)

def get_conversation_chain(retriever: ParentDocumentRetriever) -> ConversationalRetrievalChain:
    llm = get_openai_llm()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever, # Pass the ParentDocumentRetriever directly here
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory,
    )


def handle_question(question: str) -> None:
    try:
        response = st.session_state.conversation({"question": question})
    except Exception as exc:
        st.error(f"Q&A request failed: {exc}")
        return

    st.session_state.chat_history = response["chat_history"]
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def run_feature_regex(raw_text: str) -> Dict[str, List[str]]:
    extracted: Dict[str, List[str]] = {}
    for feature, patterns in INSURANCE_FEATURE_PATTERNS.items():
        matches: List[str] = []
        for pattern in patterns:
            hits = re.findall(pattern, raw_text, flags=re.IGNORECASE)
            for hit in hits:
                cleaned = " ".join(hit.split())
                if cleaned and cleaned not in matches:
                    matches.append(cleaned)
                if len(matches) >= 5:
                    break
            if len(matches) >= 5:
                break
        extracted[feature] = matches
    return extracted


def format_context_with_citations(docs) -> str:
    lines: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        lines.append(f"[Chunk {idx} | {source} p.{page}]\n{doc.page_content}")
    return "\n\n".join(lines)


def summarize_policy_with_rag(vectorstore: FAISS) -> str:
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 30})
    query = (
        "Extract all policy inclusions, exclusions, room rent limits, ICU limits, zero depreciation "
        "details, waiting periods, co-pay, deductibles, and sub-limits with citations."
    )
    docs = retriever.get_relevant_documents(query)
    context = format_context_with_citations(docs)

    llm = get_openai_llm()
    messages = [
        SystemMessage(
            content=(
                "You are an insurance policy analyst. Use only the provided context and never hallucinate. "
                "If information is missing, write 'Not found in document'."
            )
        ),
        HumanMessage(
            content=(
                "Analyze the insurance policy context below and return a concise, structured summary.\n\n"
                "Required headings:\n"
                "1) Inclusions\n"
                "2) Exclusions\n"
                "3) Room Rent and ICU Limits\n"
                "4) Zero Depreciation / Add-ons\n"
                "5) Waiting Periods\n"
                "6) Co-pay and Deductibles\n"
                "7) Sub-limits / Caps\n"
                "8) Important Conditions\n"
                "9) Missing or Unclear Items\n\n"
                "For every key point include citation in this format: [source p.X].\n\n"
                f"Context:\n{context}"
            )
        ),
    ]
    return llm.invoke(messages).content


def initialize_state() -> None:
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Insurance Policy RAG Assistant", page_icon=":shield:")
    st.write(css, unsafe_allow_html=True)
    initialize_state()

    st.title("Insurance Policy RAG Assistant")
    st.caption("Upload policy PDFs and extract inclusions, exclusions, room rent, zero dep, and more.")

    ok, config_msg = validate_config()
    if ok:
        st.success(config_msg)
    else:
        st.error(config_msg)

    with st.sidebar:
        st.subheader("Policy Documents")
        docs = st.file_uploader(
            "Upload one or more insurance policy PDFs and click Process",
            accept_multiple_files=True,
            type=["pdf"],
        )
        process_clicked = st.button("Process")

        if process_clicked:
            if not docs:
                st.warning("Please upload at least one PDF.")
            elif not ok:
                st.error("Fix OPENAI_API_KEY in .env first.")
            else:
                # Inside your main() under `if process_clicked:`
                with st.spinner("Extracting text, creating chunks, and indexing..."):
                    pages = get_pdf_pages(docs)
                    st.session_state.raw_text = "\n\n".join(page["text"] for page in pages)
                    
                    # Use our new Parent Document Retriever
                    st.session_state.retriever = get_parent_document_retriever(pages)
                    st.session_state.conversation = get_conversation_chain(st.session_state.retriever)
                    
                st.success("Documents processed successfully.")

    qa_tab, extract_tab = st.tabs(["Ask Questions", "Extract Policy Features"])

    with qa_tab:
        question = st.text_input("Ask anything from the policy documents")
        ask_clicked = st.button("Ask")
        if ask_clicked:
            if not st.session_state.conversation:
                st.warning("Process documents first.")
            elif not question.strip():
                st.warning("Please enter a question.")
            else:
                handle_question(question)

    with extract_tab:
        extract_clicked = st.button("Run Policy Extraction")
        if extract_clicked:
            if not st.session_state.vectorstore:
                st.warning("Process documents first.")
            else:
                with st.spinner("Extracting key policy features..."):
                    try:
                        rag_summary = summarize_policy_with_rag(st.session_state.vectorstore)
                    except Exception as exc:
                        st.error(f"Policy extraction failed: {exc}")
                        return
                    regex_features = run_feature_regex(st.session_state.raw_text)

                st.markdown("### RAG Policy Summary")
                st.markdown(rag_summary)

                st.markdown("### Pattern-based Feature Hits (Fast Validation Layer)")
                for feature, values in regex_features.items():
                    st.markdown(f"**{feature}**")
                    if values:
                        for val in values:
                            st.write(f"- {val}")
                    else:
                        st.write("- Not detected")

def get_parent_document_retriever(pages: list[dict]) -> ParentDocumentRetriever:
    docs = [
        Document(
            page_content=page["text"], 
            metadata={"source": page["source"], "page": page["page"]}
        )
        for page in pages
    ]

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=250)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # --- OPTIMIZATION: Use free, local HuggingFace embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # MiniLM creates 384-dimensional vectors (OpenAI used 1536)
    # We MUST change this size or FAISS will crash!
    embedding_size = 384 
    index = faiss.IndexFlatL2(embedding_size)
    
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 5} 
    )

    retriever.add_documents(docs)
    return retriever


if __name__ == "__main__":
    main()