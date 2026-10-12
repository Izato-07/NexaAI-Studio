
import ast
import json
import operator
import os
import tempfile
from datetime import datetime

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader

st.set_page_config(
    page_title="NexaAI Studio",
    page_icon="🤖",
    layout="wide",
)

CHAT_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

st.title("🤖 NexaAI Studio")
st.caption("Chat with AI, explore PDFs, and use intelligent tools.")

# ---------------- Safe Calculator ----------------

def calculate(expression):
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            if isinstance(node.op, ast.Pow):
                if abs(evaluate(node.right)) > 10:
                    raise ValueError("Exponent is too large.")

            result = operations[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )

            if isinstance(result, (int, float)) and abs(result) > 1e100:
                raise ValueError("Result is too large.")

            return result

        if isinstance(node, ast.UnaryOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.operand))

        raise ValueError("Unsupported mathematical expression.")

    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    return evaluate(ast.parse(expression, mode="eval").body)


def current_datetime():
    return datetime.now().astimezone().strftime(
        "%A, %d %B %Y, %I:%M:%S %p %Z"
    )


# ---------------- PDF Processing and RAG ----------------

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_number, text))

    return pages


def split_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


@st.cache_resource
def get_vector_db():
    return chromadb.Client()


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBED_MODEL)


def process_pdf(uploaded_file):
    pages = extract_pdf(uploaded_file)

    if not pages:
        raise ValueError(
            "No readable text was found. The PDF may be scanned or image-only."
        )

    collection_name = "nexa_" + str(abs(hash(uploaded_file.name)))[:16]

    db = get_vector_db()

    try:
        db.delete_collection(collection_name)
    except Exception:
        pass

    collection = db.create_collection(collection_name)

    documents = []
    ids = []
    metadatas = []

    for page_number, page_text in pages:
        for chunk_number, chunk in enumerate(split_text(page_text)):
            documents.append(chunk)
            ids.append(f"p{page_number}_c{chunk_number}")
            metadatas.append({
                "page": page_number,
                "filename": uploaded_file.name,
            })

    batch_size = 32

    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        embeddings = get_embedding_model().encode(
            batch,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=ids[start:start + batch_size],
            documents=batch,
            embeddings=embeddings,
            metadatas=metadatas[start:start + batch_size],
        )

    return collection, len(documents)


def retrieve_context(collection, question):
    embedding = get_embedding_model().encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(4, collection.count()),
    )

    passages = results.get("documents", [[]])[0]
    metadata = results.get("metadatas", [[]])[0]

    context = []

    for passage, info in zip(passages, metadata):
        context.append(
            f"[Source: {info['filename']}, page {info['page']}]\n{passage}"
        )

    return "\n\n".join(context)


# ---------------- Groq Tool Calling ----------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate arithmetic expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "Get the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def run_agent(client, messages, pdf_collection=None):
    conversation = [
        {
            "role": "system",
            "content": (
                "You are NexaAI Studio, a helpful AI assistant. "
                "Use tools for calculations and current date/time questions. "
                "If the user asks about an uploaded PDF, use the supplied "
                "document context and cite its filename and page number. "
                "Do not invent facts from a PDF. If context is insufficient, "
                "say so clearly."
            ),
        },
        *messages,
    ]

    logs = []

    latest_user_message = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )

    if pdf_collection is not None:
        try:
            context = retrieve_context(pdf_collection, latest_user_message)
            conversation[0]["content"] += (
                "\n\nUploaded document context:\n" + context
            )
        except Exception as error:
            logs.append(f"PDF retrieval failed: {error}")

    for _ in range(5):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=conversation,
            tools=TOOLS,
            tool_choice="auto",
            max_completion_tokens=1500,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or "No response was returned.", logs

        conversation.append(message)

        for call in message.tool_calls:
            name = call.function.name
            logs.append(f"Selected tool: {name}")

            try:
                args = json.loads(call.function.arguments or "{}")

                if name == "calculator":
                    result = calculate(args["expression"])
                elif name == "current_datetime":
                    result = current_datetime()
                else:
                    raise ValueError("Unknown tool.")

                result = str(result)
                logs.append(f"Tool result: {result}")

            except Exception as error:
                result = f"Tool error: {error}"
                logs.append(result)

            conversation.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return "Maximum tool steps reached. Please simplify your request.", logs


# ---------------- Interface ----------------

with st.sidebar:
    st.header("⚙️ Workspace")
    st.write("💬 AI Chat")
    st.write("📄 PDF Question Answering")
    st.write("🧮 Calculator Tool")
    st.write("🕒 Date & Time Tool")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.logs = {}
        st.rerun()

    st.divider()
    st.caption("Powered by Groq, Sentence Transformers, and ChromaDB.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = {}

if "pdf_collection" not in st.session_state:
    st.session_state.pdf_collection = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

if not api_key:
    st.warning(
        "Groq API key missing. Create .streamlit/secrets.toml "
        "and set GROQ_API_KEY = \"your_key_here\"."
    )
    st.stop()

client = Groq(api_key=api_key)

st.subheader("📄 Chat with your documents")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    help="Upload a text-based PDF to ask questions about its content.",
)

if uploaded_file:
    if st.button("Process PDF", type="primary"):
        try:
            with st.spinner("Extracting text and creating embeddings..."):
                collection, chunk_count = process_pdf(uploaded_file)

            st.session_state.pdf_collection = collection
            st.session_state.pdf_name = uploaded_file.name

            st.success(
                f"Processed {uploaded_file.name}: {chunk_count} text chunks."
            )

        except Exception as error:
            st.error(f"PDF processing failed: {error}")

if st.session_state.pdf_name:
    st.info(f"Active document: {st.session_state.pdf_name}")

st.divider()
st.subheader("💬 Chat with NexaAI")

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and index in st.session_state.logs:
            with st.expander("🔍 Agent action log"):
                for item in st.session_state.logs[index]:
                    st.write(item)

prompt = st.chat_input("Ask a question, calculate something, or query your PDF...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking, retrieving, and using tools..."):
            try:
                answer, logs = run_agent(
                    client,
                    st.session_state.messages,
                    st.session_state.pdf_collection,
                )

                st.markdown(answer)

                index = len(st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })
                st.session_state.logs[index] = logs

                with st.expander("🔍 Agent action log", expanded=True):
                    if logs:
                        for item in logs:
                            st.write(item)
                    else:
                        st.write("No tools were needed for this response.")

            except Exception as error:
                st.error(f"Something went wrong: {error}")