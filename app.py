"""
Cabelte Chat — agente RAG sobre o site cabelte.pt
Uso: streamlit run app.py
"""
import streamlit as st
import chromadb
from google import genai
from google.genai import types

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.0-flash"
TOP_K = 5

st.set_page_config(
    page_title="Cabelte Chat",
    page_icon="🔵",
    layout="centered",
)

st.markdown("""
<style>
[data-testid="stChatMessage"] { border-radius: 10px; }
.source-tag { font-size: 0.75rem; color: #888; }
</style>
""", unsafe_allow_html=True)

st.title("🔵 Cabelte Chat")
st.caption("Assistente sobre produtos, empresa e informações de cabelte.pt · powered by Gemini")

try:
    gemini = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("**GEMINI_API_KEY não configurada.** Adiciona a chave em `.streamlit/secrets.toml`.")
    st.stop()

SYSTEM_PROMPT = """És um assistente especializado na Cabelte — empresa portuguesa fabricante de cabos eléctricos e de telecomunicações.
Responde sempre em português europeu, de forma clara e directa.
Baseia as tuas respostas exclusivamente no contexto fornecido.
Se a informação pedida não constar no contexto, diz claramente que não tens essa informação disponível no site.
Não inventes factos.

CONTEXTO DO SITE:
{context}"""


@st.cache_resource(show_spinner="A carregar base de conhecimento...")
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("cabelte")


def embed_query(text: str) -> list[float]:
    result = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def retrieve(query: str, collection, k: int = TOP_K):
    q_emb = embed_query(query)
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    return results["documents"][0], results["metadatas"][0]


def build_context(docs: list[str], metas: list[dict]) -> str:
    parts = []
    for doc, meta in zip(docs, metas):
        parts.append(f"[{meta['title']}]\n{doc}")
    return "\n\n---\n\n".join(parts)


try:
    collection = load_collection()
except Exception as e:
    st.error(
        f"**Base de conhecimento não encontrada.**\n\n"
        f"Corre primeiro:\n```\npython ingest.py\n```\n\nErro: {e}"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = {}

col1, col2 = st.columns([5, 1])
with col2:
    if st.button("🗑️ Limpar", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.rerun()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i in st.session_state.sources:
            srcs = st.session_state.sources[i]
            seen = set()
            unique_srcs = [s for s in srcs if not (s["url"] in seen or seen.add(s["url"]))]
            with st.expander("📚 Fontes", expanded=False):
                for s in unique_srcs:
                    st.markdown(f'<span class="source-tag">• <a href="{s["url"]}" target="_blank">{s["title"]}</a></span>', unsafe_allow_html=True)

if prompt := st.chat_input("Pergunta sobre a Cabelte..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("A pesquisar..."):
            docs, metas = retrieve(prompt, collection)
            context = build_context(docs, metas)

        # Histórico no formato Gemini
        history = []
        for m in st.session_state.messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        response = gemini.models.generate_content_stream(
            model=CHAT_MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT.format(context=context),
            ),
        )

        full_response = ""
        placeholder = st.empty()
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    msg_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.sources[msg_index] = metas

    seen = set()
    unique_srcs = [s for s in metas if not (s["url"] in seen or seen.add(s["url"]))]
    with st.expander("📚 Fontes", expanded=False):
        for s in unique_srcs:
            st.markdown(f'<span class="source-tag">• <a href="{s["url"]}" target="_blank">{s["title"]}</a></span>', unsafe_allow_html=True)
