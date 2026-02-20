import base64
import io
from typing import List, Tuple

import streamlit as st
from PIL import Image

# --- OCI SDKs (Document Understanding) ---
import oci
from oci.ai_document import AIServiceDocumentClient
from oci.ai_document.models import (
    AnalyzeDocumentDetails,
    InlineDocumentDetails,
    DocumentTextExtractionFeature,
    DocumentTableExtractionFeature,
)

# --- OCI GenAI via OpenAI-compatible client ---
import httpx
from openai import OpenAI
from oci_openai import OciUserPrincipalAuth


# ---------------- Page Setup ----------------
st.set_page_config(page_title="OCI DU + Grok (xai.grok-code-fast-1)", page_icon="🤖", layout="wide")
st.title("📄 Extract with OCI Document Understanding → 💬 Ask via xai.grok-code-fast-1")


# ---------------- Helpers ----------------
def load_oci_config():
    profile = st.secrets.get("oci", {}).get("profile", "DEFAULT")
    try:
        return oci.config.from_file(profile_name=profile)
    except Exception as e:
        st.error(f"OCI config error for profile '{profile}': {e}")
        st.stop()


def get_doc_ai_client(cfg):
    ep = st.secrets.get("oci", {}).get("doc_ai_endpoint", "").strip()
    if ep:
        return AIServiceDocumentClient(cfg, service_endpoint=ep)
    return AIServiceDocumentClient(cfg)


def oci_du_extract_text_per_page(file_bytes: bytes, page_ranges: List[str] | None = None) -> Tuple[str, List[str]]:
    """
    Synchronous DU analyze_document:
      - InlineDocumentDetails (base64 data)
      - Text + Table extraction
    Returns (full_text, per_page_texts)
    """
    cfg = load_oci_config()
    client = get_doc_ai_client(cfg)

    inline = InlineDocumentDetails(
        data=base64.b64encode(file_bytes).decode("utf-8"),
        page_range=page_ranges or None,
    )

    details = AnalyzeDocumentDetails(
        features=[DocumentTextExtractionFeature(), DocumentTableExtractionFeature()],
        document=inline,
    )

    resp = client.analyze_document(details)
    result = resp.data

    per_page = []
    for pg in getattr(result, "pages", []) or []:
        parts = []
        for ln in getattr(pg, "lines", []) or []:
            txt = getattr(ln, "text", "")
            if txt:
                parts.append(txt)
        for tb in getattr(pg, "tables", []) or []:
            for row in getattr(tb, "rows", []) or []:
                cells = [getattr(c, "text", "") for c in getattr(row, "cells", []) or []]
                if any(cells):
                    parts.append("\t".join(cells))
        per_page.append("\n".join(parts).strip())

    full_text = "\n\n".join([t for t in per_page if t])
    return full_text, per_page


def build_openai_compatible_client() -> OpenAI:
    """
    OCI OpenAI compatibility: use OpenAI client with OCI signer & CompartmentId header.
    """
    base_url = st.secrets.get("oci", {}).get("genai_endpoint", "").rstrip("/")
    if not base_url:
        st.error("Missing [oci].genai_endpoint in secrets.")
        st.stop()

    compartment_id = st.secrets.get("oci", {}).get("compartment_id", "")
    if not compartment_id:
        st.error("Missing [oci].compartment_id in secrets.")
        st.stop()

    profile = st.secrets.get("oci", {}).get("profile", "DEFAULT")

    return OpenAI(
        api_key="OCI",  # placeholder; actual auth is via OciUserPrincipalAuth
        base_url=base_url,
        http_client=httpx.Client(
            auth=OciUserPrincipalAuth(profile_name=profile),
            headers={"CompartmentId": compartment_id},
            timeout=60.0,
        ),
    )


def answer_with_grok(question: str, context_text: str, temperature: float = 0.0, max_tokens: int = 600) -> str:
    client = build_openai_compatible_client()
    model_id = st.secrets.get("oci", {}).get("genai_model_id", "xai.grok-4-fast-non-reasoning")

    # Keep the prompt tight; model is code-oriented but can handle grounded Q&A.
    system = (
        "You are a precise assistant. Answer ONLY using the provided context. "
        "If the answer is not present, say you don't know."
    )
    # crude length guard to avoid oversize prompts; replace with tokenizer-aware approach as needed
    context_cap = 40_000
    ctx = context_text[:context_cap]

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\n\nGive a concise, accurate answer."},
    ]

    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Options")
multi = st.sidebar.checkbox("Allow multiple uploads", value=False)
show_raw = st.sidebar.checkbox("Show extracted raw text", value=False)
temperature = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.0, 0.1)
max_tokens = st.sidebar.slider("LLM max tokens", 100, 2000, 600, 50)

# ---------------- Uploader ----------------
accepted_types = ["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp", "docx"]
files = st.file_uploader(
    "Drag & drop or browse files (PDF, Images, DOCX*)",
    type=accepted_types,
    accept_multiple_files=multi,
)

if files and not isinstance(files, list):
    files = [files]

if not files:
    st.info("Upload a document to extract text with OCI DU and ask questions using xai.grok-code-fast-1.")
    st.stop()

# ---------------- Extract each file ----------------
corpus_sections = []
for f in files:
    file_name = f.name
    suffix = (file_name.split(".")[-1] or "").lower()
    file_bytes = f.getvalue()

    st.subheader(f"📁 {file_name}")

    # Preview
    if suffix == "pdf":
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        st.markdown(
            f"""
            <div style="border:1px solid #ddd;border-radius:8px;overflow:hidden">
              <iframe src="data:application/pdf;base64,{b64}" width="100%" height="600px"></iframe>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif suffix in ["png", "jpg", "jpeg", "tiff", "bmp", "webp"]:
        st.image(Image.open(io.BytesIO(file_bytes)), caption=file_name, use_container_width=True)
    else:
        st.caption("Preview not available; continuing to extraction…")

    # Extract with OCI DU
    with st.spinner("Extracting text with OCI Document Understanding…"):
        try:
            full_text, per_page = oci_du_extract_text_per_page(file_bytes)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            continue

    if show_raw and full_text.strip():
        with st.expander("🧾 Extracted raw text (first ~10k chars)"):
            st.text(full_text[:10_000] + ("…" if len(full_text) > 10_000 else ""))

    if full_text.strip():
        corpus_sections.append((file_name, full_text))

# ---------------- Ask Grok ----------------
st.markdown("---")
st.markdown("### 💬 Ask questions about your uploaded document(s)")
question = st.text_input("Your question", placeholder="e.g., What is the total amount and due date?")
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not corpus_sections:
        st.warning("No extracted text available.")
    else:
        # Simple concatenation; for scale/quality add retrieval based on embeddings.
        context_blocks = []
        for fname, text in corpus_sections:
            context_blocks.append(f"### Document: {fname}\n{text}")
        context_text = "\n\n".join(context_blocks)

        with st.spinner("Thinking with xai.grok-code-fast-1…"):
            try:
                answer = answer_with_grok(question, context_text, temperature=temperature, max_tokens=max_tokens)
                st.markdown("#### 🧠 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
else:
    st.caption("Tip: After uploading, ask a question. The LLM will ONLY see the text extracted by OCI DU.")