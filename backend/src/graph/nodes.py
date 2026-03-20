import json
import os
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.messages import SystemMessage, HumanMessage

from backend.src.graph.state import VideoAuditState, ComplianceIssue
from backend.src.services.video_indexer import VideoIndexerService

logger = logging.getLogger("brand-guardian")
logging.basicConfig(level=logging.INFO)


# ============================================================
# PYDANTIC SCHEMAS — Replaces manual JSON prompt instructions
# ============================================================

class ComplianceViolation(BaseModel):
    """A single compliance violation found in the video."""
    category: str = Field(
        description="Category of violation e.g. 'Claim Validation', 'Endorsement Disclosure'"
    )
    severity: str = Field(
        description="Either CRITICAL or WARNING"
    )
    description: str = Field(
        description="Detailed explanation of the violation found"
    )

class AuditResult(BaseModel):
    """Complete audit result returned by the compliance auditor."""
    compliance_results: List[ComplianceViolation] = Field(
        description="List of all compliance violations found. Empty list if none."
    )
    status: str = Field(
        description="Overall audit status. Must be exactly PASS or FAIL"
    )
    final_report: str = Field(
        description="Professional summary of all findings for the compliance report"
    )


# ============================================================
# NODE 1: THE INDEXER (unchanged)
# ============================================================
def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    """Downloads YouTube video, uploads to Azure VI, and extracts insights."""
    video_url = state.get("video_url")
    video_id_input = state.get("video_id", "vid_demo")
    
    logger.info(f"--- [Node: Indexer] Processing: {video_url} ---")
    local_filename = "temp_audit_video.mp4"
    
    try:
        vi_service = VideoIndexerService()
        
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception("Please provide a valid YouTube URL for this test.")

        azure_video_id = vi_service.upload_video(local_path, video_name=video_id_input)
        logger.info(f"Upload Success. Azure ID: {azure_video_id}")
        
        if os.path.exists(local_path):
            os.remove(local_path)

        raw_insights = vi_service.wait_for_processing(azure_video_id)
        clean_data = vi_service.extract_data(raw_insights)
        
        logger.info("--- [Node: Indexer] Extraction Complete ---")
        return clean_data

    except Exception as e:
        logger.error(f"Video Indexer Failed: {e}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL",
            "transcript": "",
            "ocr_text": []
        }


# ============================================================
# NODE 2: THE COMPLIANCE AUDITOR — Now with Pydantic
# ============================================================
def audit_content_node(state: VideoAuditState) -> Dict[str, Any]:
    """Performs RAG audit using structured Pydantic output."""
    logger.info("--- [Node: Auditor] Querying Knowledge Base & LLM ---")
    
    transcript = state.get("transcript", "")
    
    if not transcript:
        logger.warning("No transcript available. Skipping Audit.")
        return {
            "final_status": "FAIL",
            "final_report": "Audit skipped because video processing failed (No Transcript)."
        }

    # --- Initialize LLM ---
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0
    )

    # --- Bind Pydantic schema to LLM ---
    # This replaces the manual JSON instructions in the prompt
    # LangChain forces the LLM to return exactly this structure
    structured_llm = llm.with_structured_output(AuditResult)

    # --- Initialize Embeddings & Vector Store ---
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
        openai_api_version="2024-12-01-preview",
    )

    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query
    )
    
    # --- RAG Retrieval ---
    ocr_text = state.get("ocr_text", [])
    query_text = f"{transcript} {' '.join(ocr_text)}"
    docs = vector_store.similarity_search(query_text, k=3)
    retrieved_rules = "\n\n".join([doc.page_content for doc in docs])
    
    # --- Cleaner Prompt — No manual JSON schema needed ---
    system_prompt = f"""
    You are a Senior Brand Compliance Auditor.
    
    OFFICIAL REGULATORY RULES:
    {retrieved_rules}
    
    INSTRUCTIONS:
    1. Analyze the Transcript and OCR text provided.
    2. Identify ALL violations of the rules above.
    3. For each violation specify the category, severity (CRITICAL or WARNING), 
       and a detailed description.
    4. Set status to PASS only if zero violations are found, otherwise FAIL.
    5. Write a professional final_report summarizing your findings.
    """

    user_message = f"""
    VIDEO METADATA: {state.get('video_metadata', {})}
    TRANSCRIPT: {transcript}
    ON-SCREEN TEXT (OCR): {ocr_text}
    """

    try:
        # --- Invoke structured LLM ---
        # Returns AuditResult Pydantic object directly — no JSON parsing needed
        audit_result: AuditResult = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])

        logger.info(f"Audit complete. Status: {audit_result.status}")
        logger.info(f"Violations found: {len(audit_result.compliance_results)}")

        # --- Convert Pydantic objects to dicts for the graph state ---
        return {
            "compliance_results": [v.model_dump() for v in audit_result.compliance_results],
            "final_status": audit_result.status,
            "final_report": audit_result.final_report
        }

    except Exception as e:
        logger.error(f"System Error in Auditor Node: {str(e)}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL",
            "final_report": f"Audit failed due to system error: {str(e)}"
        }