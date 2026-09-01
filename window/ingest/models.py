# src/ingest/models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date

class ChunkMeta(BaseModel):
    """
    Metadata attached to every text chunk.
    
    All fields are optional except the core source identifiers,
    but we use defaults to keep construction flexible.
    """
    
    # ---- Source identification ----
    pdf_name: str = Field(..., description="Filename of the source PDF")
    page_number: int = Field(..., description="Page number where the chunk originates (0-indexed)")
    
    # ---- Contextual metadata ----
    section_header: Optional[str] = Field(None, description="Heading or subheading of the section")
    circular_number: Optional[str] = Field(None, description="RBI circular / Master Direction number, e.g., 'RBI/2023-24/123'")
    issue_date: Optional[date] = Field(None, description="Date when the circular was issued")
    is_superseded: bool = Field(False, description="Whether this regulation has been superseded or withdrawn")
    superseded_by: Optional[str] = Field(None, description="Circular number that supersedes this one, if any")
    
    # ---- Indexing & chunking details ----
    chunk_index: Optional[int] = Field(None, description="Position of this chunk within the original document")
    total_chunks_in_doc: Optional[int] = Field(None, description="Total number of chunks this document was split into")
    
    # ---- Project 2 placeholders (multi-tenancy & permissions) ----
    tenant_id: Optional[str] = Field(
        None,
        description="Placeholder for multi-tenant isolation (Project 2). e.g., 'bank_of_baroda' or 'sbi'."
    )
    acl: Optional[Dict[str, Any]] = Field(
        None,
        description="Placeholder for Access Control List / permissions (Project 2). "
                    "Can store user roles, groups, or permission bits."
    )
    
    # ---- Extra metadata (flexible) ----
    extra: Optional[Dict[str, Any]] = Field(
        None,
        description="Catch-all for any additional custom fields without changing the schema."
    )

    class Config:
        # Allows you to create instances from JSON or dicts easily
        populate_by_name = True
        str_strip_whitespace = True