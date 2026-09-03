import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest.models import ChunkMeta
from datetime import date

def test_chunk_meta():  
    meta = ChunkMeta(
        pdf_name="test.pdf",
        page_number=3,
        circular_number="RBI/2023/123",
        issue_date=date(2023, 8, 15),
        tenant_id="client_a",
        acl={"roles": ["admin", "viewer"]}
    )
    
    print("✅ Model created successfully")
    print(meta.model_dump_json(indent=2))
    
    # Check that placeholders are stored
    assert meta.tenant_id == "client_a"
    assert "admin" in meta.acl["roles"]
    print("✅ Placeholders (tenant_id, acl) are present and working")

if __name__ == "__main__":
    test_chunk_meta()