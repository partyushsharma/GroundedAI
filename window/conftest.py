import sys
from pathlib import Path

# Add workspace directory to sys.path so scripts and modules can be imported
sys.path.insert(0, str(Path(__file__).parent.resolve()))
