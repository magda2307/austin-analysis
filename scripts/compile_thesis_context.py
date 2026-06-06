"""Script to compile all important markdown files into one massive context file for an LLM."""
from pathlib import Path

def compile_context():
    root = Path(__file__).resolve().parents[1]
    
    files_to_include = [
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/METHODOLOGY.md",
        "docs/RESULTS.md",
        "docs/ROADMAP.md",
        "docs/old/thesis_technical_guide.md",
        "docs/internal/ml_code_review.md",
        "docs/target_definitions.md",
    ]
    
    # Add all reports/summary markdown files
    reports_dir = root / "reports" / "summary"
    if reports_dir.exists():
        for md_file in sorted(reports_dir.glob("*.md")):
            files_to_include.append(f"reports/summary/{md_file.name}")
            
    out_path = root / "THESIS_CONTEXT_FOR_LLM.md"
    
    with open(out_path, "w", encoding="utf-8") as outfile:
        outfile.write("# THESIS CONTEXT COMPILATION\n\n")
        outfile.write("This file is a compiled megadocument of all architectural, methodological, and result summaries. It is designed to be passed to an LLM to provide full context on the Austin Animal Center Adoption ML project.\n\n")
        
        for file_path in files_to_include:
            full_path = root / file_path
            if full_path.exists():
                outfile.write(f"\n\n{'='*80}\n")
                outfile.write(f"FILE: {file_path}\n")
                outfile.write(f"{'='*80}\n\n")
                try:
                    content = full_path.read_text(encoding="utf-8")
                    outfile.write(content)
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]\n")
            else:
                outfile.write(f"\n\n[Warning: File not found - {file_path}]\n")
                
    print(f"Compiled {len(files_to_include)} files into {out_path.name}")
    print(f"File size: {out_path.stat().st_size / 1024:.2f} KB")

if __name__ == "__main__":
    compile_context()
