# Sample Documents

Test documents for quick local testing of the RAG pipeline.

## Files

| File | Description | Key facts for testing |
|---|---|---|
| `sample.txt` | NovaTech Solutions company profile | Founded 2019, $45M Series B, FlowAI platform, 300+ clients, $28M ARR |
| `research_summary.txt` | Summary of the RAG paper (Lewis et al., 2020) | DPR + BART architecture, 44.5 EM on Natural Questions, 3,200+ citations |

## Usage

These files are automatically used by `make sanity`. You can also upload them
through the Streamlit UI for interactive testing.

To test cross-document retrieval, upload both files and ask:
- "What is NovaTech's revenue?" (should cite `sample.txt`)
- "What is the RAG architecture?" (should cite `research_summary.txt`)
- "What are the limitations of RAG?" (should cite `research_summary.txt`)
