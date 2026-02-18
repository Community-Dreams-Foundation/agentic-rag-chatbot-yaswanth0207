.PHONY: sanity run docker-run clean eval

sanity:
	@mkdir -p artifacts
	@TOKENIZERS_PARALLELISM=false .venv/bin/python scripts/run_sanity.py

run:
	streamlit run app.py

docker-run:
	docker-compose up --build

clean:
	rm -rf chroma_db artifacts/sanity_output.json __pycache__
	find . -name "*.pyc" -delete
	echo "" > USER_MEMORY.md
	echo "" > COMPANY_MEMORY.md

eval:
	@TOKENIZERS_PARALLELISM=false python3 -c "\
	import sys, os; \
	os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false'); \
	sys.path.insert(0, '.'); \
	from dotenv import load_dotenv; load_dotenv(); \
	from rag.evaluator import RAGEvaluator; \
	e = RAGEvaluator(); \
	scores = e.evaluate('What does NovaTech do?', 'NovaTech builds AI logistics software.', ['NovaTech Solutions builds AI-powered logistics software.']); \
	print('RAGAS scores:', scores)"
