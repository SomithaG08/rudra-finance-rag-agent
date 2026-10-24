"""Day-1: ask one question on the command line (retrieve 10 chunks + answer)."""
from _setup import generate_answer, retrieve

question = input("Enter your question: ")
print("\nRetrieving relevant information...")
results = retrieve(question, k=10)
print("\nGenerating answer...")
print("\n===== ANSWER =====\n" + generate_answer(question, results) + "\n==================")
