# RAG Evaluation

## Evaluation Set

The evaluation set contains 47 questions.

Categories:

- Factual: 12
- Cross-reference: 10
- Stress-test: 15
- Out-of-scope: 5
- Ambiguous: 5

## Priority Questions

Five priority questions were tested.

The system successfully answered the priority questions including:

- Rule of 72
- Time value of money
- Diversification
- NPV acceptance/rejection
- Effective annual interest rate

## Retrieval Improvement

The initial retriever used top-5 results.

Cross-reference questions showed that relevant information could exist
outside the top five results.

The retriever was therefore changed to retrieve the top 10 chunks.

This improved multi-section questions such as the American Airlines
cross-reference question and the DDM question.

## Out-of-Scope Testing

The system correctly refused questions outside the corpus, including
questions about weather, football, cooking, geography, and translation.

## Example Cross-Reference

The American Airlines question required information from two different
sections of the textbook.

The system successfully combined the retrieved information and provided
page citations.

## Conclusion

The evaluation confirmed that the RAG pipeline can retrieve relevant
financial information, answer factual questions, combine information
from multiple sections, and refuse unsupported out-of-scope questions.
