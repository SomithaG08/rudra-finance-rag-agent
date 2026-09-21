# Extraction Report

## Corpus

**Title:** Personal Finance & Financial Analysis for Students

**Source:** OpenStax — Principles of Finance

**Source files:** 1

The corpus was prepared from the OpenStax Principles of Finance PDF provided for the assignment.

## Extraction Method

PDF files are processed page by page using PyMuPDF.

The extraction process:

* Normalizes line endings
* Removes excessive spaces and tabs
* Joins words split across PDF line breaks
* Cleans unnecessary line breaks
* Removes empty lines

The extracted text is stored as UTF-8 JSON.

## Output Files

* `data/corpus.json` — complete extracted corpus
* `data/sample.json` — first 10 extracted entries

Each entry contains the source filename, page number, character count, and extracted text.

## Extraction Statistics

| Metric                              |    Result |
| ----------------------------------- | --------: |
| Source files                        |         1 |
| Total entries                       |       629 |
| Total characters                    | 1,614,510 |
| Average characters/entry            |  2,566.79 |
| Near-empty entries (<50 characters) |         3 |

## Near-Empty Entries

The following entries contain fewer than 50 characters:

* Page 1: 0 characters
* Page 2: 0 characters
* Page 14: 31 characters

These entries are retained in `corpus.json` as required.

Pages 1 and 2 produced no extractable text. Page 14 contains only a short OpenStax footer.

OCR was not performed because it is outside the scope of this assignment.

## Encoding

All generated JSON files use UTF-8 encoding.

## Extraction Status

The corpus extraction completed successfully.

Complete corpus: `data/corpus.json`

Sample: `data/sample.json`

