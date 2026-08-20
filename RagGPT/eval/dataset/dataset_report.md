# Eval dataset report

Total items: 220

| category | count | target |
|---|---|---|
| rag_qa | 80 | 80 |
| coding | 40 | 40 |
| debugging | 40 | 40 |
| small_talk | 20 | 20 |
| off_topic | 20 | 20 |
| multi_turn | 20 | 20 |

## Synthesis rejection stats

- coding_reference_check_failed: 295
- coding_rejected: 92
- debugging_check_failed: 94
- debugging_rejected: 21
- rag_qa_rejected: 4

## Source-file spread (doc-grounded categories)

- tutorials/language/coding_primer.md: 16
- reference/language/osp.md: 14
- reference/language/foundation.md: 13
- reference/plugins/byllm.md: 12
- reference/language/functions-objects.md: 12
- reference/language/walker-responses.md: 12
- tutorials/language/osp.md: 11
- reference/persistence.md: 10
- reference/language/python-integration.md: 9
- reference/testing.md: 8
- reference/language/library-mode.md: 7
- reference/language/concurrency.md: 7
- tutorials/ai/quickstart.md: 7
- reference/language/primitives.md: 6
- tutorials/language/basics.md: 6
- tutorials/ai/agentic.md: 5
- reference/language/advanced.md: 4
- tutorials/language/debugging.md: 4
- internals/interop.md: 3
- reference/language/access-modifiers.md: 3
- tutorials/ai/structured-outputs.md: 3
- tutorials/ai/multimodal.md: 2
- internals/jac_import_patterns.md: 2
- reference/code-organization.md: 2
- internals/compiler_architecture.md: 1
- internals/abstractions.md: 1

## One example per category

**rag_qa** (rag_qa-001): What does the 'jac jac2py' command do in Jac programming?

**coding** (coding-001): Write a Jac program that defines a node type called 'Person' with a 'name' field, creates three Person nodes with different names, connects them all to the root node, and then prints the names of all 

**debugging** (debugging-001): My Jac program is broken and I can't figure out why: ```jac with entry {     print("Hello, Jac!") } ```

**small_talk** (small_talk-001): Hey there!

**off_topic** (off_topic-001): How do I create a virtual environment in Python?

**multi_turn** (multi_turn-001): Write a Jac walker that searches a given list of numbers and prints the first negative number it finds, then exits the loop immediately.
