# Recorded router replies

One file per case, in the exact format `scripts/live_check.py --trace` writes: one JSON object
per model call, with `request` (the prompts and the JSON schema sent as the output format) and
`reply` (the raw text the model returned). A trace file from a real run can be copied into this
folder unchanged and `tests/acceptance/test_live_contract.py` will pick it up with no other
change.

`question` and `expected` are added here so the end-to-end test can drive the controller with
the recorded reply and check the value it produces. A record with `expected: []` is validated
for shape only.

The files in this folder were written by hand from the schema the router sends, because the
container this branch was built in has no Ollama. They cover each of the nine recipes, the two
delegated routes and `none`, in the shape a grammar-constrained model produces: compact JSON,
numbers sometimes as JSON numbers, a quantity occasionally written whole ("20 m/s"), and no
`statement`. `formalize-no-statement.jsonl` is the reply qwen3:8b actually produced on
2026-10-07, which the old validator rejected for a field the request never required.

Replace or add to them with real traces as runs produce them. The test asserts, for every
record:

1. the reply satisfies the schema the request sent, so it is a reply a real model could give;
2. the validator accepts it, so the request and the validator cannot disagree again;
3. a recipe route's slots plan, and the controller answers the question in one model call.
