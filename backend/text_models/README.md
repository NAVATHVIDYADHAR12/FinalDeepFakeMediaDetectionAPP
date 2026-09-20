# AI-text classifier

This directory contains OmniGuard's dependency-free runtime export of a
TF-IDF logistic-regression classifier trained on
[`rasbt/human-vs-ai-50k`](https://huggingface.co/datasets/rasbt/human-vs-ai-50k).
The dataset and model recipe are Apache-2.0 licensed; the license is included
in this directory.

Held-out test results for this exact export are recorded in `metadata.json`.
The current model uses the 200,000 most frequent one- and two-word features so
it fits within Render's 512 MB free instance. Runtime inference needs only
NumPy; it never loads a pickle or executes downloaded model code.

Rebuild it with `tools/train_text_detector.py`. Training requires
scikit-learn and PyArrow, but those packages are deliberately not production
dependencies.

The output is probabilistic. Generator, language, domain, editing and short or
partly assisted samples can all change performance. It must not be used alone
as proof of authorship or academic misconduct.
