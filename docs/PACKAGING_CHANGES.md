# Packaging changes

The selected Ridge model and recorded benchmark results are unchanged. This refresh adds recruiter-facing documentation and metric visualizations. The development dependency set now explicitly includes httpx, required by the API contract tests.
PyTorch is an optional `deep-learning` extra; the full TCN training workflow installs it, while Ridge/API tests do not need it.

A source check also found that the previous-day seasonal-naive helper returned an empty array when period and horizon were both 24. Its slice now uses positive start/end indices, with a regression test for the 24-hour case. The saved headline comparison uses the 168-hour weekly baseline and is unaffected.
