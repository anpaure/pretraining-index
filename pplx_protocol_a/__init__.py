"""Protocol A: 1-Call Top-20 Direct OpenAI Responses Perplexity (BPB) Harness.

Implements the preregistered, archive-hardened measurement stack for closed-source
APIs exposing output top-20 logprobs without prompt logprobs or logit_bias
(e.g., OpenAI `/v1/responses` on `gpt-5.6-sol`), together with an open-model
`/v1/completions` (`echo=true`) teacher-forced proxy & ground-truth validation gate.
"""

__version__ = "0.1.0"
