"""
Three-level output validator for LLM agent responses.

Validation pipeline:

  Level 1 -- Schema:    Validates JSON structure against a Pydantic model.
             PASS when the output parses and satisfies all field constraints.
             FAIL triggers an immediate ESCALATE decision.
             SKIP when no schema is provided (plain-text outputs).

  Level 2 -- Semantic completeness:
             Splits the task description into requirement sentences and checks
             whether the output covers each one via cosine similarity.
             score = covered_requirements / total_requirements.
             Low score -> RETRY (incomplete but not wrong).

  Level 3 -- Evidence grounding:
             Extracts all text from tool_result blocks in the conversation trace
             and checks whether each claim in the output is semantically
             supported by that evidence.
             score = grounded_claims / total_claims.
             Critically low score -> ESCALATE (hallucination risk).

Decision matrix:
  PASS      overall_confidence >= confidence_threshold (covered + grounded).
  RETRY     output is incomplete: semantic_score < semantic_threshold.
            The agent simply did not cover enough required topics -- retry to
            gather more. This is the safe failure: low coverage, no false claims.
  ESCALATE  schema failure, OR the output covers the required topics
            (semantic_score >= semantic_threshold) yet its claims are not
            grounded in evidence. This is the dangerous failure: it looks
            complete but is fabricated -- a human must review.

Why coverage separates RETRY from ESCALATE:
  Embedding similarity measures topical relevance, not factual agreement. A
  fabricated claim ("tokens expire after 15 minutes") is topically close to the
  evidence it contradicts ("JWT_EXPIRY = None"), so a confidently-wrong output
  scores HIGH on semantic coverage. High coverage + low grounding is therefore
  the signature of hallucination, while low coverage is mere incompleteness.

Overall confidence weights:
  With schema: 0.30 * schema_pass + 0.40 * semantic + 0.30 * grounding.
  Without:     0.50 * semantic                       + 0.50 * grounding.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type

import numpy as np
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer

# ── project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_MODEL             = "all-MiniLM-L6-v2"
_DEFAULT_CONFIDENCE_THRESH = 0.75
_DEFAULT_SEMANTIC_THRESH   = 0.70
_DEFAULT_COVERAGE_THRESH   = 0.60   # per-requirement similarity to count as covered
_DEFAULT_GROUNDING_THRESH  = 0.52   # per-claim similarity to count as grounded

# Weights for overall confidence
_W_SCHEMA   = 0.30
_W_SEMANTIC = 0.40
_W_GROUNDING= 0.30


# ── public types ──────────────────────────────────────────────────────────────

@dataclass
class SchemaResult:
    """Outcome of Level 1 schema validation.

    Attributes:
        passed:  True when the output satisfies the Pydantic schema.
        errors:  Human-readable validation error messages (empty on pass).
        skipped: True when no schema was configured (plain-text output).
    """
    passed:  bool
    errors:  list[str]
    skipped: bool


@dataclass
class ValidationReport:
    """Consolidated result from OutputValidator.validate().

    Attributes:
        schema_result:      Level 1 outcome.
        semantic_score:     Fraction of task requirements covered (0-1).
        missing_topics:     Requirement sentences not found in the output.
        grounding_score:    Fraction of output claims grounded in evidence (0-1).
        grounded_claims:    Count of claims supported by tool results.
        total_claims:       Total claims extracted from the output.
        overall_confidence: Weighted combination of all three levels (0-1).
        decision:           "PASS" | "RETRY" | "ESCALATE".
        reasons:            Ordered list of justifications for the decision.
    """
    schema_result:      SchemaResult
    semantic_score:     float
    missing_topics:     list[str]
    grounding_score:    float
    grounded_claims:    int
    total_claims:       int
    overall_confidence: float
    decision:           str
    reasons:            list[str]


# ── validator ─────────────────────────────────────────────────────────────────

class OutputValidator:
    """Three-level output validator for LLM agent responses.

    Combines structural, semantic, and grounding signals into a single
    confidence score and actionable decision.

    Usage::

        validator = OutputValidator(confidence_threshold=0.75)
        report = validator.validate(task=task_desc, output=agent_output, trace=conv)
        print(report.decision, report.overall_confidence)

    Args:
        schema:               Pydantic model class for Level 1 (optional).
        confidence_threshold: Minimum overall confidence to PASS (default 0.75).
        semantic_threshold:   Minimum semantic coverage for PASS vs RETRY (default 0.70).
        coverage_threshold:   Per-requirement cosine similarity cut-off (default 0.60).
        grounding_threshold:  Per-claim cosine similarity cut-off (default 0.52).
        embedding_model:      sentence-transformers model ID.
        client:               Optional Anthropic client. When provided, an LLM
                              judge refines the grounding score by detecting
                              claims that CONTRADICT the evidence -- something
                              embedding similarity cannot do (a false claim is
                              topically close to the fact it contradicts).
        llm_model:            Model ID for the optional LLM judge (default haiku).
    """

    def __init__(
        self,
        schema:               type[BaseModel] | None = None,
        confidence_threshold: float                  = _DEFAULT_CONFIDENCE_THRESH,
        semantic_threshold:   float                  = _DEFAULT_SEMANTIC_THRESH,
        coverage_threshold:   float                  = _DEFAULT_COVERAGE_THRESH,
        grounding_threshold:  float                  = _DEFAULT_GROUNDING_THRESH,
        embedding_model:      str                    = _DEFAULT_MODEL,
        client:               Any | None             = None,
        llm_model:            str                    = "claude-haiku-4-5",
    ) -> None:
        self.schema               = schema
        self.confidence_threshold = confidence_threshold
        self.semantic_threshold   = semantic_threshold
        self.coverage_threshold   = coverage_threshold
        self.grounding_threshold  = grounding_threshold
        self._client              = client
        self._llm_model           = llm_model
        self._model               = SentenceTransformer(embedding_model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(
        self,
        task:   str,
        output: str,
        trace:  list[dict[str, Any]] | None = None,
    ) -> ValidationReport:
        """Run all three validation levels and return the consolidated report.

        Args:
            task:   Original task description / requirements (plain text).
            output: Agent output to evaluate (plain text or JSON string).
            trace:  Full conversation trace as a list of message dicts.
                    Tool results inside the trace supply grounding evidence.
                    Pass None or [] to skip Level 3.

        Returns:
            ValidationReport with per-level scores and an overall decision.
        """
        trace = trace or []

        schema_result                      = self._validate_schema(output)
        semantic_score, missing_topics     = self._validate_semantic(task, output)
        grounding_score, grounded, total   = self._validate_grounding(output, trace)

        # Optional Level-3 refinement: an LLM judge can detect contradictions
        # that embedding similarity treats as "grounded" (topically close).
        llm_note: str | None = None
        if self._client is not None and total > 0:
            adjusted, llm_note = self._llm_refine_grounding(output, trace, grounding_score)
            if adjusted is not None:
                grounded        = round(adjusted * total)
                grounding_score = adjusted

        overall                            = self._compute_confidence(
            schema_result, semantic_score, grounding_score
        )
        decision, reasons                  = self._decide(
            schema_result, semantic_score, grounding_score,
            overall, missing_topics, grounded, total,
        )
        if llm_note:
            reasons.append(llm_note)

        return ValidationReport(
            schema_result=      schema_result,
            semantic_score=     round(semantic_score,  4),
            missing_topics=     missing_topics,
            grounding_score=    round(grounding_score, 4),
            grounded_claims=    grounded,
            total_claims=       total,
            overall_confidence= round(overall, 4),
            decision=           decision,
            reasons=            reasons,
        )

    # ------------------------------------------------------------------
    # Level 1: Schema validation
    # ------------------------------------------------------------------

    def _validate_schema(self, output: str) -> SchemaResult:
        """Parse output as JSON and validate against the Pydantic schema.

        Returns a skipped SchemaResult when no schema is configured.
        """
        if self.schema is None:
            return SchemaResult(passed=True, errors=[], skipped=True)

        try:
            data = json.loads(output.strip())
        except json.JSONDecodeError as exc:
            return SchemaResult(passed=False, errors=[f"JSON parse error: {exc}"], skipped=False)

        try:
            self.schema.model_validate(data)
            return SchemaResult(passed=True, errors=[], skipped=False)
        except ValidationError as exc:
            errors = [
                f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            ]
            return SchemaResult(passed=False, errors=errors, skipped=False)

    # ------------------------------------------------------------------
    # Level 2: Semantic completeness
    # ------------------------------------------------------------------

    def _validate_semantic(
        self,
        task:   str,
        output: str,
    ) -> tuple[float, list[str]]:
        """Check that the output covers all requirements mentioned in the task.

        Splits both task and output into sentences, encodes them, and computes
        the max cosine similarity between each requirement and any output sentence.
        A requirement is considered "covered" when its max similarity >= coverage_threshold.

        Returns:
            (semantic_score, list_of_uncovered_requirement_sentences)
        """
        requirements = _split_sentences(task)
        output_sents = _split_sentences(output)

        if not requirements:
            return (1.0, [])
        if not output_sents:
            return (0.0, requirements[:])

        req_embs = self._model.encode(requirements, normalize_embeddings=True)
        out_embs = self._model.encode(output_sents,  normalize_embeddings=True)

        sim_matrix   = np.array(req_embs) @ np.array(out_embs).T  # (n_req, n_out)
        max_sims     = sim_matrix.max(axis=1)                      # (n_req,)

        covered_mask  = max_sims >= self.coverage_threshold
        covered_count = int(covered_mask.sum())
        missing       = [requirements[i] for i, ok in enumerate(covered_mask) if not ok]

        return (covered_count / len(requirements), missing)

    # ------------------------------------------------------------------
    # Level 3: Evidence grounding
    # ------------------------------------------------------------------

    def _validate_grounding(
        self,
        output: str,
        trace:  list[dict[str, Any]],
    ) -> tuple[float, int, int]:
        """Verify that output claims are supported by tool results in the trace.

        Extracts all text from tool_result blocks as evidence corpus, then
        checks each output sentence (claim) for similarity to any evidence.

        Returns:
            (grounding_score, grounded_count, total_claims)
        """
        claims   = _split_sentences(output)
        evidence = _extract_evidence(trace)

        if not claims:
            return (1.0, 0, 0)
        if not evidence:
            return (0.0, 0, len(claims))

        claim_embs = self._model.encode(claims,    normalize_embeddings=True)
        evid_embs  = self._model.encode(evidence,  normalize_embeddings=True)

        sim_matrix    = np.array(claim_embs) @ np.array(evid_embs).T  # (n_claims, n_ev)
        max_sims      = sim_matrix.max(axis=1)
        grounded_mask = max_sims >= self.grounding_threshold
        grounded      = int(grounded_mask.sum())

        return (grounded / len(claims), grounded, len(claims))

    def _llm_refine_grounding(
        self,
        output:           str,
        trace:            list[dict[str, Any]],
        embedding_score:  float,
    ) -> tuple[float | None, str | None]:
        """Optional LLM judge: re-score grounding with contradiction detection.

        Embedding similarity cannot tell support from contradiction. This judge
        reads the evidence and the output and returns the fraction of factual
        claims that are actually SUPPORTED (not merely on-topic). Any exception
        or unexpected response leaves the embedding score untouched.

        Args:
            output:          The agent output under review.
            trace:           Conversation trace supplying tool-result evidence.
            embedding_score: The embedding-based grounding score, for the note.

        Returns:
            (adjusted_score, note). adjusted_score is None when the judge could
            not run, in which case the embedding score is kept.
        """
        evidence = _extract_evidence(trace)
        if not evidence:
            return (None, None)

        evidence_block = "\n---\n".join(evidence)
        prompt = (
            "You are a strict grounding verifier. Given EVIDENCE (tool outputs the "
            "agent actually observed) and an agent OUTPUT, estimate the fraction of "
            "the output's factual claims that are SUPPORTED by the evidence. A claim "
            "that CONTRADICTS the evidence counts as NOT supported, even if it is on "
            "the same topic. Reply with ONLY a number between 0.0 and 1.0.\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            f"OUTPUT:\n{output}\n\n"
            "Supported fraction (0.0-1.0):"
        )
        try:
            resp = self._client.messages.create(
                model=self._llm_model,
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            text  = resp.content[0].text.strip()
            match = re.search(r"[01](?:\.\d+)?", text)
            if not match:
                return (None, None)
            score = max(0.0, min(1.0, float(match.group())))
            note  = (
                f"[L3-LLM] LLM judge grounding {score:.2f} "
                f"(embedding-only was {embedding_score:.2f}; "
                "judge detects contradictions embeddings miss)"
            )
            return (score, note)
        except Exception as exc:   # noqa: BLE001 -- judge is best-effort
            return (None, f"[L3-LLM] judge unavailable ({type(exc).__name__}); kept embedding score")

    # ------------------------------------------------------------------
    # Confidence and decision
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        schema_result:   SchemaResult,
        semantic_score:  float,
        grounding_score: float,
    ) -> float:
        """Weighted combination of the three validation signals."""
        if self.schema is not None:
            schema_score = 1.0 if schema_result.passed else 0.0
            return _W_SCHEMA * schema_score + _W_SEMANTIC * semantic_score + _W_GROUNDING * grounding_score
        return 0.5 * semantic_score + 0.5 * grounding_score

    def _decide(
        self,
        schema_result:   SchemaResult,
        semantic_score:  float,
        grounding_score: float,
        overall:         float,
        missing_topics:  list[str],
        grounded:        int,
        total_claims:    int,
    ) -> tuple[str, list[str]]:
        """Map validation signals to a decision and ordered reasons list.

        Decision flow:
          1. Schema failure        -> ESCALATE (structurally invalid).
          2. overall >= threshold  -> PASS     (covered and grounded).
          3. semantic < threshold  -> RETRY    (incomplete: low topic coverage).
          4. otherwise             -> ESCALATE (topics covered but ungrounded =
                                                 fabrication risk).
        """
        reasons: list[str] = []

        # ── Level 1 failure → immediate ESCALATE ─────────────────────────────
        if not schema_result.skipped and not schema_result.passed:
            errs = "; ".join(schema_result.errors[:3])
            reasons.append(f"[L1-FAIL] Schema validation failed: {errs}")
            return ("ESCALATE", reasons)

        if not schema_result.skipped:
            reasons.append("[L1-PASS] Schema validation passed")

        # ── Level 2: semantic coverage reason ─────────────────────────────────
        if missing_topics:
            short = [t[:70] for t in missing_topics[:3]]
            reasons.append(
                f"[L2] Semantic coverage {semantic_score:.2f}: "
                f"{len(missing_topics)} topic(s) not covered -- {short}"
            )
        else:
            reasons.append(f"[L2] Semantic coverage {semantic_score:.2f}: all requirements covered")

        # ── Level 3: grounding reason ─────────────────────────────────────────
        if total_claims == 0:
            reasons.append("[L3] No claims to ground (empty output)")
        else:
            reasons.append(
                f"[L3] Grounding {grounded}/{total_claims} claims "
                f"(score={grounding_score:.2f})"
            )

        reasons.append(f"Overall confidence: {overall:.2f} (threshold {self.confidence_threshold})")

        # ── PASS: confident, covered, and grounded ───────────────────────────
        if overall >= self.confidence_threshold:
            reasons.append("[PASS] Confidence meets threshold")
            return ("PASS", reasons)

        # ── RETRY: incomplete — too few required topics covered ──────────────
        # Safe failure: the agent did not cover enough; retry to gather more.
        if semantic_score < self.semantic_threshold:
            reasons.append(
                f"[RETRY] Incomplete: coverage {semantic_score:.2f} < "
                f"{self.semantic_threshold} -- agent should gather more and retry"
            )
            return ("RETRY", reasons)

        # ── ESCALATE: covered the topics but claims are not grounded ─────────
        # Dangerous failure: output looks complete but contradicts/omits evidence.
        reasons.append(
            f"[ESCALATE] Topics covered (coverage {semantic_score:.2f}) but grounding "
            f"low ({grounding_score:.2f}) -- looks complete yet unsupported by evidence "
            "(fabrication risk); human review required"
        )
        return ("ESCALATE", reasons)


# ── module helpers ────────────────────────────────────────────────────────────

def _split_sentences(text: str, min_len: int = 12) -> list[str]:
    """Split text into non-trivial sentences for embedding.

    Strips code fences (``` blocks) before splitting so that raw code does not
    dilute the sentence embeddings, then splits on sentence-ending punctuation
    and newlines.

    Args:
        text:    Input text.
        min_len: Minimum character length to keep a sentence fragment.

    Returns:
        List of cleaned sentence strings.
    """
    # Remove code fences (keep code content, remove the fence markers)
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)

    # Split on period/excl/question followed by space or newline, or on newlines
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)

    cleaned: list[str] = []
    for s in raw:
        s = s.strip()
        if len(s) >= min_len:
            cleaned.append(s)
    return cleaned


def _extract_evidence(trace: list[dict[str, Any]]) -> list[str]:
    """Pull all tool_result text out of the conversation trace.

    Supports both plain-string content and the Anthropic list-of-blocks format.

    Args:
        trace: Conversation as a list of message dicts.

    Returns:
        List of evidence strings (one per tool_result block or content chunk).
    """
    evidence: list[str] = []

    for msg in trace:
        content = msg.get("content", "")

        if isinstance(content, str):
            # Plain-text user messages are not evidence
            continue

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")

                if btype == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, str) and inner.strip():
                        # Split long tool results into paragraphs
                        for para in re.split(r"\n{2,}", inner):
                            para = para.strip()
                            if para:
                                evidence.append(para)
                    elif isinstance(inner, list):
                        for sub in inner:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                t = sub.get("text", "").strip()
                                if t:
                                    evidence.append(t)

    return evidence


# ── demo task, trace, and outputs ─────────────────────────────────────────────

# Content-bearing acceptance criteria: each sentence names a specific required
# topic so that Level 2 coverage is meaningful. (Imperative instructions like
# "Identify all vulnerabilities" embed poorly and cannot be matched per-topic.)
_DEMO_TASK = (
    "The login function builds SQL queries with f-strings and is vulnerable to "
    "SQL injection; use parameterized queries. "
    "The JWT secret is hardcoded to a guessable value; load it from a secure "
    "environment variable. "
    "JWT tokens are configured with no expiry and remain valid forever; add a "
    "short token expiry. "
    "The password reset generates a weak four-digit numeric code; use a "
    "cryptographically secure token."
)

_DEMO_TRACE: list[dict[str, Any]] = [
    {
        "role": "user",
        "content": "Analyze the authentication system for security issues.",
    },
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "tu_001", "name": "read_file",
             "input": {"path": "auth.py"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_001",
                "content": (
                    "def login(username, password):\n"
                    "    query = f'SELECT * FROM users WHERE username={username} AND password={password}'\n"
                    "    result = db.execute(query)\n"
                    "    if result:\n"
                    "        return generate_token(username)\n\n"
                    "def generate_token(user):\n"
                    "    return jwt.encode({'user': user}, JWT_SECRET, algorithm='HS256')\n\n"
                    "def reset_password(email):\n"
                    "    token = str(random.randint(1000, 9999))\n"
                    "    send_email(email, token)"
                ),
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "tu_002", "name": "read_file",
             "input": {"path": "jwt_config.py"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_002",
                "content": (
                    "JWT_SECRET = 'secret123'\n"
                    "JWT_EXPIRY = None  # tokens never expire\n"
                    "ALGORITHM  = 'HS256'"
                ),
            },
        ],
    },
]

_GOOD_OUTPUT = """
Security Analysis of the Authentication System

Vulnerabilities identified:

1. SQL Injection in login()
The login function constructs SQL queries using f-strings, directly embedding user input.
An attacker can bypass authentication with input like: username = ' OR '1'='1.
Fix: use parameterised queries.

    def login(username, password):
        query = 'SELECT * FROM users WHERE username=? AND password=?'
        result = db.execute(query, (username, password))
        if result:
            return generate_token(username)

2. Hardcoded JWT secret
JWT_SECRET is set to 'secret123' which is trivially guessable.
Any attacker who knows the secret can forge valid tokens for any user.
Fix: load the secret from an environment variable and use a cryptographically random value.

    import os, secrets
    JWT_SECRET = os.environ.get('JWT_SECRET') or secrets.token_hex(32)

3. JWT tokens never expire
JWT_EXPIRY is None, meaning tokens remain valid indefinitely after issuance.
A stolen token grants permanent access with no way to revoke it.
Fix: set a short expiry and implement refresh tokens.

    JWT_EXPIRY = 3600  # 1 hour in seconds

4. Insecure password reset token
The reset token uses random.randint(1000, 9999), a 4-digit code with only 9000 possible values.
An attacker can brute-force all values within seconds.
Fix: use secrets.token_urlsafe(32) to generate a cryptographically secure token.

Remediation plan:
Apply all four fixes in the order listed, starting with the SQL injection as it is the most critical.
Deploy the patched version to staging and run automated security tests before releasing to production.
""".strip()

_PARTIAL_OUTPUT = """
Authentication System Review

I found one security issue in the code I read:

SQL Injection vulnerability: The login function uses an f-string to build the SQL query,
which allows an attacker to inject arbitrary SQL by crafting the username or password field.
This can bypass authentication entirely.

To fix this, replace the f-string with a parameterised query:
    result = db.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))

I was unable to complete the full review of all configuration files.
""".strip()

_BAD_OUTPUT = """
Security Report

The authentication system appears to be well-secured overall.
The login function uses bcrypt to hash passwords before comparing them to the database.
bcrypt provides strong protection against brute-force attacks.

The JWT implementation uses RSA-256 asymmetric signing, which is more secure than HMAC.
Tokens are configured to expire after 15 minutes, limiting the window for token theft.

The password reset flow uses a cryptographically secure random token from the secrets module.

Rate limiting is implemented on the login endpoint to prevent credential stuffing attacks.

No critical vulnerabilities were found. Minor recommendation: consider adding MFA support.
""".strip()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sep = "=" * 72

    # ── optional Anthropic client from .env (enables LLM grounding judge) ─────
    def _load_env(path: Path) -> None:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    _load_env(_ROOT / ".env")

    client  = None
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            print("Anthropic client ready -- LLM grounding judge ENABLED.\n")
        except ImportError:
            print("anthropic not installed -- embedding-only grounding.\n")
    else:
        print("ANTHROPIC_API_KEY not set -- embedding-only grounding "
              "(LLM judge would catch contradictions embeddings miss).\n")

    print(sep)
    print("  OutputValidator demo  |  3-level validation  |  3 output quality tiers")
    print(sep)
    print(f"\n  Task: {_DEMO_TASK}\n")
    print(f"  Trace: {len(_DEMO_TRACE)} messages, "
          f"{len(_extract_evidence(_DEMO_TRACE))} evidence blocks extracted")

    validator = OutputValidator(
        schema=               None,
        confidence_threshold= 0.75,
        semantic_threshold=   0.70,
        coverage_threshold=   0.60,
        grounding_threshold=  0.52,
        client=               client,
    )

    scenarios = [
        ("GOOD   output (expected: PASS)",     _GOOD_OUTPUT),
        ("PARTIAL output (expected: RETRY)",   _PARTIAL_OUTPUT),
        ("BAD    output (expected: ESCALATE)", _BAD_OUTPUT),
    ]

    for label, output in scenarios:
        print(f"\n{sep}")
        print(f"  Scenario: {label}")
        print(sep)
        print(f"  Output ({len(output)} chars): {output[:120]}...")

        report = validator.validate(
            task=_DEMO_TASK,
            output=output,
            trace=_DEMO_TRACE,
        )

        # ── schema ────────────────────────────────────────────────────────────
        sr = report.schema_result
        schema_line = (
            "SKIPPED (no schema)" if sr.skipped else
            f"PASS" if sr.passed else
            f"FAIL -- {'; '.join(sr.errors[:2])}"
        )

        print(f"\n  Level 1 -- Schema:    {schema_line}")
        print(f"  Level 2 -- Semantic:  {report.semantic_score:.4f}  "
              f"({'all covered' if not report.missing_topics else str(len(report.missing_topics)) + ' topic(s) missing'})")
        if report.missing_topics:
            for t in report.missing_topics[:3]:
                print(f"               missing: {t[:65]}")
        print(f"  Level 3 -- Grounding: {report.grounding_score:.4f}  "
              f"({report.grounded_claims}/{report.total_claims} claims grounded)")

        print(f"\n  Overall confidence : {report.overall_confidence:.4f}")

        decision_icon = {"PASS": "[PASS]", "RETRY": "[RETRY]", "ESCALATE": "[ESCALATE]"}
        print(f"  Decision           : {decision_icon.get(report.decision, '')} {report.decision}")
        print(f"\n  Reasons:")
        for reason in report.reasons:
            print(f"    - {reason}")

    print(f"\n{sep}")
