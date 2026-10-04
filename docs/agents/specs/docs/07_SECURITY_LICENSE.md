# Security and License Requirements

## 1. Checkpoint security

Checkpoint readers must treat downloaded/model data as untrusted input.

Validate:

- tensor names
- tensor dimensions
- dtype
- byte lengths
- arithmetic overflow
- unreasonable allocation sizes
- duplicate/ambiguous entries
- malformed JSON/config
- path traversal in any file-based asset handling

Do not blindly allocate using attacker-controlled dimensions.

---

# 2. Download/cache safety

If remote checkpoint resolution is implemented:

- support offline reuse
- stage incomplete downloads
- publish only complete checkpoint directories
- prefer pinned revisions for reproducibility
- use atomic rename where practical
- recover safely after interruption
- do not execute code from checkpoint repositories

---

# 3. Jev credentials

Secrets must never appear in:

- `Debug`
- `Display`
- normal log output
- error messages
- panic context
- request URL
- tracing metadata

Use a secret-aware container and zeroization where practical.

---

# 4. Jev endpoint policy

Initial network policy should permit only known API endpoints.

Do not expose arbitrary URL execution through the public client API.

Redirects should remain disabled unless a future explicit policy is designed.

---

# 5. Retry/resource safety

The client should have bounded:

- response bytes
- retries
- retry budget
- request timeout
- first-byte timeout
- maximum inflight requests

A server response must not be able to trigger unbounded memory growth or indefinite retry.

---

# 6. License considerations

The Rust implementation must separately track:

- source-code license
- copied/adapted implementation license
- checkpoint/model-weight license
- tokenizer asset license
- upstream model license

Do not assume the model weight license is the same as the software license.

The existing Laya code is Apache-2.0 licensed.

The existing JevClient code is MIT licensed.

Jeff-related code and upstream dependencies/checkpoints must be audited individually before redistribution.

Recommended repository files:

```text
LICENSE
THIRD_PARTY_NOTICES
licenses/
```

If implementation details are substantially adapted from existing source, preserve attribution and applicable notices.

---

# 7. Generated/packed weights

Backend-specific weight packing may produce derived cache files.

Such caches should include:

- source checkpoint identity/revision
- packing version
- dtype
- backend target
- shape/config fingerprint

A stale packed cache must never be silently accepted for an incompatible source model.
