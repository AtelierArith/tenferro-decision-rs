# Reference fixtures

Engine tests compare optimized Rust results against tensors captured from the
reference implementations (`extern/Laya.jl`, `extern/JeffClient.jl`). A fixture
is one JSON file with a small, explicit schema; `reference-data` loads and
validates it.

## Schema

```jsonc
{
  "name": "string",              // required
  "metadata": {                  // optional
    "source": "string",          //   reference implementation / revision
    "notes": "string",
    "dtype": "string",
    "seed": 0,                   //   any other keys are kept as-is
    "extra": "..." 
  },
  "tensors": {                   // required; keys are tensor names
    "tensor_name": {
      "dtype": "f32|f64|i64|bool",   // required
      "order": "col-major|row-major", // optional, default col-major
      "shape": [2, 3],               // required, non-negative integers
      "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] // required, flat
    }
  }
}
```

Rules enforced by the loader:

- `data` must contain exactly `prod(shape)` values.
- `dtype` selects the payload type; `data` values must match it.
- `col-major` (default) means the first axis varies fastest, matching the
  engines' native layout. `row-major` records NumPy / Julia-reversed reference
  order; consumers that need a canonical layout must convert explicitly.

## Conventions

- Keep fixtures small and deterministic; record the reference `source`.
- One fixture per behavior under test, named after the behavior.
- Never store credentials, real prompts, or model inputs in a fixture.
