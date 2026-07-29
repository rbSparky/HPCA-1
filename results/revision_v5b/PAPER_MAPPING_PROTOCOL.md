# Native paper-mapping partial-state protocol

Executable paper jobs use a two-stage deterministic protocol. The native
reference mapping is read only to supply operation-latency metadata when the
export lacks a complete FU latency table; no reference placement, route,
occupancy, or score is copied into the search state.

1. Start from an empty `NativeMappingState`.
2. Run the same deterministic beam mapper with `LengthActionScorer` and the
   frozen beam/path/action limits.
3. Stop at `ceil(0.40 * |DFG operations|)`, clamped to at least one and at
   most `|DFG|-1`, and retain the deterministic best legal partial state.
4. Resume the selected method from that state and map the remaining operations
   to completion. The final result is therefore a complete mapping from a
   reproducible empty-state initialization, not an n-1-operation witness
   completion.

The prefix is recorded with `initialization_policy=deterministic_length_prefix`,
`initialization_depth_fraction=0.40`, and
`evaluation_mode=deterministic_length_prefix_then_complete`. Prefix metrics and
timing are retained in each atomic result. Any other initialization policy is
rejected by the worker rather than silently falling back to witness placement.
