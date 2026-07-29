# Native baseline executor status

## Dual-linear

`dual_linear` is a real native executor. It solves one exact native parent
relaxation per state through `NativeRelaxationParentContextProvider`, then
scores the actual `NativeAction` set using immediate native route transitions,
parent native resource duals, and parent native DataPath compute duals. It does
not use the frozen neural checkpoint.

## FlowAdvantage proposal and top-4

`flow_proposal`, `flow_top4`, and `full_relaxed_lookahead` remain backed by the
frozen revision-v3 hybrid checkpoint and exact native child evaluator.

## No-parent proposal

The frozen checkpoint metadata declares `include_duals=true` and its 62-feature
schema contains parent objective, routing dual, compute dual, slack, and
solver-derived residual features. A solver-free scorer cannot be obtained by
zeroing those inputs or reusing parent values: that would be a proxy and would
violate the pilot protocol. `noparent_proposal` and `noparent_top4` therefore
fail closed with status `UNSUPPORTED` and an explicit explanation. They require
a separately trained no-parent checkpoint on a declared development split.

## Simulated annealing and LISA

The pinned Morpher checkout contains real native implementations. The mapper
dispatch in `src/CGRA_xml_compiler.cpp` is `-m 1` for
`SAMapper`/simulated annealing, `-m 0` for PathFinder, and `-m 2` for LISA.
The queue exposes these through `native_simulated_annealing` and `native_lisa`
method names and passes the selected native method code unchanged. No Python
proxy is used.
