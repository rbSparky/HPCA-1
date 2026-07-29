"""Capability-based operation translation; unsupported operations stay explicit."""
CAPABILITIES = ["integer_alu","floating_alu","multiply","multiply_accumulate","shift","logic","compare","select_or_mux","load","store","constant","predicate","commutative","associative","pipelined","latency_normalized"]
_MAP = {
 "ADD": {"integer_alu","commutative","associative"}, "SUB": {"integer_alu"}, "MUL": {"integer_alu","multiply","commutative"},
 "MAC": {"integer_alu","multiply_accumulate","pipelined"}, "DIV": {"integer_alu"}, "LS": {"integer_alu","shift"}, "RS": {"integer_alu","shift"}, "ARS": {"integer_alu","shift"},
 "AND": {"integer_alu","logic"}, "OR": {"integer_alu","logic"}, "XOR": {"integer_alu","logic"}, "CMP": {"integer_alu","compare"}, "CLT": {"integer_alu","compare"}, "CGT": {"integer_alu","compare"},
 "SELECT": {"select_or_mux"}, "CMERGE": {"select_or_mux"},
 # Morpher's native DFG dump uses both the checked-in XML spelling
 # (LOAD/STORE) and the output-port-prefixed spelling used by some passes
 # (OLOAD/OSTORE).  These are semantic aliases, not opcode substitutions.
 "LOAD": {"load"}, "LOADB": {"load"}, "LOADH": {"load"}, "LOADCL": {"load"},
 "OLOAD": {"load"}, "OLOADB": {"load"}, "OLOADH": {"load"}, "OLOADCL": {"load"},
 "STORE": {"store"}, "STOREB": {"store"}, "STOREH": {"store"},
 "OSTORE": {"store"}, "OSTOREB": {"store"}, "OSTOREH": {"store"},
 "MOVC": {"constant"}, "CONST": {"constant"}, "BR": {"predicate"}, "JUMPL": {"predicate"}, "NOP": set(),
}
def translate_operation(op: str) -> dict:
    key = (op or "").strip().upper()
    caps = sorted(_MAP.get(key, set()))
    return {"opcode": key, "capabilities": caps, "supported": key in _MAP, "latency_normalized": 1.0}
