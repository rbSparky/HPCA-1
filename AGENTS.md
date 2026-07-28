# AGENTS.md — QuotientFlow One-Hour Validation Prototype

## 0\. Read this first

You are implementing a scientific falsification prototype for a prospective HPCA paper on learned CGRA mapping.

The proposed method is called **QuotientFlow**:

> Conditioned on a partial CGRA mapping state, solve or predict the shadow prices of scarce routing resources from a regularized fractional placement/routing relaxation. Use those prices to guide discrete beam search. Separately, canonicalize exactly symmetric partial mapping states so the search explores one representative per symmetry orbit.

This repository does **not** begin by reproducing a complete CGRA compiler. The immediate objective is a controlled, deterministic validation suite whose execution takes approximately one hour after setup.

The suite must answer four questions:

1. **Relaxation signal:** Do routing shadow prices identify resources that are genuinely consequential to the residual mapping problem?
2. **Search utility:** Does oracle dual guidance improve mapping success or search effort over a length-only mapper?
3. **Learnability:** Can a small GNN predict enough of the oracle price ranking to retain a substantial fraction of the oracle improvement?
4. **Symmetry utility:** Does exact quotienting reduce explored states without changing the best mapping result?

Do not optimize for a visually impressive result. Optimize for a trustworthy result that can invalidate the idea early.

\---

## 1\. Scientific status and limitations

This is a **pilot**, not the final HPCA evaluation.

The pilot uses:

* small, deterministic DFG generators;
* small time-expanded CGRA graphs;
* a fixed modulo schedule derived from ASAP levels;
* a regularized fractional placement/routing problem;
* a bounded deterministic beam mapper;
* exact symmetries only;
* a small pure-PyTorch GNN.

The pilot does not claim:

* a complete modulo scheduler;
* full Morpher or CGRA-ME compatibility;
* physical timing, area, power, or energy results;
* superiority over every published learned mapper;
* final publication-quality evidence.

However, every API and result file must be designed so the implementation can later be connected to Morpher-v2 or CGRA-ME and expanded into a paper-grade evaluation.

A critical modeling point:

> Do not train or evaluate routing prices only at the fully symmetric root state.

At a symmetric root, a strongly regularized fractional optimizer may spread placement mass uniformly, producing uniform or weakly informative prices. Every learning/search sample must therefore be conditioned on a valid **partial mapping state**, normally after one canonical anchor placement and optionally after additional placements.

\---

## 2\. Non-negotiable engineering rules

1. Never fabricate, smooth, replace, or silently drop failed results.
2. Store every raw per-instance result.
3. Every experiment must be reproducible from a checked-in YAML config and explicit seed.
4. Use wall-clock timers around every major stage.
5. Enforce the total quick-suite budget. A run may stop optional work when the budget is nearly exhausted, but it must still write a valid partial report.
6. Do not install or require PyTorch Geometric. Implement the small GNN using pure PyTorch operations.
7. Do not make Morpher, CGRA-ME, LLVM, commercial EDA tools, Docker, or RTL tools prerequisites for the quick suite.
8. Do not silently use CPU PyTorch when an NVIDIA GPU is present. Verify CUDA explicitly.
9. Do not use approximate graph symmetries in this pilot.
10. Any optimization failure, infeasible instance, timeout, numerical residual violation, or automorphism truncation must be logged with a reason.
11. Run all unit tests before the experiment.
12. Keep the implementation readable enough for later artifact release.

\---

## 3\. Environment setup

### 3.1 Required platform

Preferred:

* Linux x86-64;
* NVIDIA GPU;
* CUDA-capable PyTorch;
* 16 GB system RAM;
* 8 GB VRAM is sufficient.

The quick suite should also be debuggable on CPU, but the official result run must record whether CUDA was available.

### 3.2 use a conda environment

first try to see if any pre existing conda env will be enough. only create one if all envs are insufficient. 

Use an existing `conda`, `mamba`, or `micromamba`. Prefer `mamba` when available.

Create:

```bash
conda create -n quotientflow python=3.11 pip -y
conda activate quotientflow
```

Install scientific dependencies from conda-forge:

```bash
conda install -c conda-forge \\
  numpy scipy pandas networkx cvxpy clarabel osqp \\
  scikit-learn pyyaml tqdm matplotlib pytest psutil -y
```

Install a CUDA-enabled PyTorch wheel **inside the conda environment** using the current official PyTorch selector. Do not assume that the system CUDA toolkit must match the wheel exactly; use a binary wheel compatible with the installed NVIDIA driver.

Procedure:

1. Run `nvidia-smi`.
2. Record driver version, reported CUDA compatibility, GPU model, and VRAM.
3. Open the official PyTorch “Get Started / Locally” installation selector or inspect its current wheel instructions.
4. Select Linux, Pip, Python, and a CUDA wheel supported by the driver.
5. Install `torch` using that official wheel command.
6. Do not install `torchvision` or `torchaudio` unless required.

Example only, not a version guarantee:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If that wheel is not supported, use the newest official CUDA wheel compatible with the current driver.

Verify:

```bash
python - <<'PY'
import torch
print("torch", torch.\_\_version\_\_)
print("cuda\_available", torch.cuda.is\_available())
print("torch\_cuda", torch.version.cuda)
if torch.cuda.is\_available():
    print("device", torch.cuda.get\_device\_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("gpu\_smoke", float(y\[0, 0]))
PY
```

The agent must save:

* `results/system/nvidia\_smi.txt`
* `results/system/conda\_env.yml`
* `results/system/pip\_freeze.txt`
* `results/system/system.json`

`system.json` must include:

```json
{
  "os": "",
  "cpu": "",
  "ram\_gb": 0,
  "gpu": "",
  "vram\_gb": 0,
  "nvidia\_driver": "",
  "torch\_version": "",
  "torch\_cuda\_version": "",
  "cuda\_available": false
}
```

### 3.3 GPU gate

* **GREEN:** `torch.cuda.is\_available()` is true and the GPU matrix-multiplication smoke test succeeds.
* **AMBER:** no GPU is accessible, but the complete suite runs on CPU within 75 minutes.
* **RED:** PyTorch cannot be installed or neither the GPU nor CPU run completes.

Continue implementation on CPU if necessary, but label the official environment gate accurately.

\---

## 4\. Required repository layout

Create or conform to:

```text
.
├── AGENTS.md
├── README.md
├── environment.yml
├── configs/
│   ├── quick.yaml
│   └── paper.yaml
├── quotientflow/
│   ├── \_\_init\_\_.py
│   ├── arch.py
│   ├── dfg.py
│   ├── mrrg.py
│   ├── partial\_state.py
│   ├── relaxation.py
│   ├── symmetry.py
│   ├── mapper.py
│   ├── model.py
│   ├── training.py
│   ├── metrics.py
│   └── reporting.py
├── scripts/
│   ├── doctor.py
│   ├── run\_quick\_suite.py
│   ├── run\_paper\_suite.py
│   └── make\_report.py
├── tests/
│   ├── test\_arch.py
│   ├── test\_dfg.py
│   ├── test\_flow\_conservation.py
│   ├── test\_relaxation.py
│   ├── test\_symmetry.py
│   ├── test\_mapper.py
│   └── test\_equivariance.py
└── results/
```

The single quick-suite command must be:

```bash
python scripts/run\_quick\_suite.py --config configs/quick.yaml
```

The command must resume safely if interrupted. Cache generated instances and solved relaxation targets.

\---

## 5\. Exact quick-suite configuration

Create `configs/quick.yaml` with these semantic values:

```yaml
experiment:
  name: quotientflow\_quick\_v1
  master\_seed: 24072026
  wall\_clock\_budget\_minutes: 55
  hard\_stop\_minutes: 70
  cpu\_threads: 6
  output\_dir: results/quick\_v1
  cache\_dir: results/cache
  deterministic\_torch: true

model:
  hidden\_dim: 64
  dfg\_layers: 3
  arch\_layers: 3
  head\_layers: 2
  dropout: 0.0
  epochs: 60
  early\_stop\_patience: 8
  learning\_rate: 0.001
  weight\_decay: 0.00001
  seeds: \[11, 23, 37]
  batch\_size: 8
  loss\_mse\_weight: 1.0
  loss\_rank\_weight: 0.25
  loss\_critical\_weight: 0.5

relaxation:
  tau: 0.001
  solver\_primary: CLARABEL
  solver\_fallback: OSQP
  primal\_tolerance: 0.0001
  active\_tolerance: 0.001
  max\_solve\_seconds: 8.0

architecture:
  ii: 3
  link\_capacity: 1.0
  compute\_capacity: 1.0
  include\_wait\_edges: true

data:
  train\_per\_family\_per\_arch: 8
  val\_per\_family\_per\_arch: 2
  test\_seen\_per\_family\_per\_arch: 2
  test\_per\_family\_unseen\_arch: 4
  partial\_depth\_fractions: \[0.10, 0.25, 0.40]
  min\_ops: 8
  max\_ops: 16

mapper:
  beam\_width: 32
  per\_state\_action\_limit: 6
  k\_paths: 4
  max\_expansions: 5000
  timeout\_seconds\_per\_instance: 2.0
  price\_weight: 2.0
  occupancy\_penalty: 1000.0
  canonicalize\_routes: true

evaluation:
  probe\_instances: 28
  knockout\_links\_per\_instance: 16
  mapping\_instances\_per\_split: 14
  exhaustive\_symmetry\_instances: 12
```

If a specific numerical value must be changed for correctness or to keep the run within the hard time limit, record the old and new values in:

`results/quick\_v1/config\_deviations.md`

Do not change gates after observing results.

\---

## 6\. Exact architecture dataset

Implement a typed, directed, time-expanded routing graph.

A routing node is:

```text
(x, y, t), 0 <= t < II
```

Every routing edge advances time by one modulo `II`.

All architectures include a wait/register edge:

```text
(x, y, t) -> (x, y, (t + 1) mod II)
```

All movement links are bidirectional at the physical level and become two directed time-advancing routing edges.

### A. `mesh3`

* Width: 3
* Height: 3
* II: 3
* Cardinal north/south/east/west links
* No wraparound
* Every PE supports `ADD` and `MUL`
* Exact architecture symmetry group: square-grid dihedral transformations that preserve the graph

Use for training, validation, and seen-architecture test.

### B. `torus3`

* Width: 3
* Height: 3
* II: 3
* Cardinal links with wraparound in both dimensions
* Every PE supports `ADD` and `MUL`
* Exact architecture symmetry group: translations composed with valid square dihedral transformations

Use for training, validation, and seen-architecture test.

### C. `mesh4`

* Width: 4
* Height: 4
* II: 3
* Cardinal links
* No wraparound
* Every PE supports `ADD` and `MUL`

Use only as an unseen-size test architecture.

### D. `diag4`

* Width: 4
* Height: 4
* II: 3
* Cardinal links
* Both diagonal directions between adjacent rows/columns
* No wraparound
* Every PE supports `ADD` and `MUL`

Use only as an unseen-topology test architecture.

### E. `cut4`

* Width: 4
* Height: 4
* II: 3
* Begin with cardinal mesh links
* Remove all physical links crossing the vertical cut between columns 1 and 2 except the links at rows 0 and 3
* No wraparound
* Every PE supports `ADD` and `MUL`
* Compute the exact subgroup of the square dihedral group that preserves the cut

Use only as an unseen-asymmetric-bottleneck test architecture.

Architecture split names must be:

* `train\_seen`
* `val\_seen`
* `test\_seen`
* `test\_unseen\_size`
* `test\_unseen\_topology`
* `test\_unseen\_asym`

\---

## 7\. Exact DFG dataset

A DFG is a typed directed acyclic graph. Node operation types are `ADD` and `MUL`.

Compute an ASAP level:

```text
level(v) = 0 for source operations
level(v) = 1 + max(level(u)) for predecessors u
```

The fixed modulo time is:

```text
slot(v) = level(v) mod II
```

This pilot intentionally isolates placement/routing from schedule learning.

Generate the following seven families.

### 7.1 `diamond\_chain`

A sequence of 2–4 split/join diamonds:

```text
a -> b
a -> c
b -> d
c -> d
```

Join output becomes the next block input.

Automorphisms: independent swap of the two branches in each diamond.

### 7.2 `reduction\_tree`

Balanced binary reduction with 4 or 8 leaf-producing operations.

* Leaf operations: `MUL`
* Internal operations: `ADD`

Use only instances with at most 16 total operations.

Automorphisms: recursive child swaps.

### 7.3 `dot\_product`

Use 2 or 4 independent `MUL` lanes followed by a balanced `ADD` reduction.

Automorphisms: lane permutations preserving the reduction structure. Generate the exact permutations analytically for the constructed graph.

### 7.4 `butterfly`

Width 4, with 2 or 3 butterfly stages.

Each butterfly combines paired inputs using two `ADD`-typed outputs. Keep total operations at or below 16.

Automorphisms: only include permutations known from construction and verify every permutation against the typed directed graph.

### 7.5 `fir`

A two-output FIR-style graph with 3 or 4 taps.

Each tap has one `MUL`; sums use `ADD`. Shared or repeated input-production nodes may be omitted so the graph remains 8–16 operations.

Automorphisms: only those explicitly preserved by the generated tap structure.

### 7.6 `gemm\_tile`

A small matrix-multiply micro-tile:

* shape `2 x 2`;
* reduction depth `k = 2`;
* eight `MUL` operations;
* four final or intermediate `ADD` operations.

Total: 12 operations.

Automorphisms: row/column permutations that preserve the generated typed dependencies.

### 7.7 `series\_parallel`

A random series-parallel DAG with 8–16 operations.

Construction operations:

* series extension;
* parallel split/join;
* bounded fan-in and fan-out;
* edge count between approximately `1.2n` and `1.8n`.

Assign `MUL` to approximately one-third of operations and `ADD` to the rest.

Default automorphism group: identity unless exact automorphisms are found and verified cheaply.

### Dataset counts

Training architectures are `mesh3` and `torus3`.

For each of 7 families and each training architecture:

* 8 training samples;
* 2 validation samples;
* 2 seen-test samples.

Counts:

* train: `7 \* 2 \* 8 = 112`
* validation: `7 \* 2 \* 2 = 28`
* seen test: `7 \* 2 \* 2 = 28`

For each unseen architecture `mesh4`, `diag4`, and `cut4`:

* 4 samples per family.

Counts:

* unseen size: 28
* unseen topology: 28
* unseen asymmetry: 28

Total target samples: **252**.

A sample is a DFG plus architecture plus partial mapping state, not merely a bare DFG.

\---

## 8\. Partial mapping states

For each sample:

1. Sort operations by:

   * ASAP level ascending;
   * fan-out descending;
   * stable node ID ascending.
2. Choose a partial-depth fraction from `\[0.10, 0.25, 0.40]` using the deterministic sample seed.
3. Place at least one operation.
4. Create the partial state using a deterministic greedy procedure:

   * for the first operation, select a canonical PE representative under the architecture symmetry group;
   * for later anchored operations, choose the legal PE minimizing shortest-path distance to already placed predecessors;
   * break ties lexicographically;
   * route newly completed dependencies using shortest available paths.
5. If the partial state cannot be constructed, log it and deterministically generate a replacement sample from the same family/split.

The partial state must encode:

* placed operation to PE/time-slot mapping;
* occupied compute slots;
* occupied routing links;
* already routed DFG edges;
* remaining operations and dependencies.

The relaxation and GNN input are conditioned on this state.

\---

## 9\. Fractional residual mapping relaxation

For an instance and partial state, define:

* `x\[v, r]` for each remaining operation `v` and compatible compute slot `r`;
* `f\[e, l]` for each unrouted dependency `e` and available routing edge `l`.

Occupied resources have zero residual capacity.

Solve:

```text
minimize:
    sum\_e,l base\_cost\[l] \* f\[e,l]
  + tau/2 \* (sum\_v,r x\[v,r]^2 + sum\_e,l f\[e,l]^2)
```

Subject to:

1. Every remaining operation is fractionally assigned once.
2. Compute-slot capacity.
3. Flow conservation for every remaining dependency.
4. Routing-link residual capacity.
5. `0 <= x <= 1`.
6. `0 <= f <= 1`.
7. Placement variables of already anchored operations are fixed implicitly by the residual source/sink terms.
8. Existing routes remain occupied and fixed.

Use base routing cost:

* wait edge: `1.05`;
* cardinal edge: `1.00`;
* diagonal edge: `1.15`;
* wrap edge: `1.00`.

Store the dual value of every routing-capacity constraint as the oracle routing price.

Because numerical solvers may return one of multiple nearly equivalent dual solutions:

* use a strongly convex primal objective;
* use deterministic solver settings;
* compare dual rankings and symmetrized/equivariant quantities rather than requiring bitwise equality;
* store raw duals;
* optionally compute a minimum-norm active-set dual in a second solve if implementation time permits;
* do not block the pilot on the optional second solve.

Normalize prices per sample for learning and routing:

```text
positive\_scale = median(positive duals) or 1.0 if none
target\[l] = log1p(dual\[l] / (positive\_scale + 1e-8))
```

Store both raw and normalized prices.

### Solver health metrics

For every solved sample, calculate:

* assignment equality residual;
* maximum flow-conservation residual;
* maximum capacity violation;
* minimum variable value;
* maximum variable value;
* solver status;
* solve time;
* objective value;
* fraction of routing edges with positive price.

\---

## 10\. Exact knockout test

The knockout test validates whether a high dual price marks a consequential resource.

For each of the configured 28 probe instances:

1. Rank available routing links by oracle dual price.
2. Select:

   * top 8 links;
   * 4 links near the median;
   * bottom 4 links.
3. For each selected link:

   * set its residual capacity to zero;
   * re-solve the same residual relaxation;
   * record objective increase;
   * record infeasibility.
4. Treat infeasibility as an objective increase larger than every finite increase in that instance.
5. Compute per-instance:

   * Spearman correlation between original dual price and knockout objective increase;
   * top-8 recall of links in the top-8 knockout impacts;
   * mean impact of top-price links divided by mean impact of bottom-price links.

This test is not sufficient by itself because LP sensitivity and dual prices are related by construction. Its role is to verify the implementation and establish that the chosen regularization/normalization retains useful ranking under finite capacity removal.

\---

## 11\. Exact symmetries

### 11.1 Verification

Every proposed DFG or architecture permutation must be checked programmatically:

* node types preserved;
* directed edges preserved;
* edge types preserved;
* modulo-time coordinate transformed correctly;
* capacities preserved;
* operation support preserved.

Reject any invalid transform.

### 11.2 State action

A joint transform `(g\_D, g\_A)` acts on:

* DFG operation IDs;
* placed operation assignments;
* occupied compute slots;
* occupied routing links;
* routed DFG-edge IDs;
* pending dependencies.

### 11.3 Canonical form

Serialize a transformed state as a tuple containing:

1. sorted transformed operation placements;
2. sorted occupied compute resources;
3. sorted occupied routing-edge IDs;
4. sorted routed DFG-edge IDs.

The canonical state is the lexicographically minimum serialization over all verified joint transforms.

For large transform products, cache:

* transformed resource-index maps;
* transformed routing-edge-index maps;
* DFG permutation maps;
* state hashes.

Do not truncate a group and still call the result exact. If a group is too large for the time budget:

* skip that sample in the exact joint-symmetry subtest;
* retain architecture-only quotienting for the integrated mapper;
* report the skip.

### 11.4 Exactness subtest

Use 12 small symmetric instances:

* 4 `diamond\_chain`;
* 4 `reduction\_tree` with four leaves;
* 4 `dot\_product` with two or four lanes;
* architecture: `mesh3`;
* 6–12 operations.

Run both:

* non-quotient exhaustive or sufficiently wide beam search;
* joint-quotient search.

Require identical:

* success/failure;
* best route cost;
* mapped operation count;
* legality verification result.

\---

## 12\. Small pure-PyTorch GNN

Do not use PyTorch Geometric.

### 12.1 DFG encoder

Input node features:

* one-hot operation type: ADD/MUL;
* normalized in-degree;
* normalized out-degree;
* normalized ASAP level;
* modulo slot one-hot for II=3;
* anchored flag;
* anchored x coordinate normalized, or zero;
* anchored y coordinate normalized, or zero.

Use three message-passing layers:

```text
h\_i' = ReLU(W\_self h\_i + mean\_{j -> i}(W\_in h\_j) + mean\_{i -> k}(W\_out h\_k))
```

Mean-pool all DFG node embeddings into a graph embedding.

### 12.2 Architecture encoder

Routing-node features:

* normalized x;
* normalized y;
* modulo-time one-hot;
* normalized in-degree;
* normalized out-degree;
* occupied-compute flag;
* boundary flags: north/south/east/west;
* architecture-type flags are forbidden; topology must be inferred from the graph.

Use three directed message-passing layers.

### 12.3 Routing-edge head

For each routing edge `l = (a -> b)`, concatenate:

* source architecture embedding;
* destination architecture embedding;
* pooled DFG embedding;
* edge-type one-hot;
* occupied-link flag;
* residual capacity;
* direction delta `(dx, dy)`;
* wrap flag.

Predict:

* normalized log-price regression value;
* critical-link logit.

A critical link is in the top 10% of positive oracle prices for that sample. If fewer than 10 positive links exist, use the highest-priced positive link. If no dual is positive, mark the sample as having no critical-link classification target.

### 12.4 Loss

Use:

```text
loss =
    1.0 \* masked MSE on normalized log price
  + 0.25 \* pairwise ranking loss
  + 0.5 \* weighted BCE on critical-link label
```

Pairwise ranking:

* sample up to 64 edge pairs per graph;
* include only pairs whose oracle normalized prices differ by at least 0.1;
* use a logistic pairwise loss.

Use AdamW.

Train three model seeds: 11, 23, 37.

Use early stopping on validation Spearman correlation.

Store each checkpoint and select the seed with the best validation Spearman for integrated search. Report all seeds; do not report only the best seed for prediction metrics.

\---

## 13\. Mapper

Implement a deterministic beam mapper over remaining operations.

### 13.1 Operation order

Use the same stable order as partial-state construction:

1. ASAP level ascending;
2. fan-out descending;
3. node ID ascending.

### 13.2 Candidate placement

For the next operation:

1. enumerate unoccupied PEs at the operation’s fixed modulo slot;
2. for every newly completed incoming dependency, enumerate up to four shortest feasible simple paths;
3. discard actions for which any required incoming dependency cannot be routed;
4. score actions;
5. keep the configured per-state action limit.

### 13.3 Base score

For a placement and its selected new paths:

```text
base\_score =
    sum(base edge costs)
  + 1000 \* any capacity violation
```

Capacity violations must normally be forbidden rather than accepted.

### 13.4 Dual score

For oracle or predicted guidance:

```text
dual\_score =
    base\_score
  + price\_weight \* sum(normalized price\[l] for new path links)
```

Use static prices calculated or predicted at the initial partial state. Dynamic re-solving is a paper-stage extension, not required for the one-hour run.

### 13.5 Beam objective

Lower is better.

Track:

* total route cost;
* number of mapped operations;
* expansions;
* generated successors;
* duplicate states;
* quotient merges;
* canonicalization time;
* routing attempts;
* failed routing attempts;
* wall-clock time.

### 13.6 Methods

Run exactly these methods where data is available:

1. `length`

   * length-only score;
   * no quotient.
2. `quotient\_length`

   * length-only score;
   * architecture quotient;
   * joint quotient on the exact-symmetry subset.
3. `oracle\_dual`

   * oracle normalized dual prices;
   * no quotient.
4. `oracle\_dual\_quotient`

   * oracle normalized dual prices;
   * quotient enabled.
5. `pred\_dual`

   * GNN-predicted normalized prices;
   * no quotient.
6. `pred\_dual\_quotient`

   * GNN-predicted normalized prices;
   * quotient enabled.

Every reported successful result must pass an independent legality checker.

\---

## 14\. Exact evaluation splits

### Prediction evaluation

Use all samples:

* validation: 28;
* test seen: 28;
* unseen size: 28;
* unseen topology: 28;
* unseen asymmetry: 28.

### Mapping evaluation

Use exactly 14 deterministic samples per test split:

* two samples per DFG family;
* 4 test splits;
* total 56 mapping instances.

Splits:

* `test\_seen`;
* `test\_unseen\_size`;
* `test\_unseen\_topology`;
* `test\_unseen\_asym`.

Run all six mapping methods on the same 56 instances.

If the hard wall-clock budget is approached:

1. finish all methods for `test\_seen` and `test\_unseen\_asym`;
2. then finish unseen size;
3. then unseen topology;
4. never compare methods on unequal instance sets without explicitly marking the comparison.

\---

## 15\. Metrics

### 15.1 Prediction metrics

Per split and model seed:

* MAE on normalized price;
* RMSE;
* Spearman rank correlation per graph, then median and geometric/ordinary mean as appropriate;
* top-10%-critical AUROC;
* top-10%-critical average precision;
* top-8 critical-link recall;
* action-preservation rate.

### 15.2 Action-preservation rate

For each mapping-evaluation partial state:

1. enumerate all legal first remaining-operation actions;
2. score actions using oracle prices;
3. score the same actions using predicted prices;
4. record whether the predicted best action matches an oracle-best action;
5. also record oracle-score regret of the predicted action.

Ties within `1e-8` count as matching.

### 15.3 Mapping metrics

Per method and split:

* legal mapping success rate;
* median and p90 expansions;
* median compile time;
* median routing attempts;
* median failed routing attempts;
* median route cost among instances solved by every compared method;
* timeout count;
* legality failure count.

For expansion/time comparisons, use paired instances.

### 15.4 Oracle-gain retention

For a lower-is-better metric `m`, define:

```text
oracle\_gain = m(length) - m(oracle\_dual)
pred\_gain   = m(length) - m(pred\_dual)
retention   = pred\_gain / oracle\_gain
```

Only calculate when `oracle\_gain > 0`.

For success rate, use:

```text
retention =
  (success(pred\_dual) - success(length))
  / (success(oracle\_dual) - success(length))
```

Only calculate when the denominator is positive.

Clip nothing. Report values below zero or above one.

\---

## 16\. Pass/fail gates

Do not change these thresholds after seeing results.

Each gate receives `GREEN`, `AMBER`, or `RED`.

### G0 — environment and runtime

**GREEN**

* CUDA PyTorch works;
* all tests pass;
* quick suite completes within 60 minutes.

**AMBER**

* CPU-only or completion between 60 and 75 minutes;
* no correctness failures.

**RED**

* tests fail;
* suite does not complete;
* hard stop exceeds 75 minutes;
* results are incomplete without clear status.

### G1 — relaxation correctness

**GREEN**

* at least 98% of intended samples solve;
* p99 assignment residual <= `1e-4`;
* p99 flow residual <= `1e-4`;
* p99 capacity violation <= `1e-4`;
* no successful sample violates variable bounds by more than `1e-5`.

**AMBER**

* 90–98% solve, or p99 residual <= `1e-3`.

**RED**

* fewer than 90% solve;
* systematic residual or formulation errors.

This gate is mandatory. Do not interpret downstream metrics when G1 is red.

### G2 — finite-knockout signal

Use median per-instance statistics.

**GREEN**

* median Spearman(dual, knockout impact) >= `0.60`;
* median top-8 recall >= `0.60`;
* median top-price/bottom-price impact ratio >= `2.0`.

**AMBER**

* Spearman in `\[0.35, 0.60)`, or recall in `\[0.40, 0.60)`.

**RED**

* median Spearman < `0.35`;
* high-price links are not more consequential than low-price links.

This is an implementation/signal gate, not the main research gate.

### G3 — oracle-dual search utility

Compare `oracle\_dual` with `length` using paired mapping instances.

**GREEN** if either condition holds, with no success-rate loss greater than 2 percentage points:

* absolute legal mapping success improves by at least `8` percentage points; or
* median expansions fall by at least `20%` on commonly solved instances.

Additionally, on `test\_unseen\_asym`, require either:

* success improves by at least `10` percentage points; or
* expansions fall by at least `25%`.

**AMBER**

* success improves by 3–8 points; or
* expansions fall by 8–20%.

**RED**

* oracle dual is materially worse;
* success drops by more than 5 points;
* expansions increase by more than 10% without a success improvement.

This is the most important gate. If G3 is red, the current price formulation or mapper integration must be redesigned before investing further in the GNN.

### G4 — exact quotient correctness and utility

Correctness is absolute:

* quotient and non-quotient search must return identical success/failure and best cost on all 12 exactness instances;
* every result must pass legality checking.

Any correctness mismatch is **RED**.

If correctness holds:

**GREEN**

* geometric-mean expansion reduction >= `1.5x` on the symmetric subset;
* at least one family shows >= `2.0x`;
* canonicalization overhead <= `25%` of quotient mapper runtime.

**AMBER**

* reduction between `1.1x` and `1.5x`, or overhead 25–50%.

**RED**

* no reduction, or overhead exceeds 50%, even though correctness holds.

### G5 — GNN price prediction

Aggregate over three seeds; report mean and standard deviation.

**GREEN**

Seen test:

* median per-graph Spearman >= `0.60`;
* critical-link AUROC >= `0.80`;
* action preservation >= `0.70`.

Unseen architectures, averaged over size/topology/asymmetry:

* median Spearman >= `0.40`;
* critical-link AUROC >= `0.70`;
* action preservation >= `0.55`.

**AMBER**

Seen Spearman 0.40–0.60 or unseen Spearman 0.25–0.40.

**RED**

Seen Spearman < 0.40 and critical AUROC < 0.70.

Do not discard the method solely because G5 is amber/red if oracle guidance passes. Prediction can be improved after validating the underlying signal.

### G6 — integrated predicted mapper

Compare `pred\_dual\_quotient` with `length`.

**GREEN** if:

* legal mapping success does not decrease by more than 2 percentage points;
* either success improves by at least 5 points or median expansions fall by at least 25%;
* predicted guidance retains at least 60% of the positive oracle improvement for one primary metric;
* results hold on at least three of the four test splits.

**AMBER**

* no meaningful success loss;
* expansion reduction 10–25%, or oracle-gain retention 30–60%.

**RED**

* success decreases by more than 5 points;
* prediction reverses most oracle improvements.

### Overall decision

* **PROCEED:** G1, G3, and G4 are green; G5/G6 may be amber.
* **PROCEED WITH REFINEMENT:** G1 green, G3 or G4 amber, and neither is red.
* **PIVOT TO DUAL-ONLY:** G3 green but G4 red because useful exact symmetry is too small or expensive.
* **PIVOT TO QUOTIENT-ONLY:** G4 green but G3 red after one documented price-integration revision.
* **STOP/REFORMULATE:** G1 green but both G3 and G4 are red.
* **INVALID RUN:** G1 red.

\---

## 17\. Required output tables

Write both CSV and Markdown versions.

### Table T00 — system and run manifest

Files:

* `tables/T00\_system.csv`
* `tables/T00\_system.md`

Columns:

```text
run\_id
git\_commit
timestamp\_start
timestamp\_end
wall\_minutes
os
cpu
ram\_gb
gpu
vram\_gb
nvidia\_driver
torch\_version
torch\_cuda\_version
cuda\_available
config\_hash
tests\_passed
hard\_stop\_triggered
```

### Table T01 — dataset census

Columns:

```text
split
architecture
dfg\_family
requested\_samples
generated\_samples
solved\_samples
infeasible\_samples
solver\_failed\_samples
mean\_ops
mean\_dfg\_edges
mean\_partial\_depth
mean\_routing\_nodes
mean\_routing\_edges
```

### Table T02 — relaxation health

Columns:

```text
split
architecture
num\_samples
solve\_rate
median\_solve\_ms
p95\_solve\_ms
p99\_assignment\_residual
p99\_flow\_residual
p99\_capacity\_violation
min\_variable
max\_variable
median\_positive\_dual\_fraction
```

### Table T03 — knockout signal

Columns:

```text
split
architecture
num\_probe\_instances
median\_spearman
mean\_spearman
median\_top8\_recall
median\_top\_bottom\_impact\_ratio
num\_knockout\_infeasible
gate\_G2
```

### Table T04 — prediction quality

One row per model seed and split, plus aggregate rows.

Columns:

```text
model\_seed
split
architecture
num\_samples
price\_mae
price\_rmse
median\_spearman
mean\_spearman
critical\_auroc
critical\_average\_precision
top8\_recall
action\_preservation
mean\_action\_regret
train\_seconds
best\_epoch
```

### Table T05 — symmetry exactness and reduction

Columns:

```text
instance\_id
dfg\_family
architecture
dfg\_group\_size
arch\_group\_size
joint\_group\_size
nonquot\_success
quot\_success
nonquot\_best\_cost
quot\_best\_cost
cost\_match
legality\_match
nonquot\_expansions
quot\_expansions
expansion\_reduction
canonicalization\_ms
quot\_total\_ms
canonicalization\_fraction
```

### Table T06 — mapper aggregate

Columns:

```text
split
method
num\_instances
success\_rate
timeouts
legality\_failures
median\_expansions
p90\_expansions
median\_generated\_successors
median\_duplicate\_states
median\_quotient\_merges
median\_routing\_attempts
median\_failed\_routes
median\_runtime\_ms
p90\_runtime\_ms
paired\_route\_cost\_mean
```

### Table T07 — mapper paired deltas versus length

Columns:

```text
split
method
paired\_instances
success\_delta\_points
median\_expansion\_reduction\_pct
median\_runtime\_reduction\_pct
median\_route\_cost\_delta\_pct
oracle\_gain\_retention\_success
oracle\_gain\_retention\_expansions
```

### Table T08 — gates

Columns:

```text
gate
status
primary\_metric
observed\_value
green\_threshold
amber\_threshold
notes
```

### Raw per-instance results

Write:

* `raw/instances.csv`
* `raw/relaxation.csv`
* `raw/knockouts.csv`
* `raw/predictions.csv`
* `raw/actions.csv`
* `raw/mappings.csv`
* `raw/symmetry.csv`

Do not round raw CSV values.

\---

## 18\. Required report

Write:

`results/quick\_v1/REPORT.md`

The report must contain:

1. Exact command used.
2. Exact environment.
3. Runtime breakdown.
4. Dataset census.
5. All gates.
6. The six required method comparisons.
7. Per-family and per-architecture failure analysis.
8. At least four plots:

   * dual price versus knockout impact;
   * oracle versus predicted dual rank;
   * mapping success by method/split;
   * expansion count by method/split.
9. A section titled `What failed`.
10. A section titled `Scientific interpretation`.
11. A section titled `Recommended next iteration`.
12. A section titled `Can this support an HPCA paper?`

The interpretation must separate:

* underlying optimization-signal validity;
* learned-prediction quality;
* symmetry prevalence;
* search integration;
* implementation/runtime limitations.

\---

## 19\. Mandatory unit tests

At minimum:

1. Architecture graph has expected node/edge counts.
2. Every routing edge advances time by exactly one modulo II.
3. DFGs are acyclic and within size limits.
4. Partial states do not exceed resource capacities.
5. A one-edge flow-conservation toy problem solves exactly.
6. A known infeasible capacity problem reports infeasible.
7. Identity transform leaves state unchanged.
8. Every generated automorphism preserves typed graph adjacency.
9. Canonicalization maps equivalent states to the same key.
10. Non-equivalent hand-constructed states do not collide.
11. Mapper success passes independent legality checking.
12. A transformed relaxation instance has equal objective within tolerance.
13. GNN output shape matches routing-edge count.
14. CPU and CUDA forward passes agree within normal floating-point tolerance.

\---

## 20\. Failure-driven iteration rules

If G3 oracle utility fails:

1. First inspect whether prices are nearly uniform.
2. Increase partial depth from 10% to 25–40%.
3. Re-solve prices after every two mapped operations for a small ablation.
4. Replace static path-price sum with:

   * maximum path price;
   * bottleneck top-k sum;
   * price-weighted alternative-path scarcity.
5. Test `price\_weight` in `{0.5, 1.0, 2.0, 4.0}` on validation only.
6. Do not tune on test splits.

If G5 prediction fails while G3 passes:

1. Predict price ranks or top-decile criticality instead of magnitude.
2. Add partial-state occupancy features.
3. Add edge betweenness and residual alternative-path count as non-learned input features.
4. Increase hidden dimension to 96 only after checking underfitting.
5. Add one residual primal-dual correction step as a later extension.
6. Keep the same test splits.

If G4 symmetry utility fails:

1. Break down by DFG family and architecture.
2. Measure whether group size is small or canonicalization is expensive.
3. Use stabilizer-aware action-orbit reduction rather than canonicalizing every successor.
4. Cache transform hashes.
5. Do not use approximate symmetry in the pilot.
6. If exact symmetry is inherently rare, demote it from headline contribution.

\---

## 21\. Paper-grade expansion plan

The quick suite is designed to decide whether further work is justified. It is not strong enough by itself for HPCA.

If the overall decision is `PROCEED` or `PROCEED WITH REFINEMENT`, create `configs/paper.yaml` and plan the following final evaluation.

### 21.1 Real toolchain integration

Primary integration target:

* Morpher-v2 or the stable Morpher mapper stack;
* use its architecture description, MRRG, PathFinder, simulated annealing, LISA integration, and cycle-accurate validation where available.

Secondary target:

* CGRA-ME for exact/ILP reference cases and resource-graph compatibility.

The custom pilot graph must remain as a unit-test backend.

### 21.2 Final benchmark suite

At least 30 real kernels.

PolyBench/C candidates:

```text
2mm
3mm
atax
bicg
cholesky
doitgen
gemm
gemver
gesummv
jacobi-1d
mvt
syr2k
syrk
trisolv
trmm
```

MachSuite/application candidates:

```text
aes
backprop
fft
gemm
kmp
md-knn
nw
sort
spmv-crs
stencil2d
```

Additional repeated/symmetric kernels:

```text
fir
fft-butterfly
conv1d
conv2d
depthwise-conv
dot-product
reduction
```

Use extracted DFGs, not hand-drawn replacements, for final results.

### 21.3 Final architectures

At least 12 configurations spanning:

* sizes: 4x4 and 8x8;
* topologies: mesh, torus, diagonal/HyCUBE-like;
* memory connectivity: full, edge-only, restricted;
* heterogeneity: homogeneous and sparse multiply-capable PEs;
* II search over a meaningful range.

Hold out entire architecture families during training.

### 21.4 Final baselines

Required where runnable:

* PathFinder;
* simulated annealing;
* LISA;
* E2EMap;
* exact ILP or SAT on small cases;
* the strongest available open learned mapper;
* length-only beam;
* oracle relaxation guidance;
* prediction without quotient;
* quotient without prediction;
* full method.

Do not claim a baseline result unless its artifact was run or the paper’s published number is clearly labeled as external and non-normalized.

### 21.5 Final evidence

A strong HPCA submission should include:

* minimum achieved II;
* success under compilation time budgets;
* end-to-end compile time;
* search states and routing attempts;
* generalization to unseen architecture sizes/topologies;
* ablations for price target, correction steps, quotienting, and state refresh frequency;
* exact optimality comparison on small instances;
* cycle-accurate execution validation;
* hardware-performance consequence of lower II;
* sensitivity to routing capacity, memory connectivity, and PE heterogeneity;
* at least five seeds for learned components;
* confidence intervals or paired nonparametric tests;
* artifact with one-command reproduction.

### 21.6 Final paper gates

Do not begin paper writing until the expanded system shows all of:

1. At least 10 percentage points higher mapping success under a fixed compile budget **or** at least 1.5x compile-time reduction at matched minimum-II success.
2. Improvement on at least 70% of real kernels.
3. Benefit on unseen architecture families, not only new DFGs.
4. Predicted method retains at least 70% of oracle-relaxation benefit.
5. Exact quotienting reduces expansions by at least 1.5x on the subset with nontrivial symmetry, with zero quality loss.
6. A complete ablation showing that neither the GNN nor quotient component is decorative.
7. End-to-end cycle-accurate validation for successful mappings.
8. No dependence on specialized physical hardware for reproduction.

\---

## 22\. Deliverable checklist

Before declaring completion, ensure the repository contains:

* working conda environment instructions;
* CUDA verification;
* deterministic config;
* all tests passing;
* cached generated dataset;
* cached oracle prices;
* three trained model seeds;
* all six mapper methods;
* legality checker;
* exact symmetry test;
* required CSV/Markdown tables;
* raw results;
* plots;
* `REPORT.md`;
* `RUN\_SUMMARY.txt` with the final gate decision.

`RUN\_SUMMARY.txt` must be concise and contain:

```text
OVERALL\_DECISION:
TOTAL\_WALL\_MINUTES:
CUDA\_AVAILABLE:
G0:
G1:
G2:
G3:
G4:
G5:
G6:
TOP\_THREE\_FINDINGS:
TOP\_THREE\_FAILURES:
NEXT\_ACTION:
```

Do not stop after implementing code. Run the suite and return the actual result files.
