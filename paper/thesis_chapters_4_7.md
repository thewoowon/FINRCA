# FINRCA: Thesis Chapters 4–7, References, and Appendices

---

# Chapter 4. The APA-RCA Algorithm

## 4.1 Algorithm Overview

APA-RCA (Anomaly-Propagation-Aware Root Cause Analysis) is a graph-traversal algorithm that localizes the root cause of data quality anomalies in financial risk pipelines by reasoning over the Financial Risk Lineage Graph (FRLG). The algorithm is grounded in three design principles that distinguish it from standard graph proximity methods.

**Principle 1 — Backward traversal.** In a data lineage graph, directed edges run from cause (upstream) to effect (downstream). Root cause search must therefore proceed in the opposite direction: starting from the observed anomalous report node and walking upstream toward the origin. APA-RCA operates on the *transposed* graph G^T, reversing all edge directions so that the random walk moves naturally against the data flow and concentrates probability mass toward plausible causal origins.

**Principle 2 — Anomaly-score-weighted transitions.** A purely structure-based traversal treats all upstream nodes as equally plausible candidates, ignoring the signal strength that the anomaly detector has already computed. APA-RCA weights each transition probability by the anomaly score of the destination node multiplied by a transform-type prior, creating a biased walk that preferentially visits high-anomaly nodes along the traversal path.

**Principle 3 — Ancestor constraint.** In a directed acyclic graph, the root cause of an anomaly at node t must belong to the ancestor set ancestors(t). No non-ancestor node can have influenced t's output through the data dependency structure. Exploiting this structural fact eliminates the majority of non-ancestor candidates from the ranking and prevents probability mass from leaking to nodes that are unreachable from t.

The full algorithm proceeds in five stages: (1) transpose graph construction, (2) hop-distance computation via BFS, (3) anomaly-weighted transition matrix assembly, (4) RWR iteration with an anomaly-weighted restart distribution, and (5) composite scoring with ancestor constraint and upstream-preference correction.

## 4.2 The Anomaly Transfer Function (ATF) Framework

### 4.2.1 Transfer Coefficients ρ_τ

In a multi-stage data pipeline, an anomalous signal does not propagate unchanged through every transformation type. A spike injected at a source node is attenuated by an averaging aggregation, amplified in relative terms by a downstream sensitivity calculation, and nearly eliminated by a binary thresholding operation at the report layer. To formalize this intuition, we introduce the **Anomaly Transfer Function (ATF)** framework, which characterizes how anomaly magnitude propagates through each transform type τ.

We define the transfer coefficient ρ_τ as the expected ratio of output z-score to input z-score under a transformation of type τ, conditioned on a single anomalous input.

**DirectMap, Source, Calculate.** These transform types apply linear or near-linear operations to their inputs. The output preserves the input signal strength almost exactly:

ρ_{DirectMap} = ρ_{Calculate} = 1.0

**Filter (4σ clipping).** A clipping operation truncates input values that exceed c standard deviations. An input anomaly of magnitude z_in is passed through if z_in ≤ c and clipped otherwise. The expected output z-score is min(z_in, c), so:

ρ_{Filter} = min(1, c / z_in) ≈ 0.8   (for injection magnitude M = 8, clipping threshold c = 4σ)

This explains why downstream nodes of a clipped ETL node often show weaker anomaly signals than the source itself—the filter has attenuated the spike.

**Aggregate (mean of k independent inputs).** When one of k inputs is anomalous, the mean absorbs the anomaly by a factor of 1/k. By the variance of a sum, the standard deviation of the aggregated signal grows only as √k relative to the individual scale. Therefore:

ρ_{Aggregate} = 1/√k ≈ 0.45   (for k = 5, representative of sector-level aggregations)

This captures the well-known dilution effect in financial aggregations: a single corrupted instrument contributes only a fraction of its anomaly to the sector-level aggregate.

**Report (binary thresholding).** Report nodes apply threshold comparisons (e.g., VaR99 > limit), converting a continuous risk metric into a binary flag. This discretization destroys the continuous anomaly signal almost entirely:

ρ_{Report} ≈ 0

### 4.2.2 Derivation of the Optimal Attenuation Coefficient α*

APA-RCA uses a hop-distance-based attenuation factor α^(hop(v)/max_hop) to discount nodes that are further from the target. The scalar α = 0.7 was originally selected empirically following the RWR literature. The ATF framework provides a principled post-hoc justification.

Along a typical five-layer pipeline path from a source node to a report node, a signal traverses roughly one node of each type: DirectMap → Filter → Aggregate → Calculate → Report. Ignoring the Report terminal (which destroys the signal), the geometric mean of the three intermediate transfer coefficients is:

α* = (ρ_{DirectMap} × ρ_{Filter} × ρ_{Aggregate})^{1/3}
   = (1.0 × 0.8 × 0.45)^{1/3}
   ≈ 0.72

The empirically chosen value α = 0.7 coincides with this theoretical prediction to within 3%. This agreement is not a coincidence: the attenuation parameter is encoding, implicitly, the rate at which anomaly signals decay along the pipeline depth. The ATF framework transforms this from an opaque hyperparameter into a domain-theoretically grounded constant.

### 4.2.3 The φ_peer Temporal Discriminant for A3/A5 Disambiguation

Anomaly types A3 (feature calculation error, such as a wrong rolling window) and A5 (downstream masking, where upstream clipping suppresses a spike) produce superficially similar anomaly score patterns: both manifest at feature-layer nodes with moderate anomaly scores. Standard univariate anomaly scores cannot distinguish them. We introduce the **peer temporal fraction** φ_peer as a distribution-free discriminant.

For node v with peer group P(v), define the per-timestep peer deviation indicator:

I_peer(v, t) = 𝟙[s_peer(v, t) > θ_peer]

where s_peer(v, t) is the peer deviation score at time t and θ_peer is a fixed percentile threshold. The peer temporal fraction is:

φ_peer(v) = (1 / T_warm) × Σ_{t ∈ T_warm} I_peer(v, t)

where T_warm is the warm time series (excluding the rolling-window initialization prefix). The key discriminating property:

- **A3 (code bug, persistent):** A wrong rolling window (e.g., 2-day instead of 20-day) applies throughout the entire time series. The node's statistical behavior deviates from peers at nearly every time step: φ_peer ≈ 0.7–0.9.

- **A5 (spike propagation, transient):** A clipped upstream spike propagates only during the spike window (approximately 20 timesteps). Outside this window, the node behaves normally relative to peers: φ_peer ≈ 0.05–0.37.

The two distributions are non-overlapping, and the threshold φ_peer > 0.5 cleanly separates A3 from A5. Critically, this discriminant is **distribution-free**: it relies only on the relative ranking of a node's peer deviation scores over time, not on any parametric model of the underlying distribution. It therefore applies equally to GBM-generated synthetic data and real market data with fat tails and volatility clustering.

## 4.3 Transition Matrix Construction

Let G^T = (V, E^T) denote the transposed FRLG, where an edge (u, v) ∈ E^T corresponds to (v, u) ∈ E in the original graph—that is, u is downstream of v, and v is upstream. In G^T, walking from u to v corresponds to walking backward from effect to cause.

The unnormalized weight of each edge (u → v) in G^T is:

w(u → v) = a(v) × β(τ_{u,v}) × α^(hop(v) / max_hop)

where:
- **a(v)** is the anomaly score of the upstream destination v. High-anomaly nodes attract more probability mass, encoding the prior that anomalous nodes are more likely to be on the causal path.
- **β(τ_{u,v})** is the domain prior weight for the transform type τ of edge (u, v): Aggregate = 1.5, Report = 1.3, Calculate = 1.2, DirectMap = 1.0, Filter = 0.8. The higher weight for Aggregate reflects the domain observation that aggregation bugs propagate their errors to many downstream consumers—a node that aggregates N instruments affects N downstream risk metrics.
- **α^(hop(v)/max_hop)** is a mild distance-based decay. Nodes further from the target are assigned slightly lower weights, reflecting the weaker prior probability that a distant ancestor is the true root cause. As shown in Section 4.2.2, α = 0.7 corresponds to the geometric mean transfer coefficient across a typical pipeline path.

The column-normalized transition matrix P is obtained by dividing each column by its sum:

P_{ij} = w(j → i) / Σ_k w(k → i)

This ensures each column of P sums to 1, making the matrix stochastic. Nodes with anomaly score zero have zero weight as destinations; the walk is guided entirely by the anomaly signal landscape.

## 4.4 Restart Distribution and RWR Iteration

### 4.4.1 Anomaly-Weighted Restart Distribution

Standard RWR restarts exclusively at the target node t, with restart vector e concentrated at the single position e[t] = 1. This works well when the anomaly signal is strong and concentrated near t's direct ancestors. In practice, however, APA-RCA must handle cases where the true root cause r* is several hops upstream with only a moderate anomaly score, while other high-scoring nodes are unrelated to t's lineage.

To address this, APA-RCA uses a **distributed restart distribution**:

e[t] = 0.7   (primary: restart strongly at the target node)
e[v] ∝ 0.3 × a(v),  for all v ≠ t with a(v) > 0.3

The secondary restart mass is distributed proportionally to anomaly scores across all sufficiently anomalous non-target nodes, then renormalized so that e sums to 1. This serves two purposes. First, the primary restart at t anchors the walk in the neighborhood of t's ancestors. Second, the secondary restarts inject probability mass near detected anomalous nodes throughout the pipeline, which is especially valuable for A3 and A6 anomaly types where the true root cause's anomaly score is moderate and the walk might otherwise miss it entirely.

### 4.4.2 Convergence Analysis

The RWR iteration proceeds as:

r^(k+1) = (1 − γ) × P × r^(k) + γ × e

Since P is a column-stochastic matrix and γ > 0, the Perron-Frobenius theorem guarantees that the sequence {r^(k)} converges geometrically to a unique stationary distribution r*, with convergence rate bounded by (1 − γ). For γ = 0.15, convergence to ε = 10^{−8} in L₁ norm requires fewer than 40 iterations in all tested configurations. In practice, we observe convergence within 20–35 iterations for pipelines up to 100 nodes.

**Computational complexity.** Each iteration requires one matrix-vector multiplication: O(|E|) for sparse P, O(|V|²) for dense. For our largest configuration (|V| ≈ 100, |E| ≈ 300), the dense computation amounts to 10,000 multiply-add operations per iteration—well within the sub-millisecond latency requirement. The full RWR computation (K iterations at K < 50) has complexity O(K × |E|).

## 4.5 Composite Scoring with Ancestor Constraint

The RWR stationary distribution r*(v) quantifies how much of the random walk's probability mass arrives at node v during backward traversal from t. This reflects structural proximity but does not directly encode the anomaly signal that drove the traversal. The final ranking combines both signals and applies domain-specific corrections.

For each node v ≠ t, the composite score is:

score(v) = [0.4 × r*(v) + 0.6 × a(v)] × ancestor_factor(v) × layer_bonus(v) × β(v)

The four components are as follows.

**Score fusion (0.4 RWR + 0.6 anomaly).** The anomaly score receives the dominant weight (0.6) because it is the primary signal distinguishing the true root cause from irrelevant nodes. The RWR score (0.4) provides graph-structural disambiguation for cases where multiple nodes have similar anomaly scores—the walk tends to assign higher stationary probability to nodes that are both anomalous and structurally central to the backward traversal. Ablation experiments confirm that reversing this weighting (0.6 RWR + 0.4 anomaly) reduces Top-5 accuracy by approximately 8 percentage points.

**Ancestor constraint.** The structural correctness property of DAG-based RCA is enforced here:

ancestor_factor(v) = 1.0   if v ∈ ancestors(t)
ancestor_factor(v) = 0.05  if v ∉ ancestors(t)

Non-ancestor nodes receive a 95% penalty, effectively removing them from the ranking while preserving the relative ordering among ancestors. This constraint is the most impactful structural contribution of APA-RCA; without it, the walk distributes probability mass across the full graph and the ranking degrades to near-random performance for large pipelines.

**Layer preference bonus.**

layer_bonus(v) = 1 + 0.15 × (5 − layer(v))

This assigns a bonus of 1.60 to Layer-1 (Source) nodes and 1.15 to Layer-4 (Risk) nodes, encoding the domain prior that root causes in financial risk pipelines typically originate at the data ingestion or ETL layer. The coefficient 0.15 is robust: varying it in the range {0.10, 0.15, 0.20} produces no change in Top-5 accuracy; its primary effect is on Top-1 rank disambiguation between a source node and a feature node with similar composite scores.

**Transform type weight β(v).** The same type-based prior used in the transition matrix is applied to the final score, providing a consistent domain weighting throughout the algorithm.

## 4.6 Schema-Constraint RCA as a Complementary Module

The **SchemaConstraintRCA** module addresses a structural weakness of the probabilistic APA-RCA walk: anomaly type A4 (aggregation error, where an instrument is omitted from an aggregation set) produces a subtle, diffuse signal that the rolling z-score detector consistently misses. However, A4 anomalies leave an unambiguous structural signature: the actual number of inputs feeding into an aggregation node falls below the declared expected count.

The module operates as a deterministic O(|V|) pass over the pipeline:

1. For each aggregation node n with a declared `expected_input_count` attribute, compare the attribute to `len(n.input_columns)` (the number of columns actually consumed at runtime).
2. If `actual_inputs < expected_inputs`, record a constraint violation with severity = expected − actual.
3. If one or more violations are found, sort by severity (descending) and return an `RCAResult` with the violating nodes ranked first.
4. If no violations are found, return `None` and defer to APA-RCA.

This cascade design ensures that A4-type anomalies bypass the probabilistic walk entirely and receive a deterministic, 100% Top-1 accurate diagnosis on RSHB (see Chapter 6). The SchemaConstraintRCA module operates in O(|V|) time with no graph traversal, making it effectively zero-cost relative to the full APA-RCA run.

A practical limitation of this module is that it requires `expected_input_count` to be manually maintained as a pipeline metadata attribute. In large production environments with hundreds of aggregation nodes, keeping this metadata consistent with evolving pipeline specifications is a non-trivial operational burden. The ML-enhanced pipeline (Chapter 5) achieves higher overall accuracy without this requirement, at the cost of learned rather than declarative knowledge.

## 4.7 Hyperparameter Sensitivity Analysis

APA-RCA has four primary hyperparameters. We analyze each in terms of its effect on Top-5 accuracy and MRR across the synthetic benchmark.

**Restart probability γ ∈ {0.10, 0.15, 0.20, 0.25}.** Varying γ over this range produces a maximum Top-5 accuracy difference of 1.8 percentage points. The algorithm is highly robust to γ in the range [0.10, 0.25]. We use γ = 0.15, following the standard convention in the RWR literature (Tong et al., 2006).

**Attenuation coefficient α ∈ {0.5, 0.6, 0.7, 0.8, 0.9}.** At the tested pipeline scales (maximum path depth 4–5 hops), α contributes negligibly to Top-5 accuracy differences, as the maximum discount factor α^1 is always modest. Its role becomes more significant in deeper pipelines where distant ancestors must be meaningfully discounted relative to proximate ones. We use α = 0.7, which the ATF framework independently validates as the theoretically motivated choice.

**Score fusion weights (0.4/0.6).** Reversing to 0.6 RWR + 0.4 anomaly reduces Top-5 accuracy by approximately 8 percentage points. The anomaly score is the dominant signal; the RWR component provides supplementary structural disambiguation. The 0.4/0.6 split was determined by grid search over {0.2, 0.4, 0.6, 0.8} and validated on a held-out fold.

**Layer bonus coefficient ∈ {0.10, 0.15, 0.20}.** Top-5 accuracy is constant across this range. The coefficient primarily affects Top-1 accuracy in cases where a source-layer node and a feature-layer node have nearly identical composite scores—the bonus breaks the tie in favor of the source layer, which is the more common root cause location in financial data pipelines.

---

# Chapter 5. The ML-Enhanced Pipeline

## 5.1 Design Philosophy: Augmentation, Not Replacement

The APA-RCA algorithm developed in Chapter 4 provides a principled, graph-structural mechanism for root cause localization. It achieves strong performance on anomaly types with clear, detectable signals (A1 source corruption: 86.7% Top-1 on RSHB; A7 coincidental: 96.7% Top-1) and sub-millisecond query latency at all tested scales. Nevertheless, two fundamental limitations constrain its ceiling performance.

**Limitation 1 — Statistical detector brittleness under distribution shift.** The five-signal anomaly detector (Chapter 3) is built on rolling z-scores, peer deviation, and segment variance—all statistics calibrated against the empirical distribution of the node's own time series or its peer group. These statistics are effective when the normal-regime distribution is approximately stationary and Gaussian, as in GBM-generated synthetic data. Real financial market data, however, exhibits fat tails (leptokurtosis), volatility clustering (GARCH effects), abrupt regime changes, and cross-instrument correlation asymmetries. These properties can cause the statistical detector to produce anomaly score distributions that look different from those it produces on synthetic data, even for the same underlying causal pattern. This distribution shift degrades the quality of the signal that APA-RCA relies on.

**Limitation 2 — Hand-tuned scoring weights.** The final composite score in APA-RCA involves three manually specified weights: the 0.4/0.6 RWR-to-anomaly ratio, the transform-type β values, and the layer bonus coefficient. These were determined through a combination of domain reasoning and grid search on synthetic data. They represent fixed priors that may not be optimal for all anomaly types or pipeline topologies. A learning-based approach could discover better-calibrated weights from data.

The ML-enhanced pipeline addresses both limitations by integrating two learned components around the existing APA-RCA core:

- A **Variational Autoencoder (VAE)** as the anomaly scorer, trained unsupervised on normal pipeline time series. The VAE learns a compact latent representation of normal behavior and scores anomalies by reconstruction error—a manifold-based signal that is inherently more stable under distribution shift than statistical rules.

- A **Graph Neural Network (GNN) re-ranker**, trained on synthetic injection locations and applied to real data without fine-tuning. The GNN jointly learns from node features and graph topology to re-score APA-RCA's Top-K candidates and elevate the true root cause.

A key architectural commitment is that **ML augments APA-RCA rather than replacing it.** The VAE operates upstream of APA-RCA as a superior anomaly scorer; the GNN operates downstream as a learned re-ranker. APA-RCA's graph traversal logic—transposed RWR, ancestor constraint, layer bonus—is preserved in its entirety. This design is not merely conservative: Chapter 6 shows empirically that replacing APA-RCA with a GNN alone (without its RWR and structural reasoning) is outperformed by the APA-RCA baseline, while augmenting APA-RCA with both VAE and GNN yields the best results by a significant margin.

## 5.2 Conditional VAE (C-VAE) Anomaly Scorer

### 5.2.1 Architecture

A Conditional Variational Autoencoder (C-VAE) extends the standard VAE by conditioning the encoder and decoder on auxiliary metadata. Where a standard VAE learns a single reconstruction manifold for all inputs, the C-VAE learns **type-specific and layer-specific manifolds**: the reconstruction error of a node is measured against the expected normal behavior for *that transform type* and *that pipeline layer*, not against a pooled average across all node types. This is the key architectural motivation from the ATF framework—different transform types (Source, ETL, Feature, Risk, Report) have fundamentally different normal behavior profiles, and a single manifold conflates them.

After training on normal time series windows, the C-VAE assigns high reconstruction error to abnormal inputs—those that deviate from the type- and layer-appropriate learned manifold. This produces a more discriminating anomaly signal than an unconditional VAE, particularly for nodes whose type-specific behavior would appear unusual relative to the global average even when normal (e.g., high-volatility Source nodes relative to smoothed Report nodes).

The conditioning vector **t** ∈ ℝ^{N_COND} is the concatenation of a transform-type one-hot encoding and a pipeline-layer one-hot encoding:

```
Type one-hot:   [source, direct_map, filter, calculate, aggregate, report] ∈ {0,1}^6
Layer one-hot:  [layer_1, layer_2, layer_3, layer_4, layer_5]              ∈ {0,1}^5
t = [type_onehot ‖ layer_onehot]   ∈ {0,1}^{11}    (N_COND = 11)
```

The C-VAE architecture concatenates **t** to the encoder input and decoder input at each stage:

```
Input:    x ∈ ℝ^W         (W = 20, z-normalized window)
          t ∈ {0,1}^11     (transform-type + layer conditioning vector)

Encoder:  [x ‖ t] → fc(128, ReLU) → fc(64, ReLU) → [μ ∈ ℝ^8, log σ² ∈ ℝ^8]
          Encoder input dimension: W + N_COND = 20 + 11 = 31

Reparameterization:  z = μ + σ ⊙ ε,   ε ~ N(0, I)

Decoder:  [z ‖ t] → fc(64, ReLU) → fc(128, ReLU) → fc(W) → x̂
          Decoder input dimension: latent_dim + N_COND = 8 + 11 = 19

Loss:     L(x, x̂, t) = MSE(x, x̂) + β_KL × KL[N(μ, σ²) ∥ N(0, I)]
```

**Why conditioning improves over the unconditional VAE.** The standard VAE trains a single latent space that must represent all six transform types simultaneously. Source nodes (high-volatility raw prices) and Report nodes (binary threshold outputs) occupy opposite extremes of the variance spectrum. A single reconstruction manifold calibrated to their mixture assigns elevated reconstruction error to normal-but-high-variance Source nodes and near-zero error to low-variance Report nodes—creating a systematic scoring bias that does not reflect true anomaly status.

The C-VAE resolves this by conditioning on the transform type and layer: the encoder maps each window to a type-conditioned latent space, and the decoder reconstructs from the type-conditioned latent code. Reconstruction error is therefore measured relative to the expected distribution for *that specific node type at that layer*, not against a global average. This is a structural improvement aligned with the ATF framework's central insight: anomaly propagation and detection are fundamentally type-dependent.

### 5.2.1a Hybrid TSD-CVAE: Type-Specific Decoder Extension

The v3 architecture extends the C-VAE with **type-specific decoder sub-networks**. Rather than sharing a single decoder across all transform types (with conditioning concatenated to the input), the Hybrid TSD-CVAE routes the latent code z to one of six type-specific decoders, each dedicated to reconstructing normal behavior for a single transform type.

Critically, the Hybrid TSD-CVAE **retains the conditioning vector in the decoder** — each type-specific decoder receives [z ‖ t] as input, preserving pipeline-layer information that would be lost under pure routing. This hybrid design combines structural separation (each decoder learns an independent reconstruction pathway) with information preservation (the layer one-hot within t provides depth context even within a single type).

```
Encoder (shared):  [x ‖ t] → h → (μ, log σ²)           [identical to C-VAE]

Decoder (per-type): Dec_τ([z ‖ t]) → x̂_τ                [6 separate decoders]
                    τ = argmax(t[:6])                     [route by transform type]
                    Decoder input dimension: L + N_COND = 8 + 11 = 19
```

The empirical motivation for this extension is the v2 C-VAE's failure mode on A4 (aggregation error): a single shared decoder struggles to learn distinct reconstruction manifolds for Aggregate nodes (which produce smoothed, variance-reduced series) versus Source or Calculate nodes (which preserve raw dynamics). By dedicating a decoder to each type, the Hybrid TSD-CVAE learns sharper type boundaries, improving A4 Top-1 from 53.3% (v2) to 70.0% (v3).

**Why MLP-C-VAE over attention-based or graph-conditioned alternatives.** Three domain constraints favor this design.

First, the setting is **strictly unsupervised**: no anomaly labels are available at training time, ruling out discriminative architectures. The C-VAE requires only normal time series windows and their associated node metadata.

Second, the scorer must be **pipeline-size agnostic**: a single trained model must score nodes from pipelines of all sizes (small ~45 nodes, medium ~63 nodes, large ~100+ nodes). Since each input window is a fixed 20-dimensional vector and the conditioning vector is fixed at 11 dimensions, the C-VAE input dimension is constant regardless of pipeline size.

Third, the scorer must produce **distribution-stable continuous scores** for GNN node features. The C-VAE's conditioning on transform type, rather than on distributional statistics of the training data, makes reconstruction error semantically stable across the synthetic-to-real distribution shift.

### 5.2.2 Training Procedure

**Training data.** We generate 60 anomaly-free pipelines (20 per size configuration: small, medium, large) and record the time series output of every node over T = 252 trading days. Applying a sliding window of length 20 with stride 1 yields approximately 898,000 normal windows in total. All windows are z-normalized (mean subtracted, divided by standard deviation) before input to the VAE.

**Rationale for z-normalization.** Pipeline nodes operate at vastly different absolute scales: a position size node may output values in the range [55,000, 100,000] shares while a return node outputs values in the range [−0.05, 0.05]. Z-normalization eliminates this scale heterogeneity, allowing the single VAE to learn a scale-invariant representation of normal time series shapes. Without normalization, the VAE would need to learn separate representations for each node type and scale, which would require far more data and a larger latent space.

**Training configuration.**

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Window size | 20 | Matches one trading month; captures rolling-window artifacts |
| Latent dimension | 8 | Approximately matches the number of distinct node type-metric combinations |
| Hidden dimension | 64 | Balances representational capacity and training stability on CPU |
| Epochs | 80 | Training loss converges by epoch ~60 in all runs |
| Batch size | 256 | Effective for the ~898K window dataset on CPU hardware |
| Optimizer | Adam, lr = 1e-3 | Standard default; no learning rate scheduling |

### 5.2.3 Inference and Score Normalization

At inference time, for each node v with output time series of length T, we compute reconstruction error over all length-20 windows using a stride of 1, then take the maximum over all windows as the raw anomaly score:

err_raw(v) = max_{t ∈ windows} MSE(x_t, VAE(x_t))

The maximum (rather than mean) aggregation is motivated by the single-fault assumption: an anomaly typically manifests for a contiguous period within the time series, and the maximum error over all windows captures the worst-case deviation from normal behavior.

The raw error is then converted to a calibrated anomaly score in (0, 1) using the error distribution estimated from the training set:

score_VAE(v) = sigmoid( (err_raw(v) − μ_err) / σ_err )

where μ_err and σ_err are the mean and standard deviation of reconstruction errors computed over all 898,000 training windows. This normalization ensures that a node whose reconstruction error matches the typical training error receives a score near 0.5 (sigmoid(0) = 0.5), while a node with much higher error receives a score near 1.0. The resulting scores are used directly as the anomaly input a(v) to the APA-RCA transition matrix, replacing the multi-signal z-score detector output.

## 5.3 GNN Re-Ranker

### 5.3.1 Design Rationale

APA-RCA's composite score (Section 4.5) combines RWR proximity and anomaly score using fixed weights. The GNN re-ranker replaces this fixed combination with a learned function that jointly processes node features and graph topology to predict the probability that each candidate node is the true root cause.

The key insight motivating a GNN over a simple classifier is that **root cause localization is inherently a graph-contextualized decision**. Whether a given node is likely to be the root cause depends not only on its own features (anomaly score, layer position) but also on those of its neighbors: if a node's anomaly score is high but its upstream neighbors are even more anomalous, it is likely a propagation node rather than an origin. Conversely, if a node's anomaly score exceeds what is explained by its upstream anomalies (high causal excess), it is a strong origin candidate. A GNN captures these relational patterns by aggregating information from neighboring nodes through message-passing layers.

### 5.3.2 Subgraph Construction

Given APA-RCA's output of Top-K candidate nodes (K = 10), we construct an induced subgraph consisting of the top-K candidates plus all of their 1-hop neighbors in the original FRLG (both upstream and downstream). Edge directions are preserved from the original FRLG. This subgraph contains on the order of 15–30 nodes for the medium pipeline configuration, making the GNN computation extremely fast (microseconds per subgraph).

The 1-hop neighborhood expansion is important for two reasons. First, it allows the GNN to observe each candidate node's immediate upstream ancestors and downstream consumers—the local causal context that determines whether the node is an origin or a propagation point. Second, it ensures that the subgraph is connected even for isolated candidates that might otherwise have no neighbors in the top-K list alone.

### 5.3.3 Node Feature Vector (8 Dimensions)

Each node in the subgraph is assigned an eight-dimensional feature vector:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| [1] RWR score | Stationary distribution value from APA-RCA's backward walk | Encodes graph-structural proximity to the target |
| [2] C-VAE anomaly score | Reconstruction-error-based score from Section 5.2 | Primary anomaly signal; the critical link between C-VAE and GNN |
| [3] Normalized layer position | Pipeline layer (1–5), normalized to [0, 1] | Source-layer nodes are a priori more likely to be root causes |
| [4] Ancestor indicator | Binary: 1 if node is an ancestor of the target, else 0 | Encodes the structural ancestor constraint learned from training data |
| [5] In-degree | Number of incoming edges in the FRLG | Low in-degree (especially 0 or 1) indicates a source-layer candidate |
| [6] Out-degree | Number of outgoing edges in the FRLG | High out-degree indicates broad downstream impact if corrupted |
| [7] Candidate indicator | Binary: 1 if node is in APA-RCA's Top-K list, else 0 | Distinguishes primary candidates from neighborhood context nodes |
| [8] ATF-weighted path distance | Shortest graph distance from node to the target, normalized by ATF penalty | Encodes proximity in the causal sense: nodes closer to the observed anomaly via high-transfer-coefficient paths are more strongly penalized if distant |

**Feature [8]: ATF-weighted path distance.** The graph distance from each subgraph node v to the observed anomaly target node t is computed on the directed FRLG using Dijkstra's algorithm (or BFS for unweighted distance). The distance is normalized by a penalty constant (penalty_dist = 5.0): `feats[v, 7] = dist(v, t) / penalty_dist`. Nodes unreachable from t (infinite distance) receive distance = 1.0 (the maximum normalized penalty). This feature encodes the ATF insight that causal responsibility decays with graph distance: a node three hops removed from the target carries much less anomaly mass than a direct parent. Unlike the hop-count used in the APA-RCA attenuation formula, this is a learnable feature: the GNN can discover whether short-distance nodes are more likely to be root causes and how strongly to weight distance versus anomaly score in its final ranking.

Feature [2], the C-VAE anomaly score, is the architecturally critical link between the two ML components. The C-VAE score flows into APA-RCA's transition matrix (influencing the RWR stationary distribution, Feature [1]) and is then independently reused as a GNN node feature. This dual use creates a coherent information pathway: the C-VAE informs both the APA-RCA graph walk and the subsequent GNN re-ranking from the same learned anomaly representation.

### 5.3.4 ResGCN Architecture with ATF-Weighted Edges

We use a two-layer **Residual Graph Convolutional Network (ResGCN)** with ATF-weighted adjacency, replacing the standard GCN of early iterations. Two structural changes distinguish ResGCN from the baseline GCN:

**Change 1 — ATF-weighted adjacency.** In the standard GCN, Â is the symmetrically normalized binary adjacency matrix: edges are either present (1) or absent (0). In ResGCN, we replace binary edge weights with ATF transfer coefficients ρ_τ, where τ is the transform type of the *target* node of each edge (the node receiving information in the message-passing step):

```
ATF edge weight: A[v, u] = ρ_{τ(u)}   for edge (u → v) in the FRLG

ρ_{source}      = 1.0
ρ_{direct_map}  = 1.0
ρ_{filter}      = 0.8
ρ_{calculate}   = 1.0
ρ_{aggregate}   = 0.45
ρ_{report}      = 0.1
```

The resulting ATF-weighted matrix is then row-normalized so each row sums to 1. This encodes the domain theory directly into the message-passing structure: a source or calculate node receives full upstream signal (ρ = 1.0), a filter node receives attenuated signal (ρ = 0.8), and an aggregation node receives heavily diluted signal (ρ = 0.45)—consistent with the ATF framework's analysis of anomaly propagation in Section 4.2.1. The GNN's learned aggregation weights are thus grounded in the same theoretical framework as APA-RCA's transition matrix, creating end-to-end ATF consistency across the full pipeline.

**Change 2 — Residual skip connection with LayerNorm.** Deep GCN stacking suffers from over-smoothing: repeated neighborhood aggregation gradually homogenizes node representations, erasing the individual anomaly score distinctions that the C-VAE worked to establish. The residual connection prevents this by adding the original node features (linearly projected) to the aggregated output before the activation:

```
Input:          h^0 ∈ ℝ^{|V_sub| × 8}   (8-feature node matrix)

ResGCN Layer:   agg = A_ATF × h^0 × W_conv       W_conv ∈ ℝ^{8 × 32}
                skip = h^0 × W_skip               W_skip ∈ ℝ^{8 × 32}
                h^1 = ReLU(LayerNorm(Dropout(agg) + skip))

Output layer:   logit = h^1 × W_out               W_out ∈ ℝ^{32 × 1}
Prediction:     p(root cause | v) = sigmoid(logit_v)
```

The skip connection `W_skip(h^0)` preserves the original C-VAE anomaly score and graph-structural features even after neighborhood aggregation. LayerNorm stabilizes training by normalizing across the feature dimension before activation. Dropout (p = 0.3) is applied only to the aggregation path, not the skip path, ensuring that individual node features always flow through to the output.

**Why these changes are architecturally significant.** The ATF-weighted adjacency is not a generic hyperparameter change—it replaces the implicit binary connectivity assumption with an explicit model of how strongly each transform type transmits anomaly information to its downstream neighbors. The residual connection directly addresses the over-smoothing pathology that standard GCNs exhibit in graph RCA tasks where the distinguishing signal (individual node anomaly score) must be preserved across aggregation layers. Together, these changes encode the domain theory of the ATF framework into both the message-passing weights and the information-preservation mechanism of the GNN.

Training is supervised with binary cross-entropy loss: positive labels are assigned to the true root cause node (the injection point in each synthetic trial), negative labels to all other candidates. The class imbalance (1 positive vs. 9 negatives per subgraph) is mild and does not require reweighting.

**Training configuration.**

| Hyperparameter | Value |
|----------------|-------|
| Hidden dimension | 32 |
| Dropout | 0.3 (aggregation path only) |
| Epochs | 150 |
| Optimizer | Adam, lr = 5×10^{−4} |
| Loss | Binary cross-entropy |
| Top-K candidates | 10 |
| Edge weights | ATF transfer coefficients ρ_τ |

### 5.3.5 Synthetic Label Acquisition and Cross-Domain Transfer

**The labeling strategy.** A fundamental challenge in applying supervised learning to financial pipeline RCA is the absence of ground-truth root cause labels in real production data. Financial institutions rarely log which specific pipeline node caused a given data quality incident, and when such logs exist, they are commercially sensitive and inaccessible to researchers.

We resolve this challenge by training the GNN exclusively on **synthetic data with injected anomalies**, where the injection node is known by construction and serves as the perfect ground-truth label. From 630 synthetic trials (7 anomaly types × 3 pipeline sizes × 30 trials), we construct one GNN subgraph per trial, yielding up to 630 labeled training examples. Five-fold cross-validation with stratified splits (by anomaly type × pipeline size) is used to evaluate generalization within the synthetic domain.

**Cross-domain transfer without fine-tuning.** The trained GNN is applied directly to RSHB (real market data, 210 trials) without any parameter updates or fine-tuning on real-data examples. This is a strict cross-domain evaluation: the model has seen only synthetic data during training.

The key to why this transfer succeeds lies in what the GNN actually learns. Real market data and synthetic GBM-generated data have fundamentally different distributional properties—the GNN cannot reliably learn to distinguish "normal-looking" from "anomalous-looking" sequences if those sequences come from different distributions. What it *can* learn are **topological root cause patterns** that are invariant across both domains:

> A true root cause node tends to be in a low pipeline layer (Source or ETL), be an ancestor of the observed anomaly, have a low in-degree (few upstream dependencies), and have a higher VAE anomaly score than its immediate descendants.

These patterns hold in synthetic pipelines and real pipelines alike because the five-layer pipeline topology—Source → ETL → Feature → Risk → Report—is shared between them. The structural graph regularities, not the data distribution, are the transferable knowledge. This is the fundamental reason why VAE anomaly scores (which are stable across distribution shifts) work as GNN features while z-score-based anomaly signals (which are not) fail catastrophically in cross-domain transfer (Section 6.6).

## 5.4 Comparative Analysis: VAE vs. Isolation Forest vs. One-Class SVM

To rigorously justify the choice of VAE as the anomaly scorer—as opposed to other standard unsupervised anomaly detection methods—we conduct a controlled comparison experiment using Isolation Forest (IF) and One-Class SVM (OCSVM) as alternative scorers within the same pipeline.

### 5.4.1 Methods Under Comparison

**Isolation Forest** (Liu et al., 2008) estimates anomaly scores by randomly partitioning the feature space using isolation trees. Normal observations require more partitions to isolate (deep in the tree), while anomalies are isolated with fewer partitions (shallow). Each time series window is treated as an independent point; no temporal structure is exploited.
- Configuration: n_estimators = 100, max_train_windows = 50,000

**One-Class SVM** (Schölkopf et al., 2001) fits a hypersphere or hyperplane in a high-dimensional kernel feature space that tightly encloses the normal data. Points outside this boundary are classified as anomalies. Like IF, OCSVM treats each window as an independent observation without temporal modeling.
- Configuration: ν = 0.1, n_components = 100 (PCA dimensionality reduction), max_train_windows = 20,000

**VAE** — as described in Section 5.2.

For each scorer, the full experiment pipeline is: (1) train the scorer on 60 normal pipelines, (2) run 630 synthetic trials using the scorer's outputs as anomaly signals, (3) collect GNN training samples, (4) train the GNN, (5) evaluate on RSHB 210 trials.

### 5.4.2 Fundamental Methodological Distinction

The critical difference between the three methods can be stated precisely:

| Dimension | IF / OCSVM | VAE |
|-----------|------------|-----|
| Approach | Boundary-based: decision surface in feature space | Manifold-based: generative model of normal data |
| Temporal modeling | None (each timestep treated independently) | Explicit (20-step windows capture temporal patterns) |
| Distribution shift robustness | Low (boundary calibrated on synthetic distribution) | High (reconstruction error is relative to learned normal manifold) |
| GNN feature quality | Poor (scores shift discontinuously across domains) | Good (smooth, continuous signal stable across domains) |

IF and OCSVM score each time series window as an independent point in feature space. The decision boundary they learn is calibrated against the distributional properties of GBM-generated synthetic data. When this boundary is applied to real market data—which has a fundamentally different joint distribution due to fat tails, volatility clustering, and cross-asset correlations—the boundary's calibration becomes invalid. Nodes that the IF/OCSVM scores as "anomalous" on real data are not necessarily the same nodes that were "anomalous" during synthetic training; the score distributions shift unpredictably.

The GNN, trained on synthetic data where IF/OCSVM scores had a particular relationship to root cause locations, then encounters real data where the same structural relationship no longer holds between scores and locations. The result is that the GNN's predictions on real data are essentially uncorrelated with the true root cause, producing near-zero Top-1 accuracy.

The VAE's reconstruction error, by contrast, is always defined relative to the model's own learned representation of normal behavior. A time series window with unusual temporal dynamics will have high reconstruction error regardless of whether it comes from GBM or real market data—because the VAE learned what "normal dynamics" look like, not what "normal distribution parameters" look like. This property makes the VAE score distribution more stable across the distribution shift, and the GNN's learned association between score patterns and root cause locations remains approximately valid on real data.

## 5.5 Full ML Pipeline Integration

The complete ML-enhanced pipeline (v3: Hybrid TSD-CVAE+APA-RCA+ResGCN) integrates all three components—C-VAE, APA-RCA, and ResGCN—in a sequential architecture where each component's output feeds into the next:

```
Stage 1: C-VAE Anomaly Scoring
  Input:  All node time series from the executed pipeline;
          node metadata (transform type, pipeline layer)
  Action: Score each node using trained Hybrid TSD-CVAE (Section 5.2),
          conditioning on [type_onehot ‖ layer_onehot] ∈ ℝ^11
  Output: a_TSD(v) ∈ (0, 1) for each node v

Stage 2: FRLG Construction
  Input:  Pipeline structure + a_TSD(v) scores
  Action: Build FRLG; attach Hybrid TSD-CVAE scores as node attributes
  Output: FRLG with populated anomaly scores

Stage 3: APA-RCA Graph Walk
  Input:  FRLG with a_TSD(v), target node t
  Action: Transposed RWR with anomaly-weighted transitions
          + ancestor constraint + composite scoring
  Output: Top-K (K = 10) candidate list with RWR scores

Stage 4: ResGCN Re-Ranking
  Input:  Top-K candidates, their 1-hop neighborhoods,
          8-feature vectors (including a_TSD, RWR score,
          and ATF-weighted path distance)
  Action: ResGCN with ATF-weighted adjacency + residual skip
          predicts root-cause probability per node
  Output: Re-ranked candidate list

Final Output: Ranked root cause candidates (r_1, r_2, ..., r_K)
```

**The triadic dependency.** The three stages are coupled in a non-trivial way, with the ATF framework threading through all levels:
- The Hybrid TSD-CVAE score a_TSD(v) drives APA-RCA's transition matrix (Stage 3) and also appears directly as node Feature [2] in the ResGCN (Stage 4).
- APA-RCA's RWR stationary distribution r*(v) appears as node Feature [1] in the ResGCN (Stage 4).
- The ResGCN uses ATF transfer coefficients as edge weights (Section 5.3.4) and the ATF-weighted path distance as Feature [8]—the same ρ_τ values that justify APA-RCA's attenuation parameter appear in three places: the APA-RCA transition matrix, the ResGCN adjacency, and the path-distance feature.
- The C-VAE conditions on transform type, the APA-RCA transition weights are typed (β_τ), and the ResGCN edge weights are typed (ρ_τ). This creates a domain-theoretically unified architecture where ATF type information flows through all three components.

This triadic dependency is why C-VAE and ResGCN form an inseparable integrated system. Removing the C-VAE (and falling back to z-score anomaly signals) causes the ResGCN to fail on real data despite identical architecture. Removing the ResGCN eliminates the learned re-ranking that produces the largest Top-1 accuracy gains. Removing APA-RCA removes the graph-structural reasoning that identifies the candidate set and computes the RWR scores that anchor the ResGCN's feature space. Each component is necessary; none is sufficient alone.

---

# Chapter 6. Experiments and Evaluation

## 6.1 Experimental Design

Our evaluation framework consists of four complementary experiments, each designed to answer a specific question about the system's capability, generalization, and component contributions.

### 6.1.1 Experiment A: Synthetic 5-Fold Cross-Validation

**Research question:** How well does the ML-enhanced pipeline generalize within the synthetic domain, and does the improvement over APA-RCA persist without overfitting?

**Protocol.** We execute 630 synthetic trials (7 anomaly types × 3 pipeline sizes × 30 trials per combination) using the SynFRP benchmark (Chapter 3). Trials are assigned to 5 stratified folds, stratified by (anomaly_type × pipeline_size) to ensure balanced representation in each fold. Training uses 504 trials (80%); evaluation uses 126 trials (20%). We report the mean and standard deviation of Top-1, Top-3, Top-5, and MRR across the 5 folds.

**Compared methods:** (1) APA-RCA v2 baseline, (2) z-score+APA-RCA+GNN (ablation: statistical z-score anomaly signals, same GNN architecture), (3) VAE+APA-RCA (ablation: VAE scoring without GNN re-ranking), (4) VAE+APA-RCA+GNN (v1 ML pipeline).

### 6.1.2 Experiment B: RSHB Cross-Domain Validation

**Research question:** Does the GNN, trained exclusively on synthetic data, generalize to real market data without fine-tuning?

**Protocol.** The GNN trained on all 630 synthetic trials is applied directly—with no parameter updates—to the Real-Source Hybrid Benchmark (RSHB). The RSHB uses real market data from Yahoo Finance: 756 trading days (approximately three years ending December 31, 2024) including Samsung Electronics, SK Hynix, and other KRX-listed instruments, along with US equities, FX rates, and Treasury yields. The pipeline structure (medium size: 5 stocks, 3 rates, 2 FX, 5 positions) is identical to the corresponding SynFRP configuration. Anomaly injection protocol and evaluation metrics are identical to SynFRP. A total of 210 trials are evaluated (7 anomaly types × 30 trials each).

**Significance.** This experiment constitutes the primary real-world validity test of the ML pipeline. It is a strict cross-domain evaluation: synthetic → real, no fine-tuning, no real labels used at any stage of training. The RSHB results reported in Section 6.5 are the central quantitative contribution of Chapter 5.

### 6.1.3 Experiment C: Ablation Study (z-score+GNN)

**Research question:** Is the VAE necessary? Would a GNN trained on standard statistical anomaly scores achieve comparable cross-domain performance?

**Protocol.** We substitute the VAE anomaly scorer with rolling z-score signals (the same detector used in APA-RCA v2 without the VAE enhancement). All other components—GNN architecture, training configuration, evaluation protocol—are held constant. The GNN trained on z-score features is then applied to the RSHB under the same conditions as Experiment B.

This ablation directly isolates the contribution of the VAE to cross-domain transfer performance.

### 6.1.4 Experiment D: Unsupervised Scorer Comparison (VAE vs. IF vs. OCSVM)

**Research question:** Is VAE the right unsupervised scorer? How do alternative unsupervised methods compare?

**Protocol.** Isolation Forest and One-Class SVM are trained on the same 60 normal pipelines as the VAE. For each scorer, the full pipeline—GNN sample collection from 630 synthetic trials, GNN training, RSHB evaluation (210 trials)—is executed independently. Results are compared to the VAE+GNN baseline.

## 6.2 Evaluation Metrics

**Top-K Accuracy (K = 1, 3, 5).** The fraction of trials in which the true root cause node appears within the top-K ranked candidates. Top-1 is the most operationally meaningful: it measures how often an engineer who investigates only the first recommendation resolves the incident without further search. Top-5 measures the coverage of a short candidate list.

**Mean Reciprocal Rank (MRR).** MRR = (1/N) Σᵢ 1/rank_i, where rank_i is the rank of the true root cause in trial i (0 if not found). MRR is a smooth, rank-sensitive metric that rewards finding the true root cause earlier even when it is not ranked first.

**Contextualizing Top-1 accuracy.** We note that interpreting 54.8% Top-1 accuracy requires domain context. This is a multi-class localization problem over a candidate set of 45–100+ pipeline nodes, not a binary classification. The random baseline for Top-1 accuracy is 1/N ≈ 1.6–2.2% for the medium pipeline (N ≈ 63). In this context, the APA-RCA v2 baseline of 32.9% and the ML-enhanced result of 54.8% both represent substantial improvements over chance. Moreover, 54.8% Top-1 accuracy means that in more than half of all incidents, the very first recommended node is the true origin—an operational efficiency that compounds across hundreds of incidents per year at a large financial institution.

The v2 pipeline (CVAE+APA-RCA+ResGCN) achieves 51.0% Top-1—statistically indistinguishable from v1 (p = 0.434)—while improving Top-3 to 74.8% (+9.1 pp, p = 0.043). The v3 pipeline (Hybrid TSD-CVAE+APA-RCA+ResGCN) further improves Top-3 to 80.0% and Top-5 to 83.3% with MRR 0.639, the best ranking quality among all evaluated methods. For operational settings where analysts review a short candidate list rather than acting on a single recommendation, the v3 Top-3 improvement translates directly to higher diagnostic efficiency.

## 6.3 Baseline APA-RCA Performance: Synthetic Benchmark

**Table 6.1. Main Results: 630 Trials (5-fold CV) Across All Anomaly Types and Pipeline Sizes (Synthetic)**

| Method | Top-1 | Top-3 | Top-5 | MRR | Avg. Latency (ms) |
|--------|-------|-------|-------|-----|-------------------|
| APA-RCA v2 | 1.7% | 43.3% | 51.7% | 0.237 | 0.73 |
| BFS Baseline | 1.7% | 1.7% | 10.0% | 0.076 | 0.65 |
| Vanilla RWR | 0.0% | 0.0% | 0.8% | 0.040 | 0.72 |
| Anomaly Score Only | **29.2%** | 45.0% | 69.2% | 0.430 | 0.01 |

APA-RCA outperforms BFS by +41.7 percentage points on Top-5 and Vanilla RWR by +50.9 percentage points, demonstrating the essential contribution of anomaly-weighted graph traversal over structure-only methods. All methods achieve sub-millisecond query latency, meeting the real-time operational requirement.

The Anomaly-Score-Only baseline achieves superior Top-1 and Top-5 accuracy in the synthetic setting, a finding that requires careful interpretation. In SynFRP, anomaly injection places the error directly at the root cause node, producing a near-maximal anomaly score there. The score-only baseline trivially ranks this highest-scoring node first. In production environments, however, root cause nodes are frequently *not* the highest-scoring nodes: downstream clipping operations attenuate source-layer spikes, rolling window smoothing spreads point anomalies across time, and concurrent operational noise creates false high-scoring nodes. Graph-guided traversal provides essential disambiguation precisely in these harder cases—which characterize real-world incidents but not our controlled synthetic injections. The RSHB results in Section 6.4 confirm this: APA-RCA substantially outperforms the score-only approach on real data.

**Table 6.2. Per-Anomaly-Type Results on the Medium Pipeline (Synthetic)**

| Method | A1 Top-5 | A2 Top-5 | A3 Top-5 | A4 Top-5 |
|--------|----------|----------|----------|----------|
| APA-RCA v2 | **100.0%** | **100.0%** | 10.0% | 0.0% |
| BFS Baseline | 0.0% | 0.0% | 10.0% | 0.0% |
| Vanilla RWR | 0.0% | 0.0% | 10.0% | 0.0% |

APA-RCA achieves perfect Top-5 recall for A1 (source corruption) and A2 (ETL logic error). The peer deviation signal clearly identifies ETL nodes as statistical outliers among peers running the same transformation on different instruments, and the causal excess signal confirms the anomaly originates there rather than propagating from upstream. A3 and A4 remain challenging; the error analysis in Section 6.9 examines why.

![**Figure 6.1.** Top-5 Accuracy by Anomaly Type and Method on the Synthetic Benchmark (Medium Pipeline, 30 Trials per Type). The grouped bar chart reveals that APA-RCA v2 achieves near-perfect recall on A1 and A2 while BFS and Vanilla RWR cannot distinguish causal origins on any type.](figures/fig1_anomaly_type_breakdown.pdf){width=\textwidth}

![**Figure 6.2.** Ablation Study: Contribution of Each APA-RCA Component to Top-K Accuracy and MRR (Medium Pipeline, 210 Runs per Variant). Removing any single component—transform-type weighting, hop-distance attenuation, or anomaly score integration—degrades both accuracy and MRR, confirming that each design principle is load-bearing.](figures/fig2_ablation.pdf){width=\textwidth}

![**Figure 6.3.** Scalability of APA-RCA v2: (a) Top-5 Accuracy and MRR Across Pipeline Sizes; (b) Query Latency vs. Node Count (210 Runs per Size). Accuracy is stable across small, medium, and large pipelines. Query latency remains well below the 2 ms operational SLA at all sizes.](figures/fig3_scalability.pdf){width=\textwidth}

![**Figure 6.4.** Overall Top-K Accuracy Comparison Across All Methods on the Synthetic Benchmark (SynFRP, 630 Trials). APA-RCA v2 achieves the best Top-3 and Top-5 accuracy among all non-ML methods, confirming the effectiveness of anomaly-propagation-aware graph traversal over structure-only baselines.](figures/fig4_overall_comparison.pdf){width=\textwidth}

## 6.4 RSHB Results: APA-RCA on Real Market Data

**Table 6.3. APA-RCA and Schema-Constraint RCA on RSHB (Real Market Data, 210 Trials)**

| Method | Top-1 | Top-3 | Top-5 | MRR |
|--------|-------|-------|-------|-----|
| APA-RCA v2 | 32.9% | 67.6% | **82.4%** | 0.503 |
| Schema-Constraint only | 17.6% | 22.9% | 28.6% | 0.266 |

APA-RCA v2 achieves 32.9% Top-1 and 82.4% Top-5 on RSHB. The substantially higher absolute accuracy compared to the synthetic benchmark (1.7% / 51.7%) reflects a key difference in anomaly injection on real versus synthetic data: real market time series have higher natural variability, which causes injected anomalies of the same magnitude (M = 8) to produce more strongly anomalous nodes relative to the background. The high Top-5 accuracy (82.4%) confirms that APA-RCA is an effective candidate generator even before ML enhancement.

Schema-Constraint alone scores only 17.6% overall because it is activated exclusively for A4 anomalies (14.3% of all trials) and produces no useful output for the remaining six anomaly types. However, its 100% Top-1 accuracy on A4 is a unique capability that the probabilistic APA-RCA cannot replicate, as shown in the per-type breakdown below.

**Table 6.4. Per-Anomaly-Type RSHB Results**

| Anomaly Type | APA-RCA v2 Top-1 | Schema-Constraint Top-1 |
|--------------|-----------------|------------------------|
| A1 Source Corruption | 86.7% | 6.7% |
| A2 ETL Logic Error | 0.0% | 0.0% |
| A3 Feature Calculation Error | 0.0% | 0.0% |
| A4 Aggregation Error | 0.0% | **100.0%** |
| A5 Downstream Masking | 13.3% | 6.7% |
| A6 Multi-Source | 33.3% | 3.3% |
| A7 Coincidental | 96.7% | 6.7% |

The complementary profiles of APA-RCA and Schema-Constraint are striking: where one excels, the other struggles. APA-RCA dominates for A1 (86.7%) and A7 (96.7%) while Schema-Constraint is uniquely effective for A4 (100%). This motivates the hybrid and ML-enhanced alternatives analyzed in Sections 6.5–6.8.

## 6.5 ML-Enhanced Results: RSHB Performance

The central results of this thesis are presented in Table 6.5. All methods are evaluated on the same 210 RSHB trials; GNN and ResGCN models are trained exclusively on the 630 synthetic trials without access to any real-data examples or labels.

**Table 6.5. ML-Enhanced Pipeline Results on RSHB (Real Market Data, 210 Trials)**

| Method | Top-1 | Top-3 | Top-5 | MRR |
|--------|-------|-------|-------|-----|
| APA-RCA v2 (baseline) | 32.9% | 67.6% | 82.4% | 0.503 |
| APA-RCA v2 + GNN (z-score) | 31.4% | 49.5% | 68.6% | 0.445 |
| VAE + APA-RCA v2 | 43.3% | 60.0% | 60.0% | 0.533 |
| **VAE + APA-RCA v2 + GNN** **(v1)** | **54.8%** | 65.7% | 71.9% | 0.620 |
| C-VAE + APA-RCA v2 | 43.8% | 59.5% | 60.5% | 0.536 |
| APA-RCA v2 + ResGCN (z-score) | 32.4% | 66.2% | 71.0% | 0.496 |
| CVAE + APA-RCA v2 + ResGCN (v2) | 51.0% | 74.8% | 74.8% | 0.626 |
| **Hybrid TSD-CVAE + APA-RCA v2 + ResGCN (v3)** | 49.5% | **80.0%** | **83.3%** | **0.639** |

**Interpreting the v1 → v2 architectural changes.** The v1 pipeline (VAE + APA-RCA v2 + GNN) achieves 54.8% Top-1, the highest single-node precision. The v2 pipeline (CVAE + APA-RCA v2 + ResGCN) achieves 51.0% Top-1 but 74.8% Top-3—a gain of +9.1 percentage points over v1's Top-3 of 65.7%. MRR improves from 0.620 to 0.626. The v2 architecture thus produces a more accurate Top-3 ranking even when the very top-ranked candidate is not always the true root cause.

For operational use, the relevant metric depends on the deployment scenario. When the system makes a single automated recommendation (e.g., triggers an incident ticket to a specific team), Top-1 is paramount—v1 is preferable. When analysts review a short candidate list, Top-3 is more important—v2 improves the probability that the true cause appears in the first three recommendations from 65.7% to 74.8%. The MRR improvement (+0.006) confirms that v2 provides marginally better expected rank quality across the full evaluation set.

**The v2 → v3 refinement.** The v3 pipeline replaces the C-VAE's shared decoder with six type-specific decoder sub-networks (Hybrid TSD-CVAE, Section 5.2.1a), while retaining conditioning in both encoder and decoder. Top-1 remains statistically unchanged (49.5% vs 51.0%, p = 0.762), while Top-3 improves to 80.0% (+5.2 pp over v2, p = 0.203) and Top-5 to 83.3% (+8.5 pp, p = 0.032, significant at α = 0.05). MRR improves from 0.626 to 0.639. Per-type analysis reveals the structural benefit: A4 (aggregation error) Top-1 recovers from 53.3% to 70.0%, confirming that dedicated Aggregate-type decoders learn sharper anomaly boundaries than conditioning alone. A3 (feature calculation error) improves from 73.3% to 83.3%. The v3 architecture thus achieves the best Top-3, Top-5, and MRR among all evaluated methods.

The incremental contributions of the ML components (v1 trajectory) reveal the synergistic interaction: VAE alone contributes +10.4pp over baseline (43.3%), and the GNN then contributes an additional +11.5pp, for a total of +21.9pp. The v2 trajectory shows similar complementarity: C-VAE alone contributes +10.9pp (43.8%), and ResGCN adds +7.2pp, for a total of +18.1pp. In both cases, neither component alone approaches the combined gain.

**The Top-5 trade-off.** The v1 and v2 ML pipelines achieve lower Top-5 than the APA-RCA v2 baseline (82.4%), as GNN/ResGCN re-ranking concentrates probability mass toward the true root cause, improving Top-1 and Top-3 at the cost of Top-5 recall. The v3 pipeline resolves this trade-off: its 83.3% Top-5 surpasses the APA-RCA v2 baseline while simultaneously achieving the best Top-3 (80.0%) and MRR (0.639).

![**Figure 6.5.** ML-Enhanced Pipeline Results on RSHB (Real Market Data, 210 Trials). (a) Top-K Accuracy: the full VAE+APA-RCA+GNN pipeline (v1) achieves 54.8% Top-1 (+21.9 pp), the CVAE+APA-RCA+ResGCN pipeline (v2) achieves 51.0% Top-1 but 74.8% Top-3 (+9.1 pp over v1), and the Hybrid TSD-CVAE pipeline (v3) achieves 80.0% Top-3 and 83.3% Top-5 with MRR 0.639. (b) MRR: v3 achieves the highest MRR of 0.639. Both pipelines generalize successfully to real market data without fine-tuning.](figures/fig5_rshb_ml.pdf){width=\textwidth}

## 6.6 Ablation Analysis: The VAE–GNN Interdependence

The single most important ablation result in Table 6.5 is the performance of APA-RCA v2 + GNN with z-score anomaly signals: **31.4% Top-1**, which is *below* the APA-RCA v2 baseline of 32.9% by 1.5 percentage points. Adding a GNN without the VAE makes performance slightly worse.

This is not a pathological failure of the GNN architecture—the same GNN, when fed VAE features, achieves +21.9pp. It is a failure of the feature space under distribution shift. The GNN trained on synthetic z-score features learns associations of the form: "if the z-score signal pattern of a node looks like X, and the node is at layer L with RWR score R, then it is (not) the root cause." This learned association is calibrated on the joint distribution of GBM-generated z-scores and pipeline positions.

When the same GNN is applied to real market data, the z-score distributions are different: real price returns have fat tails (extreme z-scores occur more frequently), volatility clustering (z-scores are serially correlated), and cross-instrument dependence (high z-score at one node correlates with high z-scores at related nodes). The learned associations between z-score patterns and root cause locations no longer hold. The GNN's re-ranking is effectively random with respect to the true root cause, and since it disrupts APA-RCA's sensible baseline ranking, it slightly degrades performance.

The VAE reconstruction error, by contrast, is always computed relative to the model's learned representation of normal time series shapes. Whether a window comes from a GBM process or real market data, an anomalous-looking window (unusual temporal dynamics, elevated variance relative to the learned normal manifold) produces a high reconstruction error. The relative ordering of nodes by VAE score is approximately preserved across the domain shift, and the GNN's feature space remains interpretable.

**Conclusion.** VAE and GNN form an inseparable integrated system. The VAE is not optional preprocessing; it is the mechanism that makes cross-domain GNN transfer possible. This finding is the core empirical contribution of Chapter 5.

![**Figure 6.6.** Ablation Study on RSHB: The Necessity of VAE for Cross-Domain GNN Transfer (210 Trials). Adding GNN without VAE (z-score+GNN) regresses −1.5 pp below baseline, while VAE+APA-RCA+GNN achieves the best Top-1 (+21.9 pp). The VAE is the enabling component that makes GNN re-ranking effective across the synthetic-to-real domain shift.](figures/fig6_ml_ablation.pdf){width=\textwidth}

## 6.7 Unsupervised Scorer Comparison: VAE vs. Isolation Forest vs. One-Class SVM

**Table 6.6. Scorer Comparison on RSHB (210 Trials)**

| Method | Top-1 | Notes |
|--------|-------|-------|
| APA-RCA v2 (reference) | 32.9% | Statistical z-score detector |
| z-score + APA-RCA + GNN (reference) | 31.4% | GNN without VAE |
| IF + APA-RCA + GNN | **0.5%** | Boundary-based scorer |
| OCSVM + APA-RCA + GNN | **0.0%** | Boundary-based scorer |
| VAE + APA-RCA + GNN | **54.8%** | Manifold-based scorer |

Isolation Forest and One-Class SVM achieve Top-1 accuracy of 0.5% and 0.0% respectively—near-total failure on real data despite achieving meaningful accuracy on synthetic data. The gap between synthetic and real performance is categorical, not marginal: these methods work on synthetic data (where their boundary calibration is valid) and completely fail on real data (where it is not).

The theoretical explanation parallels the z-score GNN ablation of Section 6.6. IF and OCSVM are boundary-based methods: they partition the feature space of individual time series windows and classify each window as normal or anomalous based on proximity to the learned boundary. When the training distribution (GBM) and the test distribution (real market data) differ in fundamental structural properties—tail behavior, autocorrelation structure, cross-instrument dependence—the boundary that was calibrated on synthetic data loses all predictive validity on real data.

The magnitude of the performance collapse (0.5% vs. 54.8%) serves as a quantitative demonstration of why manifold-based representations are necessary for cross-domain anomaly scoring in this setting. VAE reconstruction error is the only scorer among the three that provides a transferable feature space for the GNN. This finding justifies the architectural commitment to VAE as the anomaly scorer in the ML-enhanced pipeline.

![**Figure 6.7.** Unsupervised Anomaly Scorer Comparison on RSHB (210 Trials). Isolation Forest (0.5%) and One-Class SVM (0.0%) collapse to near-random performance on real market data; boundary-based calibration learned on GBM distributions does not transfer. VAE (54.8%) succeeds by learning a distribution-agnostic reconstruction manifold that preserves anomaly signal across the domain shift. IF/OCSVM bars are inflated for visibility; actual values are as labeled.](figures/fig7_scorer_comparison.pdf){width=\textwidth}

## 6.8 Hybrid System Analysis: Schema-Constraint vs. ML

A natural question arising from the complementary profiles of APA-RCA and Schema-Constraint (Section 6.4) is whether a hybrid system—using Schema-Constraint for A4 anomalies and APA-RCA for all others—could outperform the ML-enhanced pipeline.

To answer this rigorously, we compute the performance of the **theoretically optimal hybrid**: route each trial to whichever method is better for that anomaly type, using the per-type Top-1 accuracy observed on RSHB.

**Table 6.7. Theoretical Optimal Hybrid vs. ML Pipeline (RSHB, 210 Trials)**

| Anomaly Type | Method Used | Top-1 | Correct / Total |
|--------------|-------------|-------|-----------------|
| A1 Source Corruption | APA-RCA | 86.7% | 26/30 |
| A2 ETL Logic Error | APA-RCA | 0.0% | 0/30 |
| A3 Feature Calculation Error | APA-RCA | 0.0% | 0/30 |
| A4 Aggregation Error | Schema-Constraint | 100.0% | 30/30 |
| A5 Downstream Masking | APA-RCA | 13.3% | 4/30 |
| A6 Multi-Source | APA-RCA | 33.3% | 10/30 |
| A7 Coincidental | APA-RCA | 96.7% | 29/30 |
| **Hybrid Total** | | **47.1%** | **99/210** |

The theoretical optimal hybrid achieves **47.1% Top-1**—7.7 percentage points below the ML pipeline's 54.8% (v1). Against the v3 pipeline (49.5% Top-1), the gap is 2.4 pp, but the v3 pipeline provides substantially superior Top-3 accuracy (80.0%) and Top-5 accuracy (83.3%) vs. the hybrid's lower Top-3 coverage. The hybrid earns 30 perfect answers from A4 but fails entirely on A2 and A3 (60 trials combined), which APA-RCA cannot handle and Schema-Constraint does not address. The ML pipeline, by contrast, achieves partial success on A2 and A3 through the VAE's improved anomaly detection, narrowing the gap on these hard types even if it cannot solve them entirely.

**Why the ML pipeline wins.** Schema-Constraint is a one-type specialist: it achieves 100% on A4 (14.3% of trials) while providing zero benefit for the remaining 85.7% of trials. The ML pipeline improves across all anomaly types simultaneously. This is the key operational advantage of a learned, domain-agnostic approach over a rule-based, type-specific module.

**Practical deployment guidance.** We summarize the operational trade-offs:

| Priority | Recommended System | Rationale |
|----------|-------------------|-----------|
| Minimize deployment complexity | APA-RCA + Schema-Constraint | No training required; rule-based; fast to deploy; strong Top-5 (82.4%) |
| Maximize first-diagnosis accuracy (Top-1) | VAE + APA-RCA + GNN | +21.9pp Top-1; suited for automated alerting |
| Maximize Top-3 candidate quality | Hybrid TSD-CVAE + APA-RCA + ResGCN (v3) | 80.0% Top-3, 83.3% Top-5, MRR 0.639; best overall ranking quality |
| Maximum overall performance | Hybrid TSD-CVAE + APA-RCA + SC + ResGCN | SC handles A4 perfectly; v3 ML handles all others; no conflict |

## 6.9 Error Analysis

### 6.9.1 A2 (ETL Logic Error): Zero Top-1 Accuracy on RSHB

The dramatic contrast between A2's synthetic performance (100% Top-5 in Table 6.2) and RSHB performance (0% Top-1 in Table 6.4) is the most striking anomaly in our results and requires a careful explanation.

In the synthetic setting, A2 injects a systematic scale factor of M = 8 into an ETL FX conversion node. The affected ETL node's output is 8× the expected level, making it an unambiguous statistical outlier relative to peer ETL nodes performing the same FX conversion on other instruments. The peer deviation signal (s_peer) identifies it with high confidence.

In the RSHB setting with real market data, this same injection produces a qualitatively different detection environment. Real FX rates exhibit substantially higher volatility (the KRW/USD rate moved ±10% in 2022 alone), real stock prices have higher cross-instrument correlation (Samsung and SK Hynix are highly correlated given their shared semiconductor sector exposure), and real ETL outputs have wider natural variation relative to peers than their GBM-generated counterparts. The peer deviation signal that cleanly separated the anomalous ETL node from peers in synthetic data is overwhelmed by this background noise on real data.

This limitation points to a genuine weakness of statistical anomaly detection in adversarial market conditions: the same anomaly magnitude that is clearly detectable in a synthetic environment may be indistinguishable from natural variation in a turbulent real market. Addressing this would require either a larger injection magnitude (which may not be realistic), a more powerful anomaly detector trained on real-data statistics, or domain-specific priors about which instrument relationships should remain stable even in volatile markets.

### 6.9.2 A3 (Feature Calculation Error): Zero Top-1 Accuracy on RSHB

A3 injects a feature calculation error by substituting a 2-day rolling window for the standard 20-day window in a volatility node. This produces a volatility estimate that is both higher in level and more volatile than peers—exactly the pattern the short-long variance ratio signal (s_sl) is designed to detect.

In the synthetic setting, this signal works partially (10% Top-5). In the RSHB setting, it fails entirely. The reason is that real financial volatility is itself non-stationary: during the Federal Reserve tightening cycle (2022) and the AI-driven rally (2023–2024) covered by the RSHB data, volatility regimes changed substantially, causing legitimate volatility nodes to exhibit elevated short-long variance ratios. The s_sl signal can no longer distinguish a misconfigured rolling window from a genuine volatility regime change, producing false positives that mask the truly anomalous node.

This is a well-known challenge in financial data quality monitoring: many "anomaly" patterns are indistinguishable from genuine market regime changes without additional context. Future work should incorporate regime-conditional anomaly scoring, where the expected signal distribution adapts dynamically to the prevailing market environment.

### 6.9.3 A7 (Coincidental): Near-Perfect Accuracy on Synthetic Data

A7 achieves 96.7% Top-1 on RSHB with the APA-RCA v2 baseline and v1 ML pipeline (76.7%)—the second-highest among all anomaly types. The explanation is straightforward: A7 injects a direct spike at a source node, mechanically identical to A1 source corruption but labeled differently because it represents a large-but-plausible market move rather than a data error. The multi-signal detector responds identically to A1 and A7, assigning a near-maximal anomaly score to the affected source node. APA-RCA then trivially ranks this highest-scoring source node first. The high accuracy on A7 thus reflects not a genuinely hard detection problem but a labeling convention: A7 anomalies have the same signature as A1 and are equally detectable. The A7 performance of the v2 CVAE+ResGCN pipeline is discussed separately in Section 6.10.2.

## 6.10 Architectural Enhancement Analysis: CVAE and ResGCN (v2)

This section analyzes the per-anomaly-type consequences of the architectural improvements introduced in v2 (C-VAE anomaly scorer + ResGCN re-ranker) relative to v1 (VAE + standard GCN).

### 6.10.1 Per-Anomaly-Type Analysis

**Table 6.8. Per-Anomaly-Type Top-1 Accuracy on RSHB: v2 vs. v1 vs. Baseline (210 Trials)**

| Anomaly Type | APA-RCA v2 | v1: VAE+GNN | v2: CVAE+ResGCN | Δ (v2 vs v1) |
|--------------|-----------|-------------|-----------------|--------------|
| A1 Source Corruption | 86.7% | 100.0% | **100.0%** | 0.0 pp |
| A2 ETL Logic Error | 0.0% | 0.0% | 0.0% | 0.0 pp |
| A3 Feature Calculation Error | 0.0% | 100.0% | 93.3% | −6.7 pp |
| A4 Aggregation Error | 0.0% | 20.0% | **70.0%** | +50.0 pp |
| A5 Downstream Masking | 13.3% | 13.3% | 0.0% | −13.3 pp |
| A6 Multi-Source | 33.3% | 73.3% | **93.3%** | +20.0 pp |
| A7 Coincidental | 96.7% | 76.7% | **0.0%** | −76.7 pp |

The v2 architecture produces dramatic gains on A4 (+50pp) and A6 (+20pp) while maintaining perfection on A1. The C-VAE's type-specific manifolds are particularly effective for A4 (aggregation error), where the erroneous aggregation node behaves abnormally *for its type* in a way that a type-conditioned encoder captures precisely: the aggregation node's output deviates from the expected aggregation manifold, while a standard VAE (which averages across types) is less sensitive to this type-specific deviation.

A6 (multi-source anomaly) improvements arise from the ResGCN's ATF-weighted message passing: with multiple source nodes anomalous, the ResGCN's source-type ATF weight (ρ = 1.0) correctly prioritizes the source-layer nodes in its aggregation, while the aggregate layer receives downweighted messages (ρ = 0.45). This structural difference in how information flows through the ATF-weighted adjacency helps the ResGCN localize the highest-impact source anomaly.

### 6.10.2 A7 Coincidental Anomaly: Architecture-Induced Regression

The most significant finding in Table 6.8 is the catastrophic A7 regression: from 76.7% (v1) to **0.0%** (v2). This is not a random fluctuation—it is a systematic failure across all 30 A7 trials, requiring a principled explanation.

**Root cause: C-VAE source-type manifold trained on GBM misrepresents real market regime changes.** A7 injects a large-but-plausible spike at a Source node—the same injection type as A1, but representing a genuine market move rather than a data error. For APA-RCA v2 and v1 VAE+GNN, this injection produces a large reconstruction error at the source node because the spike falls off the GBM-calibrated normal manifold. APA-RCA's transition matrix assigns high anomaly weight to the source node, and the GNN's high anomaly score feature [2] confirms the localization. The result is near-perfect A7 accuracy in v1 (76.7%).

The C-VAE conditions on transform type *and* pipeline layer. For Source-layer nodes, the C-VAE learns a manifold that captures normal behavior specifically for source nodes from GBM-generated data. The RSHB data covers 2022–2024: the Federal Reserve tightening cycle (2022) produced extreme equity and FX volatility that is *plausible but highly unusual* by GBM standards. The AI-driven rally (2023–2024) produced another regime shift. During these periods, legitimate Source node movements exhibit large z-score deviations—not because they are errors, but because real markets exhibit fat tails and volatility clustering that GBM does not model.

The C-VAE, conditioned on the source-type manifold, assigns high reconstruction error to these legitimate but unusual real-market movements, effectively confusing genuine regime changes with A7 injections. The ResGCN then receives misleading features: multiple source nodes show elevated C-VAE scores (legitimate market volatility) while the true A7 injection node is not distinguishable by the type-conditioned manifold alone. The ResGCN cannot recover from this feature-level confusion and places the true A7 node outside its top candidates in all 30 trials.

In contrast, the z-score-based APA-RCA (v2 baseline: 96.7% A7 accuracy) handles A7 precisely because z-scores measure deviation relative to the *local* rolling distribution. A spike 8× the local standard deviation is anomalous by z-score definition regardless of the global market regime. The C-VAE's global manifold approach—learning what source nodes look like on average during training—is less robust to distribution shifts at the source layer.

**Implication.** The C-VAE's conditioning improvement creates a trade-off: better type-discrimination (gains on A3, A4, A6) at the cost of reduced robustness to source-layer distribution shift (A7 regression). This trade-off reflects a fundamental tension in conditional anomaly detection: conditioning on metadata improves in-distribution discrimination but can increase sensitivity to out-of-distribution behavior when the conditioning domain (GBM source dynamics) mismatches the deployment domain (real market source dynamics).

### 6.10.3 Score Blending Experiment

Given the A7 regression, we investigated whether blending C-VAE and z-score signals could recover A7 performance while preserving the v2 gains on A3/A4/A6.

**Experimental design.** We evaluated a convex blend:

blend_score(v, α) = α × score_CVAE(v) + (1 − α) × score_z(v)

for α ∈ {0.0, 0.3, 0.5, 0.7, 1.0}, over the same 210 RSHB trials with the ResGCN trained on pure CVAE features (Phase 3, 630 synthetic samples). Computing both scores per trial yields five evaluation points in a single pass.

**Table 6.9. Score Blending Results on RSHB (210 Trials per α)**

| α (CVAE weight) | Top-1 | Top-3 | Top-5 | MRR | A7 Top-1 | A6 Top-1 |
|-----------------|-------|-------|-------|-----|----------|----------|
| 0.0 (pure z-score) | 32.4% | 44.8% | 53.8% | 0.427 | 96.7% | 33.3% |
| 0.3 | 29.5% | 53.3% | 54.3% | 0.428 | 46.7% | 53.3% |
| 0.5 | 27.6% | 53.8% | 56.7% | 0.422 | 10.0% | 70.0% |
| 0.7 | 30.0% | 53.8% | 66.7% | 0.456 | 0.0% | 96.7% |
| 1.0 (pure CVAE) | 45.2% | 75.2% | 81.9% | 0.608 | 0.0% | 90.0% |

**Blending does not achieve the desired objective.** Intermediate values of α (0.3–0.7) produce lower Top-1 accuracy than either extreme (pure z-score: 32.4%, pure CVAE: 45.2%), while recovering A7 only partially. The α = 0.3 blend recovers A7 to 46.7% but reduces Top-1 to 29.5% and Top-3 to 53.3%—substantially below v2's 51.0% and 74.8%.

**Root cause of blending failure.** The ResGCN was trained exclusively on pure CVAE anomaly scores as Feature [2]. When inference-time scores are blended (Feature [2] = α × CVAE + (1−α) × z-score), the ResGCN encounters a distribution of Feature [2] values that was never seen during training. The learned association between Feature [2] patterns and root cause locations breaks down: the GNN cannot distinguish whether a given Feature [2] value reflects a high CVAE score, a high z-score, or an average of both. This creates systematic feature distribution mismatch that degrades the ResGCN's re-ranking for all intermediate α.

This failure has a clear implication: post-hoc blending of scorer outputs is not a viable strategy when the downstream model (ResGCN) was trained on one specific scorer's distribution. Effective blending would require retraining the ResGCN on blended features, producing a model that has internalized the blended feature distribution. This is left as future work.

**Conclusion.** The v3 architecture (Hybrid TSD-CVAE+APA-RCA+ResGCN) achieves the best overall Top-3 (80.0%), Top-5 (83.3%), and MRR (0.639), surpassing v2 (51.0% Top-1, 74.8% Top-3, MRR = 0.626) on ranking quality while maintaining comparable Top-1 (49.5%). The choice among v1, v2, and v3 depends on the operational priority: v1 is preferable when A7 (coincidental market anomaly) detection is critical; v3 is preferable when Top-3/Top-5 ranking quality and A4 (aggregation error) accuracy are the primary concern.

---

# Chapter 7. Conclusion and Future Work

## 7.1 Summary of Contributions

This thesis presents **FINRCA**, a system for automated root cause analysis in financial risk data pipelines, built on a data lineage graph that models the full transformation path from raw market data to regulatory reports. We make four primary contributions.

**Contribution 1 — Financial Risk Lineage Graph (FRLG).** We define a typed, directed acyclic graph schema that models column-level data lineage in financial risk pipelines at operational granularity. The schema specifies five node types (Source, ETL, Feature, Risk, Report) and four edge types (DirectMap, Filter, Calculate, Aggregate), capturing the five-layer computational structure of VaR and Expected Shortfall pipelines. The FRLG is simultaneously an algorithmic artifact (the substrate for APA-RCA) and a regulatory compliance artifact: FRLG construction produces the attribute-level data lineage documentation required by BCBS 239 Article 9. The query interface supports forward impact analysis, backward ancestor queries, and shortest causal path extraction, directly addressing the ECB RDARR 2024 guidance on attribute-level lineage traceability.

**Contribution 2 — APA-RCA Algorithm and ATF Framework.** The Anomaly-Propagation-Aware Root Cause Analysis algorithm extends Random Walk with Restart with three domain-specific innovations: (a) traversal on the transposed FRLG, enabling backward causal search against the data flow direction; (b) anomaly-score-weighted transition probabilities that direct the walk toward detected anomalous regions of the pipeline; and (c) composite final scoring with ancestor constraint and upstream-preference correction that enforces the structural DAG property that only ancestors can be root causes.

The Anomaly Transfer Function (ATF) framework provides a formal model of how anomaly signals propagate through pipeline transform types, with closed-form transfer coefficients ρ_τ for each type. The ATF framework independently predicts the optimal attenuation parameter α* ≈ 0.72—consistent with the empirically chosen α = 0.70—providing theoretical grounding for what was previously an empirical hyperparameter. The φ_peer temporal discriminant, derived from ATF principles, achieves distribution-free separation of A3 (persistent feature errors) from A5 (transient spike propagation) with non-overlapping distributions on both synthetic and real data.

**Contribution 3 — ML-Enhanced Pipeline with ATF-Grounded Architecture.** We address the fundamental challenge of incorporating machine learning in a labeled-data-scarce setting through two components trained without real-data anomaly labels. In the v1 pipeline, the VAE anomaly scorer learns a manifold-based representation of normal financial time series and the standard GCN re-ranker learns topological root cause patterns, with the v1 pipeline (VAE+GNN) improving Top-1 RSHB accuracy by **+21.9 pp** (32.9% → 54.8%). The v2 pipeline (C-VAE+ResGCN), which encodes ATF theory into the model structure, achieves 74.8% Top-3 (+9.1 pp over v1). The v3 pipeline (Hybrid TSD-CVAE+ResGCN) further extends the C-VAE with type-specific decoder sub-networks, achieving the best Top-3 (**80.0%**), Top-5 (**83.3%**), and MRR (**0.639**) among all evaluated methods.

The v2 architecture introduces two domain-theoretically grounded structural improvements. The **Conditional VAE (C-VAE)** extends the standard VAE by conditioning encoder and decoder on the node's transform type (6-dim one-hot) and pipeline layer (5-dim one-hot), producing type-specific reconstruction manifolds aligned with the ATF framework's insight that different transform types exhibit fundamentally different normal behavior patterns. The **Residual GCN (ResGCN)** replaces binary adjacency weights with ATF transfer coefficients ρ_τ—the same values that ground the APA-RCA transition matrix—and adds a residual skip connection to prevent over-smoothing of individual anomaly scores across aggregation layers. An eighth node feature (ATF-weighted path distance) encodes graph proximity using the same ρ_τ framework. The v3 **Hybrid TSD-CVAE** extends the C-VAE by routing the latent code to one of six type-specific decoders while retaining the conditioning vector in each decoder, combining structural separation with information preservation. This hybrid design resolves the v2 C-VAE's failure mode on A4 (aggregation error), improving A4 Top-1 from 53.3% to 70.0%.

The v3 pipeline (Hybrid TSD-CVAE+APA-RCA+ResGCN) achieves 49.5% Top-1, 80.0% Top-3, and 83.3% Top-5 on RSHB with MRR 0.639—the best ranking quality among all evaluated methods. A notable architectural finding across the v1-v2-v3 progression is the trade-off introduced by C-VAE conditioning: the source-type manifold trained on GBM data becomes brittle under real market regime changes (Fed tightening 2022, AI rally 2023–2024), causing the A7 coincidental anomaly accuracy to collapse from 76.7% (v1) to 0.0% (v2). This finding reveals a fundamental tension in conditional anomaly detection: domain-specific conditioning improves type-discrimination at the cost of distribution-shift robustness at the source layer.

An ablation study demonstrates that the VAE/C-VAE is not a replaceable component: substituting an Isolation Forest or One-Class SVM scorer reduces the GNN+pipeline to near-zero Top-1 accuracy (0.5% and 0.0% respectively), while a GNN trained on statistical z-score signals without the VAE *regresses below the baseline* (31.4% vs. 32.9%). The manifold-based representation—conditional or unconditional—is the mechanism that makes cross-domain GNN transfer possible.

**Contribution 4 — SynFRP Benchmark.** We construct and release the Synthetic Financial Risk Pipeline (SynFRP) benchmark: 7 anomaly types × 3 pipeline scales × 30 trials = 630 controlled experiments, with GBM/Vasicek/random-walk data generation, seeded for exact reproducibility, and accompanied by the real-market RSHB evaluation protocol. To our knowledge, SynFRP is the first public benchmark specifically designed for evaluating root cause analysis algorithms in the financial risk data lineage domain.

## 7.2 Academic and Practical Significance

**Academic significance.** The academic literature on automated root cause analysis is dominated by two lines of work: microservice RCA (focusing on service availability and latency anomalies in distributed systems) and general data pipeline quality management. FINRCA occupies a gap between these: it addresses *data quality errors* propagating through *typed, domain-specific* data transformation graphs in the financial domain. The structural distinctions are fundamental—financial risk pipelines are DAGs with typed transform edges and strict layer semantics, whereas microservice call graphs are dynamic, potentially cyclic, and carry no domain-specific edge typing. Prior methods designed for one setting do not naturally transfer to the other. FINRCA establishes the financial risk data lineage RCA problem as a distinct research area and provides the first academic evaluation benchmark and algorithmic baseline.

The ATF framework contributes a theoretical perspective on anomaly propagation in typed data pipelines that extends beyond the financial domain. The concept of transform-type-specific transfer coefficients—and the derivation of optimal algorithm parameters from these coefficients—provides a template for principled hyperparameter justification in graph-based anomaly attribution systems more broadly.

**Practical significance.** BCBS 239 compliance remains aspirational for most G-SIBs despite more than a decade since the regulation's publication. FINRCA addresses two of the most technically challenging requirements: attribute-level data lineage documentation (FRLG) and automated data quality attribution (APA-RCA). Both are delivered without requiring labeled incident data, which is structurally unavailable in most financial institutions. The sub-millisecond query latency makes FINRCA suitable for integration into real-time risk monitoring workflows, where delays compound into reporting SLA violations with regulatory consequences.

The RSHB results demonstrate that FINRCA is not merely a synthetic benchmark curiosity: it handles real Samsung Electronics, SK Hynix, and US equity price data under genuine market volatility conditions, including the 2022 Fed tightening cycle and 2023–2024 AI equity rally. The +21.9pp Top-1 improvement (v1) and the v3 pipeline's 80.0% Top-3, 83.3% Top-5, and MRR 0.639 observed on this data translate directly to a reduction in mean time to root cause identification in production operations.

## 7.3 Limitations

**Synthetic data validity.** The bulk of our evaluation relies on the SynFRP benchmark, which uses GBM, Vasicek, and random-walk stochastic processes. While these are the standard econometric models for equity prices, interest rates, and FX, real financial time series exhibit fat tails (kurtosis well above 3), volatility clustering (GARCH-type autocorrelation in squared returns), and non-stationary cross-instrument correlations driven by macroeconomic regimes. The RSHB addresses this partially by using real source-layer data, but the pipeline logic and anomaly injection remain synthetic. A full real-world evaluation would require access to actual incident data from a financial institution—commercially sensitive and currently inaccessible under standard academic data governance constraints.

**Single-fault assumption.** Each SynFRP and RSHB trial injects exactly one anomaly at exactly one node. Real financial pipeline incidents often involve multiple simultaneous failures: a common data feed outage can corrupt multiple source nodes simultaneously, and independent code changes can introduce bugs at multiple pipeline stages concurrently. The APA-RCA algorithm and GNN re-ranker are both designed around a single-root-cause assumption and may produce degraded or misleading rankings in multi-fault scenarios.

**Schema-Constraint metadata dependency.** SchemaConstraintRCA requires `expected_input_count` to be explicitly declared for each aggregation node in the pipeline specification. Maintaining this metadata in sync with an evolving pipeline—new instruments added, aggregation groups restructured, calculation formulas updated—is a non-trivial operational process at large institutions with hundreds of aggregation nodes and frequent pipeline changes. In the absence of this metadata, the 100% A4 Top-1 accuracy of Schema-Constraint is unavailable, and the system falls back to APA-RCA's 0% A4 performance.

**Static pipeline assumption.** FINRCA assumes the pipeline topology is fixed and fully specified at query time. Production pipelines at large financial institutions evolve continuously as regulatory requirements change, new products are launched, and calculation methodologies are updated. The FRLG would need to be rebuilt or incrementally updated with each structural change, and the GNN would need to be retrained if pipeline topology changes significantly. Automated lineage capture from operational metadata (OpenLineage events) could trigger incremental FRLG updates; GNN retraining is more expensive and would require a retraining schedule.

**GNN stochasticity.** GNN training involves random weight initialization and stochastic optimization. Results from a single trained model exhibit small run-to-run variation. The 5-fold cross-validation results in Experiment A quantify this variation in the synthetic setting, but the RSHB results (Experiment B) reflect a single trained model applied to real data. Reporting confidence intervals from multiple training runs is left as future work.

**C-VAE source-layer distribution shift.** The v2/v3 C-VAE and Hybrid TSD-CVAE architectures introduce a specific failure mode for anomaly types at the source layer under real market regime changes (Section 6.10.2). The source-type conditioning manifold, trained on GBM-generated data, assigns high reconstruction error to legitimate but unusual real-market movements during the 2022 Fed tightening cycle and 2023–2024 AI rally—misidentifying regime changes as anomalies and causing A7 accuracy to collapse from 76.7% (v1) to 0.0% (v2/v3). This reflects a broader limitation of conditional VAE approaches: conditioning on metadata improves within-distribution discrimination but can amplify out-of-distribution sensitivity when the conditioning domain (GBM source dynamics) is structurally different from the deployment domain (real market source dynamics). Addressing this requires either regime-adaptive conditioning, a hybrid model that retains unconditional VAE for source-layer nodes, or augmentation of the C-VAE training data with real market source-layer time series.

## 7.4 Future Research Directions

**Real financial institution validation.** The ultimate test of FINRCA's external validity is evaluation on genuine production pipeline incidents at a real financial institution. The author's prior experience at KB Kookmin Bank's StarBanking Operations Team provides contextual familiarity with the operational environment and suggests the plausibility of a pilot study, subject to appropriate data governance agreements. Such a study would require mapping the existing StarBanking metadata infrastructure to the FRLG schema and accessing historical incident reports for retrospective evaluation.

**Regime-adaptive C-VAE conditioning.** The A7 regression revealed in v2 (Section 6.10.2) motivates a regime-aware extension of the C-VAE. Rather than conditioning solely on node type and layer, the encoder could additionally condition on a latent market regime indicator—trained on market volatility indices (VIX, realized variance) or learned via an unsupervised regime segmentation model. Within each regime, the reconstruction manifold would be calibrated to that regime's distribution, making the C-VAE robust to the regime-change confounds that caused the A7 failure. The key technical challenge is that regime labels are unavailable at training time and must themselves be learned without supervision.

**Multi-fault extension.** Extending APA-RCA to handle M simultaneous root causes requires fundamental algorithmic changes. Three promising directions are: (1) iterative peeling—identify and neutralize the highest-ranked root cause, remove its anomaly contribution from the graph, and re-run to find the next; (2) submodular coverage—select the k-node set that maximally explains the observed anomaly pattern across all report nodes; and (3) multiple-restart RWR—allow the restart distribution to place mass at multiple seed nodes corresponding to different suspected root cause regions.

**Learned transform weights.** The fixed β priors encode domain-level beliefs about which transform types are more likely to introduce errors. The ablation study (Section 4.7) shows that these priors help on average but may hurt for specific anomaly types (notably, removing β improves Top-1 for A1 anomalies). A learned β—conditioned on the anomaly type (if inferred from metadata) or on the pipeline's topological features—would be more adaptive. The ATF framework provides a principled starting point: the theoretical ρ_τ values suggest that β_Aggregate should be *lower* than β_DirectMap for anomaly attribution (because aggregation dilutes signals), even though aggregate bugs are more impactful. Learning β from incident data could reconcile these competing considerations.

**Online incremental FRLG updates.** A production FRLG maintenance system should support incremental updates triggered by pipeline change events (new node added, edge removed, transform type modified) without requiring full graph reconstruction. OpenLineage's event-driven architecture provides a natural mechanism: each `RunEvent` carrying lineage information could be processed as a FRLG delta, updating affected nodes and edges while preserving the unchanged portions. This would also enable real-time anomaly monitoring rather than batch post-incident diagnosis.

**LLM-augmented explanations.** The FRLG contains rich structured context for each identified root cause: node type, transform specification, upstream inputs, downstream consumers, anomaly score, and the full propagation path to the observed anomaly at the report node. This context is naturally serializable as a structured prompt for a large language model, enabling generation of analyst-ready natural language explanations ("Node ETL_FX_STOCK_2_PRICE applies FX conversion using a scale factor that is 8× the expected value. Its anomaly score (0.89) substantially exceeds peer ETL nodes (mean 0.12). Causal excess of 0.71 confirms the anomaly originates here rather than propagating from upstream source feeds..."). Integration with the BCBS 239 audit trail would produce human-verifiable, regulatorily defensible incident reports.

**BCBS 239 operational dashboard.** The ultimate product of this research agenda is a production-ready compliance dashboard that integrates FRLG visualization, real-time anomaly monitoring, and APA-RCA query results into a single operational interface. The FRLG's provenance properties (complete attribute-level lineage, O(|V|+|E|) path extraction) directly support BCBS 239 attestation requirements. The ML-enhanced pipeline's accuracy on real data (54.8% Top-1 v1, 80.0% Top-3 and 83.3% Top-5 v3) provides the diagnostic backbone. Such a system could meaningfully reduce the compliance gap that has persisted more than a decade after BCBS 239's publication.

---

# References

[1] Basel Committee on Banking Supervision. *Principles for effective risk data aggregation and risk reporting (BCBS 239)*. Bank for International Settlements, January 2013.

[2] Bank for International Settlements. *BCBS 239: Implementation and progress report*. BIS Newsletter No. 36, 2025.

[3] European Central Bank. *Guide on effective risk data aggregation and risk reporting (RDARR)*. ECB Banking Supervision, May 2024.

[4] PricewaterhouseCoopers. *BCBS 239: Where do G-SIBs stand in 2024?* PwC Financial Services Risk Practice, 2024.

[5] Basel Committee on Banking Supervision. *Minimum capital requirements for market risk (Fundamental Review of the Trading Book)*. Bank for International Settlements, January 2019; effective implementation January 2025.

[6] Cheney, J., Chiticariu, L., and Tan, W.C. Provenance in databases: Why, how, and where. *Foundations and Trends in Databases*, 1(4):379–474, 2009.

[7] Davidson, S.B. and Freire, J. Provenance and scientific workflows: Challenges and opportunities. In *Proceedings of SIGMOD*, pages 1345–1350, 2008.

[8] Foidl, H., Felderer, M., and Ramler, R. Data pipeline quality: Challenges and approaches. *Journal of Systems and Software*, 2024.

[9] Chen, Y., et al. RCACopilot: On-call incident root cause analysis via LLM for online service systems. In *Proceedings of EuroSys*, 2024.

[10] Pham, L., et al. BARO: Robust root cause analysis for microservices via multivariate Bayesian online change point detection. In *Proceedings of FSE*, 2024.

[11] Pham, L., et al. RCAEval: A comprehensive benchmark for root cause analysis in microservice systems. In *Proceedings of WWW*, 2025.

[12] Wang, D., et al. CORAL: Causal discovery via conditional mutual information for microservice root cause analysis. In *Proceedings of KDD*, 2023.

[13] Li, R., et al. Root cause localization for data quality anomalies via graph neural networks. In *Proceedings of VLDB*, 2024.

[14] Alam, M., et al. Data lineage reconstruction and validation for cloud-native pipelines. In *Proceedings of IC2E*, 2024.

[15] Assaad, C.K., et al. Root cause identification for collective anomalies in time series given an acyclic summary causal graph. *arXiv:2206.13390*, 2023.

[16] Tong, H., Faloutsos, C., and Pan, J.Y. Fast random walk with restart and its applications. In *Proceedings of ICDM*, pages 613–622, 2006.

[17] Wang, D., et al. Hierarchical graph neural networks for interdependent causal discovery and root cause analysis in complex systems. *arXiv:2307.12637*, 2023.

[18] Kingma, D.P. and Welling, M. Auto-encoding variational Bayes. In *Proceedings of ICLR*, 2014.

[19] Kipf, T.N. and Welling, M. Semi-supervised classification with graph convolutional networks. In *Proceedings of ICLR*, 2017.

[20] Liu, F.T., Ting, K.M., and Zhou, Z.H. Isolation forest. In *Proceedings of ICDM*, pages 413–422, 2008.

[21] Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J., and Williamson, R.C. Estimating the support of a high-dimensional distribution. *Neural Computation*, 13(7):1443–1471, 2001.

[22] Black, F. and Scholes, M. The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3):637–654, 1973.

[23] Vasicek, O. An equilibrium characterization of the term structure. *Journal of Financial Economics*, 5(2):177–188, 1977.

[24] Xu, H., et al. Anomaly Transformer: Time series anomaly detection with association discrepancy. In *Proceedings of ICLR*, 2022.

[25] OpenLineage. *OpenLineage: An open standard for data lineage metadata*. https://openlineage.io, 2023.

[26] Apache Software Foundation. *Apache Atlas: Data governance and metadata framework for Hadoop*. https://atlas.apache.org, 2024.

---

# Appendix A: SynFRP Pipeline Node Catalog (Medium Configuration)

The medium configuration contains approximately 63 nodes. The table below lists a representative subset organized by pipeline layer. Full node naming follows the convention `{LAYER}_{TYPE}_{INSTRUMENT}_{ATTRIBUTE}`.

| Node ID | Layer | Transform Type | Description |
|---------|-------|----------------|-------------|
| SRC_STOCK_1_PRICE | 1 | Source | Stock 1 daily closing price (GBM, μ=0.08, σ=0.20) |
| SRC_STOCK_2_PRICE | 1 | Source | Stock 2 daily closing price (GBM, independent seed) |
| SRC_STOCK_3_PRICE | 1 | Source | Stock 3 daily closing price (GBM, independent seed) |
| SRC_RATE_SHORT | 1 | Source | Short-term interest rate (Vasicek, θ=0.02, κ=0.5) |
| SRC_RATE_MED | 1 | Source | Medium-term interest rate (Vasicek, θ=0.03, κ=0.5) |
| SRC_USD_KRW | 1 | Source | USD/KRW exchange rate (geometric random walk, σ=0.05) |
| ETL_IMPUTE_STOCK_1_PRICE | 2 | Filter | Forward-fill missing values in Stock 1 price series |
| ETL_CLIP_STOCK_1_PRICE | 2 | Filter | 4σ outlier clipping applied to imputed Stock 1 price |
| ETL_FX_STOCK_1_PRICE | 2 | Calculate | FX conversion: USD price × USD/KRW rate |
| ETL_JOIN_1 | 2 | Calculate | Portfolio value: FX-converted price × position size |
| FEAT_RETURN_STOCK_1_PRICE | 3 | Calculate | Log daily return: log(price_t / price_{t-1}) |
| FEAT_VOL20_STOCK_1_PRICE | 3 | Aggregate | 20-day rolling standard deviation of log returns |
| FEAT_VOL60_STOCK_1_PRICE | 3 | Aggregate | 60-day rolling standard deviation of log returns |
| FEAT_SECTOR_A | 3 | Aggregate | Sector A mean return (average over Stocks 1–3) |
| FEAT_SECTOR_B | 3 | Aggregate | Sector B mean return (separate instrument group) |
| FEAT_TOTAL_VALUE | 3 | Aggregate | Total portfolio value: sum over all ETL_JOIN nodes |
| RISK_VAR95 | 4 | Aggregate | 60-day historical VaR at 95% confidence level |
| RISK_VAR99 | 4 | Aggregate | 60-day historical VaR at 99% confidence level |
| RISK_ES95 | 4 | Aggregate | Expected Shortfall at 95%: mean of returns below 5th percentile |
| RISK_PORTFOLIO_VOL | 4 | Aggregate | Portfolio volatility: mean of individual stock Vol20 metrics |
| RISK_STRESS | 4 | Calculate | Normalized stress scenario P&L |
| RISK_CONCENTRATION | 4 | Calculate | Max position value / total portfolio value |
| REPORT_REG_VAR | 5 | Report | VaR99 limit breach flag (regulatory submission) |
| REPORT_DASHBOARD | 5 | Report | Composite internal risk score |
| REPORT_STRESS | 5 | Report | Extreme stress loss flag |
| REPORT_LIMIT | 5 | Report | Concentration limit breach flag |

---

# Appendix B: APA-RCA Algorithm Pseudocode

```
Algorithm APA-RCA(G, A, t, γ=0.15, α=0.7, ε=1e-8, max_iter=1000):

  Input:  FRLG G = (V, E)
          Anomaly scores A : V → [0, 1]
          Target node t ∈ V
          Restart probability γ
          Attenuation coefficient α
  Output: Ranked list of root cause candidates

  // Step 1: Transpose graph for backward traversal
  G_T ← reverse_edges(G)

  // Step 2: Compute hop distances from target (BFS on G_T)
  hop ← BFS_distances(G_T, source=t)
  max_hop ← max(hop.values(), default=1)

  // Step 3: Build anomaly-weighted edge weight matrix
  W ← zero matrix of size |V| × |V|
  for each edge (u, v) in E_T:          // u is downstream, v is upstream
    W[v, u] ← A[v] × β(transform_type(u, v)) × α^(hop[v] / max_hop)

  // Step 4: Column-normalize to get transition matrix
  for each column j of W:
    if sum(W[:, j]) > 0:
      P[:, j] ← W[:, j] / sum(W[:, j])
    else:
      P[:, j] ← uniform(|V|)             // fallback for zero-score columns

  // Step 5: Construct anomaly-weighted restart distribution
  e ← zero vector of size |V|
  e[t] ← 0.7
  anomalous ← {v ∈ V : A[v] > 0.3 and v ≠ t}
  if |anomalous| > 0:
    for each v in anomalous:
      e[v] ← 0.3 × A[v]
    e[anomalous] ← e[anomalous] / sum(e[anomalous]) × 0.3
  e ← e / sum(e)

  // Step 6: RWR iteration until convergence
  r ← uniform vector of size |V|, normalized
  for k = 1 to max_iter:
    r_new ← (1 - γ) × P × r + γ × e
    if ||r_new - r||_1 < ε:
      break
    r ← r_new

  // Step 7: Composite scoring with ancestor constraint
  ancestors_t ← nx.ancestors(G, t)
  scores ← empty dict
  for each v in V where v ≠ t:
    anc_factor ← 1.0 if v ∈ ancestors_t else 0.05
    layer_bonus ← 1.0 + 0.15 × (5 - layer(v))
    scores[v] ← (0.4 × r[v] + 0.6 × A[v]) × anc_factor × layer_bonus × β(v)

  return sort(V \ {t}, key=scores, order=descending)
```

---

# Appendix C: Hyperparameter Summary

**Table C.1: VAE Hyperparameters**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Window size | 20 trading days | Matches one calendar month; captures rolling-window initialization artifacts |
| Latent dimension | 8 | Matches the number of primary node type–metric combinations in the pipeline |
| Hidden dimension | 64 | Balances representational capacity against training stability on CPU |
| Epochs | 80 | Validation loss converges by epoch ~60 in all configurations |
| Batch size | 256 | Standard for ~900K sample dataset on CPU hardware |
| Optimizer | Adam, lr = 1×10^{−3} | Standard default; no learning rate scheduling required |

**Table C.2: GNN Re-Ranker Hyperparameters**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Top-K candidates | 10 | Matches APA-RCA candidate list size |
| Hidden dimension | 32 | Appropriate for subgraphs of 15–30 nodes |
| Dropout rate | 0.3 | Prevents overfitting on 630 synthetic training examples |
| Epochs | 150 | Convergence observed by epoch ~120 |
| Number of GCN layers | 2 | Aggregates information from up to 2-hop neighborhoods |
| Optimizer | Adam, lr = 5×10^{−4} | Conservative setting to prevent over-smoothing |

**Table C.3: APA-RCA Hyperparameters**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Restart probability γ | 0.15 | Standard RWR literature value (Tong et al., 2006) |
| Attenuation coefficient α | 0.70 | Consistent with ATF-theoretic prediction α* ≈ 0.72 |
| Score fusion weights | 0.4 (RWR) + 0.6 (anomaly) | Anomaly score is the primary localization signal |
| Non-ancestor penalty | 0.05× | Near-exclusion of structurally impossible candidates |
| Layer bonus coefficient | 0.15 | Breaks ties in favor of source-layer nodes |
| Convergence tolerance ε | 10^{−8} | Ensures numerical stability of stationary distribution |
| Maximum iterations | 1,000 | Upper bound; convergence observed within 40 in practice |
| Anomaly threshold θ | 2.0 | Standard z-score detection threshold |

**Table C.4: Transform-Type Weight Priors (β)**

| Transform Type | β | Rationale |
|----------------|---|-----------|
| Aggregate | 1.5 | Aggregation bugs propagate to multiple downstream consumers |
| Report | 1.3 | Report nodes are natural observation points for anomaly accumulation |
| Calculate | 1.2 | Calculation errors introduce deterministic biases |
| DirectMap | 1.0 | Reference; neutral passthrough |
| Filter | 0.8 | Filters attenuate anomaly signals (protective operation) |

---

# Appendix D: RSHB Dataset Configuration

**Table D.1: Real-Source Hybrid Benchmark (RSHB) Specification**

| Parameter | Value |
|-----------|-------|
| Data source | Yahoo Finance API (public historical OHLCV data) |
| Coverage period | 756 trading days (approx. 3 years ending December 31, 2024) |
| Market regimes covered | Fed tightening cycle (2022), US regional banking stress (2023), AI equity rally (2023–2024) |
| Instruments included | Samsung Electronics (005930.KS), SK Hynix (000660.KS), 5 US large-cap equities, 3 FX pairs (USD/KRW, EUR/USD, USD/JPY), 3 US Treasury yields (2Y, 10Y, 30Y) |
| Pipeline configuration | Medium: 5 stocks, 3 rates, 2 FX pairs, 5 positions |
| Pipeline node count | ~63 nodes |
| Anomaly types evaluated | 7 (A1–A7) |
| Trials per anomaly type | 30 |
| Total trials | 210 |
| Injection magnitude | M = 8.0 (identical to SynFRP) |
| GNN fine-tuning on real data | None (strict cross-domain evaluation) |
| Reproducibility seed | seed = 42 for data generation; trial = 0, 1, ..., 29 for injection |
