# FINRCA: Anomaly-Propagation-Aware Lineage Graph for Automated Root Cause Analysis in Financial Risk Pipelines

> **Thesis Chapters 1–3 Draft**
> Master of Science in Computer Science, Hanyang University

---

# Abstract (영문 초록)

Modern financial institutions are required under BCBS 239 to maintain complete, accurate, and traceable data lineage across their risk reporting pipelines. Yet as of 2025, fewer than 10% of globally systemically important banks fully comply, with data lineage cited as the most persistent technical challenge. When anomalies occur in multi-stage financial risk pipelines—spanning raw market data ingestion, ETL transformations, feature engineering, and regulatory reporting—identifying the root cause manually requires hours to days of expert investigation.

This thesis proposes **FINRCA**, a system for automated root cause analysis in financial risk pipelines. FINRCA automatically constructs a column-level **Financial Risk Lineage Graph (FRLG)** from a pipeline definition, capturing a five-layer node hierarchy (Source, ETL, Feature, Risk, Report) and four transform-typed edges (DirectMap, Filter, Calculate, Aggregate). An **Anomaly-Propagation-Aware Root Cause Analysis (APA-RCA)** algorithm runs on the graph, extending Random Walk with Restart with hop-distance-adaptive weighting, transform-type-based edge weights, and ancestor-constrained ranking. A five-signal anomaly detector—combining rolling z-score, short-long variance ratio, segment variance, peer deviation, and causal excess isolation—provides the node-level anomaly scores. A complementary **Schema-Constraint RCA** module detects aggregation schema violations with 100% Top-1 accuracy and zero false positives.

We construct the **Synthetic Financial Risk Pipeline (SynFRP)** benchmark: 630 controlled trials across seven anomaly types and three pipeline scales, providing ground-truth root cause labels. APA-RCA v2 achieves 77.9% overall Top-5 accuracy, outperforming the anomaly-score-only baseline by 7.7 percentage points. An **Anomaly Transfer Function (ATF)** framework provides a formal model of anomaly propagation through pipeline transforms, deriving principled algorithm parameters and a distribution-free temporal discriminant that achieves perfect A3/A5 separation on both synthetic and real data.

To validate real-market robustness, we design the **Real-Source Hybrid Benchmark (RSHB)**: 12 instruments (US equities, Korean Exchange equities, FX, Treasury rates) over 756 trading days spanning three market regimes, achieving 82.4% Top-5 accuracy on real market data.

Finally, we address the challenge of incorporating machine learning without labeled anomaly data—a structural constraint of the financial domain. A **VAE/Conditional VAE (C-VAE)/Hybrid TSD-CVAE anomaly scorer**, trained unsupervised on approximately 898,000 windows from anomaly-free pipelines, replaces the z-score detector. A **GNN/ResGCN re-ranker**, trained on synthetic injection locations and applied to real data without fine-tuning, achieves 54.8% Top-1 accuracy on RSHB—a +21.9 percentage point improvement over APA-RCA v2 alone. An architecturally enhanced v2 pipeline (C-VAE + ResGCN), which encodes ATF domain theory directly into the model structure, further improves Top-3 accuracy to 74.8% (+9.1 pp) and MRR to 0.626. A further architectural refinement, v3 (Hybrid TSD-CVAE + ResGCN), introduces type-specific decoder sub-networks while preserving conditioning, achieving 80.0% Top-3 (+5.2 pp over v2), 83.3% Top-5, and the highest MRR of 0.639. An ablation study comparing three unsupervised scorers (VAE, IsolationForest, OneClass-SVM) confirms that VAE reconstruction error is the only scorer that provides transferable GNN features across the synthetic-to-real distribution shift: IsolationForest achieves 0.5% and OneClass-SVM achieves 0.0% Top-1, while VAE achieves 54.8%.

**Keywords:** financial risk pipeline, data lineage, root cause analysis, anomaly detection, variational autoencoder, graph neural network, BCBS 239, random walk with restart

---

# 국문 초록

BCBS 239 규정에 따라 글로벌 시스템적 중요 은행(G-SIB)은 리스크 보고 파이프라인 전반에 걸쳐 완전하고 정확하며 추적 가능한 데이터 계보(lineage)를 유지해야 한다. 그러나 2025년 현재, 전체 G-SIB 중 완전 준수 기관은 10% 미만이며, 데이터 계보 관리는 가장 지속적인 기술적 과제로 지목되고 있다. 원시 시장 데이터 수집, ETL 변환, 피처 엔지니어링, 규제 보고에 이르는 다단계 금융 리스크 파이프라인에서 이상 현상이 발생했을 때, 근본 원인을 수동으로 식별하는 데는 수 시간에서 수 일이 소요된다.

본 논문은 금융 리스크 파이프라인에서의 자동 근본 원인 분석 시스템인 **FINRCA**를 제안한다. FINRCA는 파이프라인 정의로부터 컬럼 수준의 **금융 리스크 계보 그래프(FRLG)**를 자동 구성한다. FRLG는 5계층 노드 구조(Source, ETL, Feature, Risk, Report)와 4가지 변환 유형 에지(DirectMap, Filter, Calculate, Aggregate)를 포착한다. 그래프 위에서 **이상 전파 인식 근본 원인 분석(APA-RCA)** 알고리즘이 실행되며, 홉 거리 적응형 가중치, 변환 유형 기반 에지 가중치, 조상 제약 순위화를 통해 재시작 확률을 갖는 랜덤 워크(RWR)를 확장한다. 롤링 z-점수, 단기-장기 분산 비율, 구간 분산, 피어 편차, 인과 초과 격리의 5가지 신호를 결합한 이상 감지기가 노드 수준의 이상 점수를 산출한다. 보완적인 **스키마 제약 RCA** 모듈은 집계 스키마 위반을 Top-1 정확도 100%, 오탐률 0%로 탐지한다.

평가를 위해 **SynFRP** 벤치마크를 구축하였다: 7가지 이상 유형과 3가지 파이프라인 규모에 걸쳐 630회의 통제된 실험으로 구성되며 정답 레이블을 제공한다. APA-RCA v2는 전체 Top-5 정확도 77.9%를 달성하여 이상 점수 단독 기준선을 7.7 퍼센트포인트 상회한다. **이상 전달 함수(ATF)** 프레임워크는 파이프라인 변환을 통한 이상 전파의 수학적 모델을 제공하고, 알고리즘 파라미터에 대한 원리적 근거 및 합성·실제 데이터 모두에서 완벽한 분리 성능을 달성하는 분포-무관 시간적 판별자를 도출한다.

실제 시장 견고성 검증을 위해 **실소스 하이브리드 벤치마크(RSHB)**를 설계하였다: 세 가지 시장 국면에 걸친 756 거래일 동안 12개 금융 상품(미국 주식, 한국거래소 주식, 외환, 국채 금리)을 포함하며, Top-5 정확도 82.4%를 실제 시장 데이터에서 달성한다.

마지막으로, 이상 레이블 데이터 없이 머신러닝을 통합하는 방법론을 제안한다. 이상이 없는 파이프라인의 약 89만 8천 개 윈도우로 비지도 학습된 **VAE/조건부 VAE(C-VAE) 이상 스코어러**가 z-점수 탐지기를 대체한다. 합성 주입 위치를 학습 레이블로 사용하고 파인튜닝 없이 실제 데이터에 적용되는 **GNN/ResGCN 재순위화기**는 RSHB에서 Top-1 정확도 54.8%를 달성하며, 이는 APA-RCA v2 단독 대비 +21.9 퍼센트포인트 향상이다. ATF 도메인 이론을 모델 구조에 직접 반영한 아키텍처 강화 v2 파이프라인(C-VAE + ResGCN)은 Top-3 정확도를 74.8%(v1 대비 +9.1pp)로, MRR을 0.626으로 추가 향상시킨다. 추가적인 아키텍처 개선인 v3(Hybrid TSD-CVAE + ResGCN)는 변환 유형별 전용 디코더를 도입하면서 conditioning을 유지하여, Top-3 정확도 80.0%(v2 대비 +5.2pp), Top-5 83.3%, 최고 MRR 0.639를 달성한다. VAE, IsolationForest, OneClass-SVM을 비교하는 ablation 연구는 VAE 재구성 오차만이 합성-실제 분포 이동을 초월하는 전이 가능한 GNN 피처를 제공함을 실험으로 확인한다.

**핵심어:** 금융 리스크 파이프라인, 데이터 계보, 근본 원인 분석, 이상 탐지, 변분 오토인코더, 그래프 신경망, BCBS 239, 재시작 확률을 갖는 랜덤 워크

---

---

# Chapter 1. Introduction

## 1.1 Research Background

### 1.1.1 The BCBS 239 Compliance Gap

In January 2013, the Basel Committee on Banking Supervision published BCBS 239, formally titled *Principles for effective risk data aggregation and risk reporting*. The regulation mandates that globally systemically important banks (G-SIBs) ensure the accuracy, completeness, and timeliness of their risk data, and demonstrate end-to-end traceability from raw source systems to regulatory reports. The deadline for initial compliance was January 2016.

More than a decade later, the gap between regulation and practice remains striking. The Bank for International Settlements' 2025 implementation review reports that among 31 G-SIBs, only two institutions are in full compliance, and no single principle has achieved universal compliance across the peer group. The European Central Bank's 2024 Risk Data Aggregation and Risk Reporting guide explicitly names attribute-level data lineage as one of seven supervisory priority areas requiring sustained attention. Industry surveys confirm that structural data governance deficiencies persist across the majority of G-SIBs, with data lineage cited as the single most persistent technical challenge.

The practical difficulty is real. Financial risk pipelines are multi-stage systems in which raw market data—equity prices, interest rates, foreign exchange rates, credit spreads—undergoes dozens of sequential transformations before reaching regulatory reports. These stages include data cleaning and normalization, currency conversion, feature engineering, risk metric calculation (VaR, Expected Shortfall, sensitivities), and compliance reporting. Each stage introduces potential failure points. When a reported Value-at-Risk figure is incorrect, identifying which upstream transformation introduced the error requires traversing a complex dependency graph, a task that today remains largely manual and may take hours to days of investigation by skilled quantitative analysts.

### 1.1.2 Heightened Stakes Under FRTB 2025

The Fundamental Review of the Trading Book (FRTB), mandatory for G-SIBs from January 2025, substantially increases both the computational complexity and the auditability requirements of risk pipelines. Two FRTB provisions directly amplify the need for automated data lineage root cause analysis.

First, the **P&L Attribution Test (PLAT)** requires firms to demonstrate monthly that their Internal Model Approach risk-theoretical P&L closely tracks the hypothetical P&L computed from front-office pricing. Failure causes automatic fallback to the Standardized Approach, which typically carries 40–60% higher capital requirements. A PLAT failure is, at its root, a data lineage discrepancy: some upstream transformation produced risk figures inconsistent with the front-office source of truth. Identifying which pipeline node is responsible is precisely the problem this research addresses.

Second, FRTB's **Sensitivity-Based Approach** mandates risk factor sensitivity calculations at the individual instrument level, adding computation layers that were previously aggregated. This increases node counts in existing VaR pipelines proportionally, making manual root cause analysis even more burdensome.

These regulatory developments create a clear and urgent need for automated, interpretable root cause analysis tooling that can operate at financial pipeline scale and latency requirements.

### 1.1.3 The Fundamental Data Scarcity Problem

A distinctive challenge in applying machine learning to financial pipeline root cause analysis is the near-total absence of labeled anomaly data. Financial institutions experience pipeline failures infrequently—by design, given their operational criticality—and when failures occur, they are rarely labeled, classified, and archived in a machine-readable format suitable for supervised learning.

This is not merely a practical inconvenience. The Basel Committee's own surveys indicate that the majority of G-SIBs lack systematic incident tracking for data quality events in their risk pipelines. Commercial financial data providers such as Bloomberg Terminal (approximately USD 25,000 per terminal per year, institutional contracts only) and Refinitiv Eikon provide raw market time series but carry no anomaly annotations. Korean Exchange (KRX) data contracts are restricted to institutional counterparties and similarly provide no labeled failure events.

The consequence is that purely supervised machine learning approaches—classifiers trained on historical labeled incidents—are structurally infeasible for this domain. Any viable machine learning component must operate without relying on real labeled anomaly data.

## 1.2 Research Motivation

The author's direct experience at KB Kookmin Bank's StarBanking Operations Team provided concrete motivation for this research. When discrepancies arose in reported risk metrics, operational teams spent hours to days tracing data flows across disparate systems, relying on institutional knowledge and ad hoc SQL queries rather than systematic tooling. Root cause identification was a manual, expertise-dependent process with no reproducible methodology.

This experience motivates the central research question:

> *Can the root cause of a data quality anomaly in a financial risk reporting pipeline be identified automatically, rapidly, and accurately using a data lineage graph—without requiring labeled anomaly data?*

## 1.3 Research Objectives and Scope

This thesis pursues four objectives:

**Objective 1 — Formal Representation.** Define a column-level lineage graph schema specifically designed for financial risk computation pipelines, capturing the five-layer node hierarchy and four-type edge taxonomy that characterize VaR/ES computation systems.

**Objective 2 — Automated RCA Algorithm.** Design and implement a root cause analysis algorithm that exploits both graph structure and node-level anomaly signals to rank candidate root causes, with emphasis on interpretability and sub-second latency.

**Objective 3 — Reproducible Evaluation Infrastructure.** Construct a reproducible benchmark with controlled anomaly injection across a taxonomy of seven financial anomaly types, and validate against real market data from multiple equity markets.

**Objective 4 — ML Integration Without Label Dependency.** Investigate how machine learning components can augment the RCA pipeline without requiring labeled anomaly data, and empirically characterize which unsupervised methods can generalize across synthetic-to-real distribution shifts.

### Scope

The system targets **financial risk data pipelines** in the sense of BCBS 239: multi-stage computational systems that transform raw market data into regulatory risk reports. The anomaly taxonomy covers seven types spanning source data corruption, ETL logic errors, feature calculation bugs, aggregation errors, downstream masking, multi-source interference, and coincidental anomalies. The evaluation covers pipelines of 45–100+ nodes. Real-time streaming pipelines and high-frequency trading systems are outside scope.

## 1.4 Contributions

This thesis makes six contributions:

**C1 — Financial Risk Lineage Graph (FRLG).** A column-level lineage graph schema for financial risk pipelines, capturing five node types (Source, ETL, Feature, Risk, Report) and four edge types (DirectMap, Filter, Calculate, Aggregate). This is, to the author's knowledge, the first lineage schema specifically designed for financial risk computation as opposed to general-purpose metadata catalogs.

**C2 — Hop-Distance-Adaptive APA-RCA.** An Anomaly-Propagation-Aware Root Cause Analysis algorithm that extends Random Walk with Restart with a novel hop-distance-adaptive weighting scheme. Unlike all prior graph-walk RCA methods (MicroRCA, CloudRanger, CORAL, MULAN, REASON), which use fixed transition weights, APA-RCA v2 adapts restart probability based on hop distance from the observed anomaly target. A companion five-signal anomaly detector combines causal excess isolation, peer deviation scoring, short-long variance ratio, rolling z-score, and segment variance signals.

**C3 — Schema-Constraint RCA and ML Enhancement Pipeline.** A lightweight O(|V|) structural integrity check on the FRLG that detects aggregation schema violations (A4 anomaly type) with 100% Top-1 accuracy and zero false positives on non-A4 types. Additionally, a three-version ML enhancement pipeline: v1 (VAE + GNN), v2 (C-VAE + ResGCN), and v3 (Hybrid TSD-CVAE + ResGCN), progressively encoding ATF domain theory into the model architecture through conditioning design (v2) and type-specific decoder routing (v3).

**C4 — SynFRP Benchmark.** A configurable synthetic financial risk pipeline benchmark with GBM/Vasicek/Random-Walk data generation and a controlled seven-type anomaly injection protocol, providing ground truth for 630 experimental runs across three pipeline scales.

**C5 — Anomaly Transfer Function (ATF) Framework.** A formal model of anomaly propagation through financial pipeline transforms, yielding closed-form transfer coefficients per transform type. The ATF provides principled values for APA-RCA's parameters and derives a distribution-free temporal discriminant that achieves perfect separation of A3 (feature calculation bug) from A5 (downstream masking) on both synthetic and real data.

**C6 — Real-Source Hybrid Benchmark (RSHB) and ML Validation.** An expanded real-market evaluation benchmark covering 12 instruments across US equities, Korean Exchange equities, foreign exchange, and Treasury rates over 756 trading days spanning three market regimes. The RSHB serves as the cross-domain transfer evaluation dataset for a VAE + GNN ML enhancement pipeline, demonstrating +21.9 percentage points Top-1 improvement without fine-tuning on real data. The v3 pipeline (Hybrid TSD-CVAE + ResGCN) achieves the highest Top-3 of 80.0% (+14.3 pp over v1) and MRR of 0.639, while also recovering A4 accuracy from 53.3% (v2) to 70.0% through type-specific decoder separation.

## 1.5 Thesis Organization

The remainder of this thesis is structured as follows.

**Chapter 2** surveys related work in four areas: data lineage and provenance systems, root cause analysis methods, graph neural networks applied to RCA, and unsupervised anomaly detection methods.

**Chapter 3** defines the system design and problem formulation: the financial risk pipeline model, the FRLG schema, the multi-signal anomaly detector, the Anomaly Transfer Function framework, and the evaluation benchmarks (SynFRP and RSHB).

**Chapter 4** presents the APA-RCA algorithm in detail, including the hop-distance-adaptive RWR formulation, the Schema-Constraint RCA module, and the OpenLineage compatibility layer.

**Chapter 5** describes the ML-enhanced pipeline: the VAE/C-VAE anomaly scorer, the GNN/ResGCN re-ranker, the ATF-aligned feature design, the cross-domain transfer protocol, and the ablation study comparing VAE against alternative unsupervised scorers (IsolationForest, OneClass-SVM).

**Chapter 6** presents experimental results on the SynFRP benchmark and the RSHB, including per-type breakdown, scalability analysis, and architectural enhancement analysis.

**Chapter 7** concludes with a summary of findings, limitations, and directions for future research.

---

# Chapter 2. Related Work

## 2.1 Data Lineage and Provenance Systems

### 2.1.1 Foundational Provenance Research

The concept of data provenance—tracking the origin and transformation history of data—has been studied in the database and scientific workflow communities since the early 2000s. Cheney et al. provide a foundational survey distinguishing three classes of provenance: *why-provenance* (which input tuples contributed to an output), *where-provenance* (which input cells were copied), and *how-provenance* (the transformation derivation tree). Davidson and Freire survey provenance in scientific workflows, establishing the importance of lineage for reproducibility and error diagnosis.

These foundational works establish that data lineage graphs—directed acyclic graphs in which nodes represent data artifacts or transformations and edges represent derivation relationships—are the canonical data structure for provenance tracking. This thesis adopts this representation as the basis for the FRLG.

### 2.1.2 Industrial Lineage Platforms

Industrial lineage tools have matured substantially in the past decade. Apache Atlas provides a metadata governance framework for Hadoop ecosystems, with lineage tracking at the dataset level. OpenLineage defines an open standard for capturing lineage events emitted by data pipelines (Airflow, Spark, dbt), with a schema based on *Job* (transformation) and *Dataset* (data artifact) entities. Atlan and Solidatus offer commercial data catalog products with lineage visualization capabilities.

These platforms share a common limitation relative to the goals of this thesis: **none provides root cause analysis capability**. They record *what* lineage exists but offer no automated method for identifying *which* node caused an observed anomaly. Furthermore, none has been evaluated with published RCA accuracy metrics. This thesis bridges the gap between lineage *recording* (what existing tools do) and lineage *exploitation for diagnosis* (what this thesis introduces).

A secondary limitation is granularity. Most industrial tools operate at the *dataset* level (table or file), while FRLG operates at the *column* level, capturing which specific metrics flow through which transformation nodes. Column-level lineage is essential for financial pipelines where a single table may contain dozens of metrics with distinct data quality characteristics.

### 2.1.3 Financial Domain Specificity

No published academic or industrial work addresses lineage-based root cause analysis specifically for financial risk computation pipelines in the BCBS 239 sense. The closest industry artifacts are internal documentation from large financial institutions describing manual RCA procedures, none of which is publicly available. This thesis fills this gap.

## 2.2 Anomaly Detection in Financial Time Series

### 2.2.1 Classical Statistical Methods

The most widely deployed anomaly detection methods in operational financial systems rely on rolling z-scores and control chart methods. A z-score detector flags observations whose deviation from the rolling mean exceeds a threshold in units of rolling standard deviation. These methods are computationally efficient and interpretable, but carry two well-known limitations in financial contexts: (1) the assumption of Gaussian-distributed residuals conflicts with the fat-tail distributions empirically observed in financial return series, and (2) fixed thresholds require manual calibration per node, which is impractical at pipeline scale.

The Augmented Dickey-Fuller test and KPSS test detect structural breaks and non-stationarity in time series, which are relevant signals for some financial anomaly types but do not directly localize anomalies to specific pipeline nodes.

### 2.2.2 Machine Learning Approaches

Isolation Forest (Liu et al., 2008) constructs an ensemble of random isolation trees, assigning anomaly scores inversely proportional to the average path length required to isolate a point. It is effective for point anomalies in moderate-dimensional feature spaces and requires no labeled data. However, its score is a function of the partition structure learned from training data; under significant distribution shift between training and deployment contexts, the partition boundaries may not transfer.

One-Class Support Vector Machines learn a decision boundary in a kernel-induced feature space that encloses the training data, assigning positive scores to in-distribution points and negative scores to anomalies. Scholkopf et al. establish the theoretical foundations. Practical deployment at large scale is complicated by the O(n²) training complexity of kernel SVM; SGD-based approximations using Nyström kernel maps address this at the cost of some approximation error.

Variational Autoencoders (Kingma and Welling, 2013) train a generative model to encode input data into a low-dimensional latent distribution and reconstruct the input from samples thereof. Anomaly detection uses the reconstruction error as the anomaly score: inputs that deviate from the learned normal manifold incur high reconstruction error. The key property relevant to this thesis is that VAE reconstruction error is a *continuous, learned representation* of deviation from normality. Unlike IF and OCSVM boundary scores, which are sensitive to the precise location of learned boundaries, reconstruction error magnitude is more stable across distribution shifts because it measures deviation from a manifold rather than distance to a boundary.

Anomaly Transformer (Xu et al., 2022) introduces an attention-based anomaly detection mechanism for multivariate time series, achieving state-of-the-art performance on several benchmarks. MTAD-GAT (Zhao et al., 2020) combines temporal convolutions with graph attention networks for multivariate anomaly detection.

### 2.2.3 Relevance to Financial Pipeline Anomaly Detection

The distinguishing requirement in this thesis's setting is that anomaly detection must operate **without any labeled anomaly data**—a constraint that rules out supervised methods—and the resulting scores must serve as **transferable features** for a downstream GNN re-ranker trained on synthetic data and applied to real data. This latter requirement (cross-domain feature quality) explains the experimental finding that VAE substantially outperforms IF and OCSVM as anomaly scorers in this context, which is analyzed in detail in Chapter 5.

## 2.3 Root Cause Analysis

### 2.3.1 Graph-Walk Methods in Cloud and Microservices

The dominant paradigm for automated root cause analysis in production systems is graph-walk over a dependency or causal graph. MonitorRank (Kim et al., 2013) pioneered random walk on sensor dependency graphs at LinkedIn, establishing the propagation-walk paradigm. CloudRanger (Wang et al., 2018) applied second-order random walk on PC-algorithm causal graphs for cloud-native root cause localization. MicroRCA (Wu et al., 2020) introduced Personalized PageRank on attributed service topology graphs, achieving 89% precision and becoming the canonical reference for graph-walk RCA.

GROOT (Wang et al., 2021) deployed GrootRank (a PageRank variant) on event causal graphs in production e-commerce, achieving 78% Top-1 accuracy. More recent methods include CIRCA (Li et al., 2022), which uses causal Bayesian networks with intervention recognition; CORAL and REASON (Wang et al., 2023), which apply incremental and hierarchical causal graph methods with RWR; and MULAN (Zheng et al., 2024), which uses multi-modal causal learning with RWR.

Nezha (Yu et al., 2023) constructs a multi-modal event graph from metrics, logs, and traces. Chain-of-Event (Yao et al., 2024) builds weighted causal graphs from deployment history. RCACopilot (Chen et al., 2024) uses large language models for incident root-cause generation. BARO (Pham et al., 2024) applies Bayesian change-point detection. AERCA (Xiao et al., 2025, ICLR Oral) uses Granger causal discovery to identify anomaly-introducing interventions.

**Critical distinction.** All of the above methods target *system failures*—service crashes, latency spikes, resource exhaustion—in microservice or cloud environments. None addresses *data quality anomalies propagating through financial risk computation pipelines*, and none uses data lineage graphs as the propagation substrate. The anomaly signals, graph construction methods, and domain-specific assumptions are fundamentally different from the financial pipeline setting.

### 2.3.2 Causal Discovery Methods

Pearl's do-calculus and structural causal models provide the theoretical foundation for causal reasoning from observational data. PC algorithm (Spirtes et al.) and related constraint-based methods learn causal DAGs from conditional independence tests. GES (Chickering) uses score-based search. These methods require either sufficient observational data for reliable conditional independence testing or interventional data—conditions that are difficult to satisfy in financial pipeline settings where anomaly events are rare.

### 2.3.3 RCA Benchmarks

RCAEval (Pham et al., 2025) provides the most comprehensive benchmark for cloud RCA, covering 14 methods across multiple microservice datasets. No equivalent benchmark exists for financial pipeline RCA prior to this thesis's SynFRP contribution.

### 2.3.4 APA-RCA in Relation to Prior Work

The primary algorithmic distinction of APA-RCA is the **hop-distance-adaptive restart probability**. In standard RWR, the restart probability c is a fixed scalar applied uniformly at each step. All prior graph-walk RCA methods (MicroRCA, CloudRanger, CORAL, MULAN, REASON) use fixed transition weights. APA-RCA v2 computes the restart probability as a function of the hop distance from the observed anomaly target, reflecting the empirical observation that nodes close to the observed anomaly are more likely to be symptomatic (high graph-structure trust) while distant ancestors require stronger direct anomaly evidence to be ranked highly.

## 2.4 Graph Neural Networks for RCA

### 2.4.1 GNN Foundations

Graph Neural Networks (Kipf and Welling, 2017; Hamilton et al., 2017) extend neural network computation to graph-structured data through neighborhood aggregation. Graph Convolutional Networks (GCN) use symmetric-normalized adjacency matrices to propagate node features across edges. Graph Attention Networks (GAT, Veličković et al., 2018) weight neighborhood contributions via learned attention coefficients. GraphSAGE (Hamilton et al., 2017) generalizes to inductive settings by learning neighborhood sampling and aggregation functions.

### 2.4.2 GNNs Applied to Root Cause Analysis

Li et al. propose GNN-based root cause localization for general data pipelines, learning to rank candidate nodes using graph structure and node features. Wang et al. propose hierarchical GNNs for causal discovery. These works demonstrate that GNNs can learn structural patterns useful for RCA, but neither targets financial regulatory pipelines nor provides a domain-specific anomaly taxonomy.

A key limitation of supervised GNN approaches for RCA is the requirement for labeled training data—known root causes for historical incidents. This thesis addresses this through **cross-domain transfer**: the GNN is trained on synthetic data where injection locations provide perfect labels, and applied to real data without fine-tuning. The viability of this transfer depends critically on the quality of node features, which is the motivation for using VAE reconstruction error rather than simpler anomaly scores.

### 2.4.3 Feature Design for GNN Re-ranking

The node feature design for the GNN re-ranker in this thesis draws on both domain knowledge and the APA-RCA output. Eight features are used: (1) the normalized RWR score from APA-RCA, (2) the VAE anomaly score (C-VAE in v2), (3) the normalized layer position (L1–L5), (4) whether the node is an ancestor of the observed target, (5) normalized in-degree, (6) normalized out-degree, (7) whether the node appears in the Top-10 candidate set, and (8) the ATF-weighted path distance from the node to the target, using the same $\rho_\tau$ transfer coefficients as the APA-RCA transition matrix. Features (3)–(8) are structural and domain-specific; their values are determined by the pipeline graph topology, which is identical in structure between synthetic and real pipelines. This structural invariance is the theoretical basis for cross-domain generalization. Note that v2 uses C-VAE instead of VAE for Feature (2), making the feature set 8-dimensional across both versions.

## 2.5 Unsupervised Anomaly Scoring: Comparative Perspective

This section positions the VAE, IsolationForest, and OneClass-SVM anomaly scorers used in the ablation study of Chapter 5.

| Method | Score Type | Training Requirement | Cross-domain Stability |
|---|---|---|---|
| Rolling z-score | Statistical threshold | None (parametric) | Low (distributional assumptions) |
| Isolation Forest | Tree partition depth | Anomaly-free data | Low (boundary-sensitive) |
| OneClass-SVM | Decision boundary distance | Anomaly-free data | Low (boundary-sensitive) |
| VAE | Reconstruction error | Anomaly-free data | High (manifold deviation) |

The key theoretical distinction is between *boundary-based* scorers (IF, OCSVM) and *manifold-based* scorers (VAE). Boundary-based methods learn a partition or decision surface in feature space; their scores reflect proximity to this surface, which is sensitive to the training distribution. Under distribution shift between synthetic and real financial data, the learned boundaries may not transfer. VAE reconstruction error measures deviation from a *learned manifold of normal behavior*, a quantity whose relative magnitude—higher for more anomalous nodes—is more stable across distribution shifts.

This theoretical distinction is empirically confirmed in Chapter 5: IF and OCSVM achieve near-zero Top-1 accuracy on RSHB (0.5% and 0.0% respectively), while VAE achieves 54.8%.

## 2.6 Positioning of This Work

Table 2.1 summarizes the positioning of FINRCA relative to the closest related works.

**Table 2.1: Positioning of FINRCA.**

| System | Domain | Lineage Graph | No Label Required | Financial Benchmark | Real-Market Validation |
|---|---|---|---|---|---|
| MicroRCA | Cloud | Service topology | ✗ | ✗ | ✗ |
| CORAL/REASON | Cloud | Causal graph | ✗ | ✗ | ✗ |
| AERCA | General TS | — | ✗ | ✗ | ✗ |
| Li et al. GNN | Data pipeline | DAG | ✗ | ✗ | ✗ |
| **FINRCA (This thesis)** | **Financial risk** | **FRLG (column-level)** | **✓** | **✓ (SynFRP + RSHB)** | **✓ (US + KRX)** |

FINRCA is the first system to combine (1) a domain-specific column-level lineage graph schema, (2) a label-free ML pipeline for anomaly scoring and re-ranking, and (3) real-market validation across two equity markets, in the specific context of BCBS 239 financial risk pipeline compliance.

---

# Chapter 3. System Design and Problem Formulation

## 3.1 Financial Risk Pipeline Architecture

### 3.1.1 Pipeline Structure

A **financial risk pipeline** $P = (N, D)$ consists of a set of processing nodes $N$ and directed data-flow edges $D \subseteq N \times N$. Nodes are organized in five functional layers reflecting the computational stages of a VaR/ES reporting system:

- **L1 — Source**: Raw market data ingestion nodes. Each node represents a single instrument's time series (e.g., equity price, FX rate, interest rate). Source nodes have no upstream dependencies within the pipeline.
- **L2 — ETL**: Extract-Transform-Load nodes that perform data cleaning, currency conversion, and normalization.
- **L3 — Feature**: Feature engineering nodes that compute derived quantities such as log-returns, volatility estimates, and correlation metrics.
- **L4 — Risk**: Risk metric computation nodes, including Value-at-Risk, Expected Shortfall, and Greek sensitivities.
- **L5 — Report**: Regulatory reporting nodes that aggregate risk metrics into final report figures subject to BCBS 239 auditability requirements.

Data flows strictly from lower to higher layers (L1 → L5), making the pipeline a directed acyclic graph. In practice, pipelines range from approximately 45 nodes (small, 3-instrument configuration) to over 100 nodes (large, 8+ instrument configuration).

### 3.1.2 Transform Types

Each directed edge $(u, v) \in D$ corresponds to a data transformation applied at node $v$ to inputs from node $u$. Four transform types are distinguished:

- **DirectMap**: Identity or scalar multiplication. The input series passes through unchanged in distribution.
- **Filter**: Clipping or threshold-based transformations that may suppress large values.
- **Calculate**: Mathematical operations producing derived series (log-return, rolling statistics).
- **Aggregate**: Multi-input aggregations (weighted mean, portfolio sum) that combine multiple upstream series.

The distinction between transform types is critical for anomaly propagation: different transforms attenuate anomaly signals by different amounts. This observation is formalized in the Anomaly Transfer Function framework (Section 3.4).

### 3.1.3 Pipeline Scale

Three pipeline scales are used in evaluation:

| Scale | Stocks | Rates | FX | Positions | Approx. Nodes |
|---|---|---|---|---|---|
| Small | 3 | 2 | 1 | 3 | ~45 |
| Medium | 5 | 3 | 2 | 5 | ~65 |
| Large | 8 | 3 | 3 | 8 | ~100+ |

## 3.2 Financial Risk Lineage Graph (FRLG)

### 3.2.1 Formal Definition

The **Financial Risk Lineage Graph** is a labeled directed acyclic graph:

$$\mathcal{G} = (V, E, \ell_V, \ell_E)$$

where:
- $V = N$ is the node set (pipeline processing nodes)
- $E = D$ is the edge set (data flow dependencies)
- $\ell_V : V \to \{\text{Source, ETL, Feature, Risk, Report}\} \times [0,1]$ assigns each node its layer type and anomaly score
- $\ell_E : E \to \{\text{DirectMap, Filter, Calculate, Aggregate}\}$ assigns each edge its transform type

The FRLG is constructed automatically from a pipeline definition: for each processing node in the pipeline, one graph node is created; for each data dependency between nodes, one directed edge is added with the transform type inferred from the node's computation type.

### 3.2.2 Edge Weight Prior

Structural edge weights $\beta_e$ encode the domain prior that different transform types carry different causal significance for anomaly propagation:

$$\beta(\text{Aggregate}) = 1.5, \quad \beta(\text{Report}) = 1.3, \quad \beta(\text{Calculate}) = 1.2$$
$$\beta(\text{DirectMap}) = 1.0, \quad \beta(\text{Filter}) = 0.8$$

These weights reflect two domain insights: (1) aggregation operations that combine multiple upstream series are high-signal causal links—anomalies at an aggregation node often have a specific upstream cause; (2) filter operations may suppress anomaly signals, making them lower-weight candidates for causal chains.

### 3.2.3 OpenLineage Compatibility

A practical deployment requirement is compatibility with existing data lineage infrastructure. Table 3.1 shows the field-level mapping between the OpenLineage RunEvent schema and the FRLG node/edge schema. Each OpenLineage Job maps to one FRLG node; each InputDataset/OutputDataset pair defines an edge. Column-level lineage facets map directly to FRLG column_in/column_out fields.

**Table 3.1: OpenLineage to FRLG field mapping.**

| OpenLineage Field | FRLG Target | Notes |
|---|---|---|
| job.name | node_id | Unique identifier |
| JobTypeJobFacet.jobType | transform_type | SQL→calculate, AGG→aggregate |
| inputs[].name | upstream edge source | (src, dst) edge |
| outputs[].name | downstream edge target | (src, dst) edge |
| ColumnLineageDatasetFacet.fields | column_in/out | Column-level lineage |
| DataSourceDatasetFacet.sourceType | node_type | SOURCE vs TRANSFORM |

This mapping confirms that FINRCA can be deployed on any OpenLineage-compliant pipeline without custom instrumentation.

## 3.3 Multi-Signal Anomaly Detector

For each node $v \in V$ with output time series $x_v$, an anomaly score $a(v) \in [0,1]$ is computed by combining five complementary signals.

### 3.3.1 Signal Definitions

**S1 — Rolling z-score.** The proportion and magnitude of observations deviating from the rolling mean:

$$s_z(v) = 0.6 \cdot \frac{|\{t : |z(t)| > \theta\}|}{T} + 0.4 \cdot \min\!\left(\frac{\max_t|z(t)|}{3\theta}, 1\right)$$

where $z(t) = (x(t) - \mu_{\text{roll}}(t)) / \sigma_{\text{roll}}(t)$ and $\theta = 2.0$.

**S2 — Short-long variance ratio.** Detects volatility regime changes:

$$s_{\text{sl}}(v) = \min\!\left(\frac{\sigma_5(v) / (\sigma_{30}(v) + \varepsilon) - 1}{5}, 1\right)$$

where $\sigma_k$ denotes the median $k$-day rolling standard deviation.

**S3 — Segment variance.** Detects structural breaks by comparing variance across temporal segments:

$$s_v(v) = 0.8 \cdot \min\!\left(\frac{\sigma_{\max}/\sigma_{\min} - 1}{10}, 1\right)$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are the maximum and minimum variance across four equal temporal quarters.

**S4 — Peer deviation with CV normalization.** Detects nodes that deviate from their peer group (nodes in the same layer, metric type, and instrument family):

$$s_{\text{peer}}(v) = \min\!\left(\frac{|\text{CV}(v) - \overline{\text{CV}}|}{\text{std}(\text{CV}) + \varepsilon \cdot 5}, 1\right)$$

where $\text{CV}(v) = \sigma(x_v)/(|\mu(x_v)| + \varepsilon)$ is the Coefficient of Variation, providing scale-invariant peer comparison.

**S5 — Causal excess.** Isolates nodes whose anomaly score exceeds what is explainable by upstream propagation:

$$\text{excess}(v) = a_{\text{raw}}(v) - 0.7 \cdot \max_{u \in \text{upstream}(v)} a_{\text{raw}}(u)$$

where $a_{\text{raw}}$ uses only S1–S3. Source nodes with no upstream receive $\text{excess} = a_{\text{raw}}$. This signal is crucial for distinguishing root causes (high excess) from symptom nodes that merely propagate upstream anomalies.

### 3.3.2 Layer-Aware Fusion

The five signals are fused using layer-specific weights reflecting the diagnostic value of each signal at each pipeline stage:

$$a(v) = \max(s_z, s_{\text{sl}} \cdot 0.5, s_{\text{peer}} \cdot 0.75) \quad \text{if layer}(v) \in \{1, 2\}$$

$$a(v) = \max(s_z \cdot 0.55, s_{\text{causal}} \cdot 0.85, s_{\text{sl}} \cdot 0.80, s_{\text{peer\_causal}} \cdot 0.90) \quad \text{if layer}(v) = 3$$

$$a(v) = \max(s_z \cdot 0.40, s_{\text{causal}} \cdot 0.60, s_{\text{peer}} \cdot 0.35) \quad \text{if layer}(v) \in \{4, 5\}$$

where $s_{\text{peer\_causal}} = s_{\text{peer}} \cdot \min(\text{excess}/0.3, 1)$.

The layer-aware design reflects domain knowledge: at source and ETL layers, peer deviation and short-long variance are primary signals; at the feature layer, causal excess is most informative for distinguishing root causes from downstream effects; at risk and report layers, anomaly scores are expected to reflect upstream propagation rather than local causation.

## 3.4 Anomaly Transfer Function Framework

### 3.4.1 Motivation

A systematic model of how anomalies propagate through pipeline transforms provides principled values for APA-RCA's algorithmic parameters and explains empirical performance patterns.

### 3.4.2 Definition

The **Anomaly Transfer Function** (ATF) for transform type $\tau$ is defined as:

$$\rho_\tau = \frac{\|f_\tau(x + \delta) - f_\tau(x)\|_z}{\|\delta\|_z}$$

where $\|s\|_z = \max_t |s_t - \mu_s|/\sigma_s$ is the maximum z-score of series $s$. The coefficient $\rho_\tau$ quantifies what fraction of an input anomaly's z-score magnitude survives the transform.

### 3.4.3 Per-Transform Coefficients

- **Source / DirectMap ($\rho = 1.0$)**: Identity or scalar multiplication; anomaly passes through unchanged.
- **Filter—clip at $c$ std ($\rho = \min(1, c/z_{\text{in}})$)**: Anomalies within the clip boundary pass through; larger spikes are attenuated. For $c = 4$ and input z-score 8: $\rho = 0.5$.
- **Calculate—log-return ($\rho \approx 1.0$)**: A spike at $t_0$ produces non-zero ATF at both $t_0$ and $t_0{+}1$, doubling the temporal footprint.
- **Aggregate—mean of $k$ inputs ($\rho = 1/\sqrt{k}$)**: For i.i.d. inputs with one anomalous: $\rho = 1/\sqrt{k}$. For $k = 5$ (medium pipeline): $\rho \approx 0.45$.
- **Report—binary threshold ($\rho \approx 0$)**: Discontinuous output carries near-zero z-score information.

### 3.4.4 Implications for Algorithm Design

The ATF framework provides three contributions to APA-RCA design. First, it derives principled edge weights ($\beta$) matching the empirically-tuned values. Second, it derives the hop attenuation parameter $\alpha^* \approx 0.72$, matching the hand-tuned value $\alpha = 0.7$. Third, it derives a distribution-free temporal discriminant $\phi_{\text{peer}}$ that achieves perfect separation between A3 (feature calculation bugs, which produce persistent anomalies across all time steps) and A5 (downstream masking, which produces transient spikes). The discriminant $\phi_{\text{peer}} > 0.5$ correctly classifies all A3 and A5 cases on both synthetic and real data.

## 3.5 SynFRP Benchmark

### 3.5.1 Benchmark Design

The **Synthetic Financial Risk Pipeline (SynFRP)** benchmark provides a controlled evaluation environment with ground-truth root cause labels. The benchmark generates synthetic financial time series using economically motivated stochastic processes:

- **Equity prices**: Geometric Brownian Motion (GBM) with sector-correlated drift and volatility
- **Interest rates**: Vasicek mean-reverting process with realistic long-run means
- **FX rates**: Random Walk with calibrated volatility

These processes produce time series with distributional properties (fat tails, volatility clustering, cross-instrument correlation) representative of real financial market data, while allowing exact control over anomaly injection.

### 3.5.2 Anomaly Taxonomy

Seven anomaly types spanning the full pipeline are defined:

| Type | Name | Description | Primary Layer |
|---|---|---|---|
| A1 | Source Corruption | Spike or level shift in raw market data | L1 |
| A2 | ETL Logic Error | Incorrect transformation coefficient or formula | L2 |
| A3 | Feature Calculation Error | Wrong window parameter or calculation bug | L3 |
| A4 | Aggregation Error | Schema violation—incorrect set of inputs aggregated | L4 |
| A5 | Downstream Masking | Upstream clipping suppresses anomaly signal | L3/L4 |
| A6 | Multi-Source | Anomalies at multiple source nodes with interference | L1 |
| A7 | Coincidental | Anomalous-looking but benign fluctuation | L1–L3 |

This taxonomy covers the primary failure modes observed in operational financial pipelines and is designed so that no single detection method is uniformly optimal across all types.

### 3.5.3 Experimental Protocol

The benchmark runs 630 trials: 7 anomaly types × 3 pipeline scales (small, medium, large) × 30 trials each. Each trial uses a distinct random seed, ensuring genuinely different pipeline data, injection targets, and anomaly characteristics. Anomaly magnitude is fixed at 8.0 z-score units, representing a clearly detectable but challenging injection. Performance is measured by Top-1, Top-3, Top-5 accuracy (whether the true root cause node appears in the top-k ranked candidates) and Mean Reciprocal Rank (MRR).

For the ML-enhanced pipeline's 5-fold cross-validation, trials are assigned to folds using stratified sampling by (anomaly_type × pipeline_size), ensuring each fold contains a balanced representation of experimental conditions. Training uses 504 trials; evaluation uses 126.

## 3.6 Real-Source Hybrid Benchmark (RSHB)

### 3.6.1 Design Rationale

The SynFRP benchmark uses synthetic market data. While the stochastic process models are econometrically motivated, real financial time series exhibit characteristics—non-stationarity, fat tails, regime changes, cross-market correlation asymmetries—that synthetic data may not fully capture. The Real-Source Hybrid Benchmark (RSHB) addresses this by replacing the source layer (L1) with actual historical market data while keeping the pipeline logic and anomaly injection protocol identical to SynFRP.

### 3.6.2 Data Configuration

The RSHB source layer uses 12 instruments drawn from Yahoo Finance over 756 trading days (approximately three years):

- **US equities (5)**: Five representative large-cap stocks spanning multiple sectors
- **Korean Exchange equities (2)**: Samsung Electronics and SK Hynix, representing the KRX market with distinct microstructure and partial trading-hour overlap with US markets
- **Foreign exchange rates (2)**: USD/KRW, EUR/USD
- **Treasury rates (3)**: US 2-year, 10-year, and 30-year yields

The 756-day window spans three distinct market regimes: the Federal Reserve tightening cycle (2022), the US regional banking stress period (2023), and the AI-driven equity rally (2023–2024). This regime diversity tests robustness across different volatility and correlation environments.

### 3.6.3 Experimental Protocol

The RSHB runs 210 trials: 7 anomaly types × 30 trials each, all using a medium-size pipeline. The same injection protocol as SynFRP is used, with injection magnitude fixed at 8.0 z-score units. Pipeline logic, graph construction, and evaluation metrics are identical to SynFRP.

The RSHB serves a dual purpose. First, it evaluates robustness of APA-RCA to real market data distribution (Chapter 6). Second, it serves as the cross-domain transfer test for the ML-enhanced pipeline: the GNN re-ranker is trained exclusively on the 630 synthetic trials and applied to RSHB without any fine-tuning, testing whether structural patterns learned on synthetic data generalize to real market conditions (Chapter 5).

---

*[End of Chapters 1–3 Draft]*

---

## Notes for HWP Formatting

- Apply school thesis template styles: 본문 (body text), 제목1 (Chapter heading), 제목2 (Section heading), 제목3 (Subsection heading)
- Tables: use the school's table style with borders top/bottom on header row
- Equations: if HWP equation editor is used, replicate the LaTeX formulas above
- References: citations marked as (Author, Year) above should be converted to the school's reference format (numbered or author-year as specified in guidelines)
- Figure/Table numbering: Chapter-prefixed (Table 3.1, Figure 3.1, etc.)
