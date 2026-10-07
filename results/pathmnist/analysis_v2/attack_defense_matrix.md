# Attack x Defense matrix (compact view)

Full data with all metrics (mean+-std, TPR/FPR, runtime) in `attack_defense_matrix.csv`. Compact tables below show final accuracy and divergence rate; Regime A (condition-specific) and `not_applicable` (non-detector) rows shown per partition; Regime B shown separately.


## Partition: iid (Regime A for detectors)

### Final accuracy (mean over 3 seeds) — iid

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | 0.442 | 0.256 | 0.419 | 0.442 | 0.444 | 0.447 |
| large_norm | 0.121 | 0.262 | 0.455 | 0.121 | 0.450 | 0.447 |
| low_norm | 0.429 | 0.262 | 0.418 | 0.429 | 0.439 | 0.437 |
| full_sign_flip | 0.391 | 0.259 | 0.410 | 0.437 | 0.437 | 0.437 |
| directional_poisoning | 0.391 | 0.259 | 0.410 | 0.437 | 0.436 | 0.437 |
| sparse_coordinate_attack | 0.306 | 0.261 | 0.460 | 0.460 | 0.459 | 0.460 |

### Divergence fraction — iid

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| large_norm | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| low_norm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| full_sign_flip | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| directional_poisoning | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sparse_coordinate_attack | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |


## Partition: dirichlet_a1.0 (Regime A for detectors)

### Final accuracy (mean over 3 seeds) — dirichlet_a1.0

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | 0.450 | 0.442 | 0.449 | 0.454 | 0.409 | 0.437 |
| large_norm | 0.077 | 0.428 | 0.140 | 0.077 | 0.408 | 0.422 |
| low_norm | 0.413 | 0.359 | 0.443 | 0.413 | 0.271 | 0.395 |
| full_sign_flip | 0.313 | 0.372 | 0.445 | 0.326 | 0.320 | 0.438 |
| directional_poisoning | 0.314 | 0.372 | 0.445 | 0.326 | 0.315 | 0.438 |
| sparse_coordinate_attack | 0.376 | 0.434 | 0.298 | 0.458 | 0.344 | 0.457 |

### Divergence fraction — dirichlet_a1.0

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| large_norm | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| low_norm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| full_sign_flip | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| directional_poisoning | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sparse_coordinate_attack | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |


## Partition: dirichlet_a0.5 (Regime A for detectors)

### Final accuracy (mean over 3 seeds) — dirichlet_a0.5

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | 0.409 | 0.352 | 0.382 | 0.405 | 0.294 | 0.378 |
| large_norm | 0.066 | 0.391 | 0.096 | 0.066 | 0.312 | 0.393 |
| low_norm | 0.364 | 0.326 | 0.375 | 0.364 | 0.264 | 0.385 |
| full_sign_flip | 0.275 | 0.301 | 0.375 | 0.275 | 0.299 | 0.372 |
| directional_poisoning | 0.276 | 0.301 | 0.378 | 0.274 | 0.298 | 0.374 |
| sparse_coordinate_attack | 0.257 | 0.395 | 0.225 | 0.383 | 0.288 | 0.402 |

### Divergence fraction — dirichlet_a0.5

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| large_norm | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| low_norm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| full_sign_flip | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| directional_poisoning | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sparse_coordinate_attack | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 |


## Partition: dirichlet_a0.1 (Regime A for detectors)

### Final accuracy (mean over 3 seeds) — dirichlet_a0.1

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | 0.196 | 0.193 | 0.205 | 0.196 | 0.207 | 0.194 |
| large_norm | 0.081 | 0.288 | 0.094 | 0.081 | 0.120 | 0.299 |
| low_norm | 0.243 | 0.248 | 0.246 | 0.266 | 0.104 | 0.251 |
| full_sign_flip | 0.111 | 0.247 | 0.305 | 0.111 | 0.096 | 0.261 |
| directional_poisoning | 0.111 | 0.280 | 0.290 | 0.111 | 0.102 | 0.312 |
| sparse_coordinate_attack | 0.154 | 0.275 | 0.125 | 0.276 | 0.104 | 0.282 |

### Divergence fraction — dirichlet_a0.1

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| large_norm | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| low_norm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| full_sign_flip | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| directional_poisoning | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sparse_coordinate_attack | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 |


## Regime B (IID-calibrated, frozen) — detectors only, non-IID partitions


### dirichlet_a1.0

### Final accuracy, Regime B — dirichlet_a1.0

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | - | 0.074 | 0.067 | 0.453 | - | - |
| large_norm | - | 0.074 | 0.067 | 0.077 | - | - |
| low_norm | - | 0.076 | 0.067 | 0.414 | - | - |
| full_sign_flip | - | 0.074 | 0.067 | 0.425 | - | - |
| directional_poisoning | - | 0.074 | 0.067 | 0.427 | - | - |
| sparse_coordinate_attack | - | 0.074 | 0.104 | 0.457 | - | - |


### dirichlet_a0.5

### Final accuracy, Regime B — dirichlet_a0.5

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | - | 0.074 | 0.063 | 0.406 | - | - |
| large_norm | - | 0.074 | 0.063 | 0.066 | - | - |
| low_norm | - | 0.100 | 0.063 | 0.375 | - | - |
| full_sign_flip | - | 0.074 | 0.063 | 0.309 | - | - |
| directional_poisoning | - | 0.074 | 0.063 | 0.310 | - | - |
| sparse_coordinate_attack | - | 0.074 | 0.063 | 0.384 | - | - |


### dirichlet_a0.1

### Final accuracy, Regime B — dirichlet_a0.1

| attack | fedavg | norm | cosine | sign_consensus | median | multi_krum |
|---|---|---|---|---|---|---|
| no_attack | - | 0.074 | 0.074 | 0.225 | - | - |
| large_norm | - | 0.074 | 0.074 | 0.080 | - | - |
| low_norm | - | 0.102 | 0.074 | 0.281 | - | - |
| full_sign_flip | - | 0.074 | 0.074 | 0.170 | - | - |
| directional_poisoning | - | 0.074 | 0.074 | 0.159 | - | - |
| sparse_coordinate_attack | - | 0.074 | 0.074 | 0.287 | - | - |
