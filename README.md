## 1. Codebase structure 
```
scHIIT/
├── src/
│   ├── engines/
│   │   ├── CohortEngine.py          # Main cohort processing engine
│   │   └── io/
│   │       ├── cell_ontology_utils.py
│   │       ├── cellxgene_pp_utils.py
│   │       ├── integration_utils.py
│   │       └── sampler_utils.py
│   │
│   └── methods/
│       ├── bm_methods/              # Benchmarking methods
│       │   ├── base_method.py       # Base class for all methods
│       │   ├── wilcoxon_method.py
│       │   ├── cosg_method.py
│       │   └── ...                  # Additional benchmark methods
│       │
│       └── schiit_method/           # scHIIT pipeline implementation
│           ├── __init__.py
│           ├── network
│           |    └── grn_base.py     # network creation for transcriptional core 
│           ├── tf_filters
|                ├── base_filter.py  # base filter 
│                ├── jsd_filter.py  # GJSD based algorithm
|                └── fast_oi_filter.py    # O information based identity TF algorithm 
│
├── demo_nb/
│   └── cellxgene_utils_demo.ipynb                   # cellxgene wrapper Usage examples
│   └── scHIIT_pipeline_v0.0.6.ipynb                 # scHIIT Usage examples
│
├── tests/                           # Unit tests
│
├── README.md
├── LICENSE
└── env_generic.yml
```

## 2. Usage
The repository is currently in private mode - request access by emailing abhilash.dhal@uni.lu

- Install the repository 

```git clone https://github.com/adhal007/scHIIT.git```

- Create conda environment for env_generic.yml 

```conda env create -f env_generic.yml```
```conda activate schiit_main```

- Run the first demo notebook in 

```demo_nb/scHIIT_pipeline_v0.1.3_main.ipynb```
