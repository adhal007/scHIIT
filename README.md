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
│           ├── stage1_filter.py     # High expression & uniqueness
│           ├── stage2_specificity.py # JSD-based specificity
│           └── stage3_network.py    # Network-based identity core
│
├── demo_nb/
│   └── cellxgene_utils_demo.ipynb                   # cellxgene wrapper Usage examples
│   └── scHIIT_pipeline_v0.0.6.ipynb                 # scHIIT Usage examples
│
├── tests/                           # Unit tests
│
├── README.md
├── LICENSE
└── requirements.txt
```