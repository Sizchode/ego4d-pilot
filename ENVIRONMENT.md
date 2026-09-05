# Environment record

The local environment is used for CPU-safe protocol and schema checks only.
The HPC run directory must record:

- CUDA, PyTorch, Transformers, PyAV, and torchvision versions;
- immutable model and dataset revisions;
- model-specific official preprocessing/config revisions;
- random seed, GPU allocation, exact command, and C0 timestamp JSON;
- condition parameters, including the C2 offset and any C1 ratio.

Missing revisions or missing official canonical timestamps are hard errors.
