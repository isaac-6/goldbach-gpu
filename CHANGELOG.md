## v1.1.0 – 2026-03-02

**Added**
- Multi-GPU Goldbach verifier with work-stealing across devices, integrated as the new `goldbach_gpu3`.
- Multi-GPU benchmarking support and updated parameter tuning for modern GPUs (e.g., H100 SXM).

**Improved**
- CMake configuration now defaults to the native CUDA architecture and has cleaner, finalized targets.
- Repository layout: legacy tools and older GPU variants moved under legacy targets.

**Documentation**
- Expanded README with multi-GPU usage examples, including an 8× H100 cluster run.
- Updated build instructions, version badge, RESULTS log, and Zenodo DOIs in `README` and `CITATION.cff`.
- Added `CHANGELOG.md` to report version updates.