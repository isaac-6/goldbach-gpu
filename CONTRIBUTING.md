# Contributing to GoldbachGPU

Thank you for your interest in contributing to this project! As an open-source academic tool, we welcome community involvement.

## Reporting Bugs and Issues
If you encounter a bug, an Out-Of-Memory (OOM) error, or unexpected behavior, please open an issue in the [GitHub Issues tracker](https://github.com/isaac-6/goldbach-gpu/issues). Please include:
- Your OS and CUDA version.
- Your GPU model and VRAM.
- The exact command you ran.

## Seeking Support
If you have questions about the mathematics, the architecture, or how to adapt the codebase for other HPC clusters, please open a "Question" issue or reach out via email (listed in the accompanying paper).

## Contributing Code
1. Fork the repository.
2. Create a new branch for your feature or optimization (`git checkout -b feature-gpu-sieve`).
3. Ensure your code passes the automated validation suite (`cd tests && ./validation.sh`).
4. Submit a Pull Request detailing the performance improvements or bug fixes.