# Third-party provenance

`disaster_response/staeformer.py` is adapted from the authors' STAEformer implementation:

- Repository: https://github.com/XDZhelheim/STAEformer
- Paper: https://arxiv.org/abs/2308.10425
- Local adaptation: separate `encode`/`decode` methods and remove the optional demo.
  Attention, embeddings and the mixed projection are preserved.
- The GRU fusion and structured disaster response branch are additions in this project;
  they are not attributed to the STAEformer paper.

```bibtex
@inproceedings{liu2023spatio,
  title={Spatio-temporal adaptive embedding makes vanilla transformer sota for traffic forecasting},
  author={Liu, Hangchen and Dong, Zheng and Jiang, Renhe and Deng, Jiewen and Deng, Jinliang and Chen, Quanjun and Song, Xuan},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={4125--4129},
  year={2023}
}
```

The `SharedRateBank` compatibility initializer documents mathematical inspiration
from UMNN (Wehenkel & Louppe, 2019, https://arxiv.org/abs/1908.05164). It is a
locally written piecewise-rate adaptation, not an imported implementation.

NumPy and PyTorch are external dependencies installed separately, not vendored.
The archive does not relicense them or the STAEformer-derived source.
See [LICENSE_STATUS.md](LICENSE_STATUS.md) before public redistribution.
