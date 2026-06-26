# Data

This directory is gitignored. Populate it with `scripts/download_data.py`.

## PhenoBench v1.1.0

| Field       | Value |
|-------------|-------|
| Source      | University of Bonn — [phenobench.org/dataset.html](https://phenobench.org/dataset.html) |
| Content     | UAV imagery of sugar-beet fields (two growing seasons, varied lighting) |
| Labels      | Dense pixel-wise: `semantics` (crop/weed), `plant_instances`, `leaf_instances` |
| Size        | ~1,407 train + 772 val images; ~7.6 GB unzipped |
| License     | CC BY 4.0 — see official page for exact terms |

### Citation

```
@dataset{phenobench2023,
  author    = {Weyler, Jonas and others},
  title     = {PhenoBench — A Large Dataset and Benchmarks for Semantic Image Interpretation of Sugar Beet Fields},
  year      = {2023},
  publisher = {University of Bonn},
  url       = {https://phenobench.org}
}
```
