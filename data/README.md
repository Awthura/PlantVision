# Data

This directory is gitignored. Populate it with `scripts/download_data.py`.

## PhenoBench v1.1.0

| Field       | Value |
|-------------|-------|
| Source      | University of Bonn — [phenobench.org/dataset.html](https://phenobench.org/dataset.html) |
| Content     | UAV imagery of sugar-beet fields (two growing seasons, varied lighting) |
| Labels      | Dense pixel-wise: `semantics` (crop/weed), `plant_instances`, `leaf_instances` |
| Size        | ~1,407 train + 772 val images; ~7.6 GB unzipped |
| Masks       | uint16 PNG: semantics (0=bg,1=crop,2=weed,3=partial-crop,4=partial-weed), plant_instances, leaf_instances |
| License     | **CC BY-NC-SA 4.0** — non-commercial, share-alike |

### Citation

```
@article{weyler2023dataset,
  author = {Jan Weyler and Federico Magistri and Elias Marks and Yue Linn Chong and Matteo Sodano
            and Gianmarco Roggiolani and Nived Chebrolu and Cyrill Stachniss and Jens Behley},
  title  = {{PhenoBench --- A Large Dataset and Benchmarks for Semantic Image Interpretation
             in the Agricultural Domain}},
  journal = {arXiv preprint},
  year    = {2023}
}
```
