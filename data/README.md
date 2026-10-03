# Oxford-IIIT Pet Corruption Suite

Run the pipeline with:

```text
dvc repro
```

The suite creates three severities of each corruption for the official test
split. Outputs are written below `data/corrupted/`. The combined manifest at
`data/manifest.json` contains both clean source records and corrupted records.

| Corruption | Severity 1 | Severity 2 | Severity 3 |
|---|---:|---:|---:|
| Gaussian blur radius | 1 | 2 | 4 |
| Brightness up factor | 1.3 | 1.6 | 2.0 |
| Brightness down factor | 0.7 | 0.5 | 0.3 |
| JPEG quality | 50 | 30 | 10 |
| Downscale shorter side | 128 | 96 | 64 |
| Motion blur kernel size | 5 | 9 | 15 |

Downscale/upscale preserves the source aspect ratio: the shorter side is
reduced to the severity target, then the image is resized back to its original
dimensions.
