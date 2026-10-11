# Oxford-IIIT Pet Corruption Suite

Run the pipeline with:

```text
dvc repro
```

The suite creates three severities of each corruption for the official test
split. Outputs are written below `data/corrupted/`. The committed split index
is `data/split_index.json`, and the combined manifest at
`data/processed/manifest.json` contains both clean source records and corrupted
records.

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

## Rebuilding without access to the DVC remote

The DVC remote is local to the owner's machine (configured in the git-ignored
`.dvc/config.local`), so a reviewer rebuilds the data from the official archives.
The md5 values are the ones recorded in `data/raw/*.tar.gz.dvc`:

| Archive | URL | md5 |
|---|---|---|
| `images.tar.gz` | https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz | `5c4f3ee8e5d25df40f4fd59a7f44e54c` |
| `annotations.tar.gz` | https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz | `95a8c909bbe2e81eed6a22bccdf3f68f` |

```bash
cd data/raw
curl -LO https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz
curl -LO https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz
md5sum images.tar.gz annotations.tar.gz   # must match the table above
tar xzf images.tar.gz && tar xzf annotations.tar.gz
cd ../..
dvc repro            # corruptions, manifest, validation and training (GPU)
```

`config.py` is not a dependency of the data stages; after changing a setting that
affects the corruption suite, run `dvc repro -f generate_corruptions`.
