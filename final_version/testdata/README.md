# Raw data (23 runs)

One folder per run (R01 … R29; R00, R03, R15, R16, R18, R23 and R27 were planned but never run).
Only the files the pipeline reads are included:

| file | content |
|---|---|
| `Rxx.csv` | RTEC friction log, 100 Hz (Fz, Fx, instrument COF, step, timestamp) |
| `Rxx.lrx` | test recipe (XML): load, step times, frequency |
| `Rxx*.bcrf` | white-light-interferometer height maps, one per scan location (m, a, b, c) |
| `Rxx*.info` | scan metadata (XML): pixel size, lens, stage position |

The vendor containers (`.rtec`, `.Raw`, about 1.3 GB here and ~150 GB with the raw frames) and `.png` previews are
never opened by the pipeline and are not shipped. Copies are byte-identical to the originals.
