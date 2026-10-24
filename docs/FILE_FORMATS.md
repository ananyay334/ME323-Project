# Per-run file formats (confirmed on the October 2026 campaign data)

The machine-readable version is `config/file_layout.yaml`. A run folder (`testdata/R04/`)
holds one tribometer run and the white-light (WLI) scans of its track:

| file | role | size | format | used for |
|---|---|---|---|---|
| `R04.csv` | `cof_csv` | 0.5 MB | RTEC export (below) | COF, cycles, Fz |
| `R04.lrx` | `recipe` | 0.5 kB | XML recipe summary | **load, frequency, run time** |
| `R04.bcrf`, `R04a.bcrf`, `R04_b.bcrf` … | `height` | 2.3 MB | BCR-F height map (below) | wear |
| `R04.info`, `R04a.info` … | `metadata` | 0.6 kB | XML `CRunDetail` | lens, pixel size, stage XY |
| `R04.png` … | `preview` | 15 kB | preview image | not needed |
| `R04.Raw` … | `ignore` | **≈1.2 GB** | raw interferogram frame stack | **never opened** |
| `R04.rtec`, `R04.enc` | `ignore` | 14 MB / 0.2 MB | RTEC surface file, encoder log | not needed |

The `.Raw` stacks are ~96 % of the data volume. The height map is already reconstructed in
the `.bcrf`, so only `.csv .lrx .bcrf .info` (~12 MB per run) need to be copied.

**Run ID and location.** The run ID is the leading `R\d\d` (or `T\d\d` for top-ups). The scan
location is the letter right after it, with or without an underscore: `R04` → `m`,
`R04a`/`R04_a` → `a`, `R01a_13` → `a`, `R01_13` → `m` (a trailing number is a scan counter).
All scans in a folder are post-run scans (checkpoint 2); there are no baseline or mid-run scans.

**Recipe `.lrx`.** `<StepDetail>` blocks such as
`2> Fz:32 N,  t:1.4 min  R - Continuous  Frequency (Hz): 7.5`. Step 1 is a 2 s static
loading step; step 2 is the single continuous sliding step. Load, frequency and
run time (sum of the sliding steps) are read from here. Without an `.lrx`, they are measured
from the CSV (mean Fz rounded to 0.5 N, reversal frequency, sliding duration).

**COF CSV.** Line 1 = metadata names, a bare CR, metadata values (`Averaging 10` → logged at
**100 Hz**; `TestDate`). Then the channel header: `Step, Timestamp, RecipeStep, DAQ.Fz (N),
DAQ.COF (Fx), DAQ.Fx (N), XYZ.Z Depth (mm)`, plus `XYZ.X Velocity (mm/s)` and
`Reciprocating.Frequency (Hz)` on the 13 Oct runs (R01–R10) only. `XYZ.Z Depth` is a servo
diagnostic and is never used as wear.

**BCR-F height map.** Unicode BCR-F: a UTF-16 header of 2048 characters (4096 bytes) with
`xpixels = 960`, `ypixels = 600`, `xlength`/`ylength` in nm (1.875 µm pixels, 1.8 × 1.125 mm
field of view with the I-10x lens), `zunit = um`, `voidpixels = 3.402823E+38`; then float32
little-endian heights in µm, row by row. `read_bcrf` also reads the ASCII/int16 BCR variant.
The track runs along x (the plate's sliding direction).

**`.info`.** XML: `PixelSizeuM`, `Lens`, `Mode` (WLI), `XPos`/`YPos` (stage mm).
