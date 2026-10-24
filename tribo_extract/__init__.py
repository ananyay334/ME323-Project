"""tribo_extract - turn raw tribometer runs into ML targets (per-cycle COF, specific wear rate).

Modules
-------
tribometer_csv  parse the tribometer CSV export
cof             per-cycle COF per sliding step (extract_cof_cycles); old 1 Hz COF(t) (extract_cof)
layout          file roles, scan location, checkpoint ladder; organize a flat dump
height_io       read RAW / text height scans (format ladder + validation)
wli             rebuild a height map from the WLI screenshot (or load a text export)
wear            locate the wear track, integrate groove and pile-up cross-sections
physics         sliding distance, Hertz contact, k = V/(F*S)
pipeline        batch over data/raw/<Experiment_ID>/ -> data/processed/*.csv (three tables)
"""
__version__ = "0.2.0"
