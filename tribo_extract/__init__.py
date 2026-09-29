"""tribo_extract - turn raw tribometer runs into ML targets (COF, specific wear rate).

Modules
-------
tribometer_csv  parse the tribometer CSV export
cof             steady-state COF, running-in time, 1 Hz COF(t) series
wli             rebuild a height map from the WLI screenshot (or load a raw export)
wear            locate the wear track, integrate its cross-section
physics         sliding distance, Hertz contact, k = V/(F*S), wear-trajectory fit
pipeline        batch over data/raw/<Experiment_ID>/ -> data/processed/*.csv
"""
__version__ = "0.1.0"
