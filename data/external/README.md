# External data

`ponnuswamy2024_table5.csv`: Table 5 of P. Ponnuswamy and M. Manoharan, "Prediction of wear volume and
friction coefficients of SS304 alloy using grey Taguchi-based response surface methodology",
*Mater. Res. Express* **11** (2024) 066515, https://doi.org/10.1088/2053-1591/ad45bb (CC BY 4.0).
Values transcribed as published; units as stated in the paper.

- **Test:** SS304 pin (6 × 6 mm, semi-cylindrical end, R = 3 mm) on an SS304 disk, unidirectional pin-on-disk, dry.
- **Conditions:** 32 °C, 84 % RH.
- **Factors:** 10/15/20 N, 1/2/3 m/s, track diameter 60/120 mm.
- **Duration:** 200 s per section 4 of the paper (the abstract says "up to 2000 s").
- **Wear_volume_mm3:** the "wear volume stabilisation value".
- **COF:** the "coefficient of friction stabilisation value".

Plausibility notes:
- The published COF (0.002–0.007) is about 30× below our measurements and far below typical dry steel-on-steel values.
- The published wear volumes (290–1620 mm³ in 200 s) are comparable to the whole pin volume (≈ 1000 mm³).
- The units may therefore differ from what is stated. `ml/literature_model.py` only uses these numbers in ways that do not
  depend on a constant unit factor (log space with a study offset), except in the naive-pooling check.
