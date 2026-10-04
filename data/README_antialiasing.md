# Anti-aliased 1,400-sample files

    experimental/csv_1400/   al_small_antialiased_1400.csv    al_large_antialiased_1400.csv
                             al_heldout_antialiased_1400.csv  steel_antialiased_1400.csv
                             al_both_antialiased_1400.csv

Each file is the matching 14,000-sample file in `experimental/csv_14000/` reduced by a factor of 10.
Same rows in the same order, same label and material columns; signal columns `s0000`-`s1399`.

## Method

`scipy.signal.resample_poly(x, up=1, down=10)` on every signal:

1. **Low-pass filter.** A linear-phase FIR filter (Kaiser window) removes everything above the new
   Nyquist frequency (0.05 cycles per original sample) before any sample is discarded.
2. **Zero-phase.** The filter delay is compensated, so arrival times are not shifted.
3. **Decimation.** Every 10th filtered sample is kept.

## Checks

- Only 0.12% of the signal energy lies above the new Nyquist frequency, all of it noise; the pulse
  peaks at ~0.0068 cycles per original sample (about 15 samples per cycle after decimation).
- The result agrees with an ideal band-limited (FFT) resampling to within 0.5% of full scale
  (correlation 0.999999), including the first and last samples.
- Values are stored to 6 significant digits.
