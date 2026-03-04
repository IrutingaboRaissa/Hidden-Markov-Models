# Human Activity Recognition Using Hidden Markov Models

## Project Overview

This project classifies human activities from smartphone sensor data using Hidden Markov Models (HMMs). We recorded accelerometer and gyroscope readings from two phones and trained a generative per-class HMM to distinguish four activities: Walking, Jumping, Standing, and Still.

## Activities

| Activity | Description | Recordings (Train) | Recordings (Test) |
|----------|-------------|--------------------|--------------------|
| Walking  | Normal pace walking | 12 | 3 |
| Jumping  | Jumping in place | 12 | 3 |
| Standing | Standing still, phone in hand | 13 | 3 |
| Still    | Phone on a flat surface, no movement | 13 | 3 |

Total: 50 training + 12 test paired recordings (each pair = accelerometer CSV + gyroscope CSV).

## Data Collection

- **App:** Sensor Logger (available on iOS and Android)
- **Sensors:** Accelerometer (x, y, z) and Gyroscope (x, y, z)
- **Phone 1:** ~100 Hz sampling rate (Jumping, Walking)
- **Phone 2:** ~50 Hz sampling rate (Standing, Still)
- **Format:** CSV files with columns: time, seconds_elapsed, z, y, x
- **Minimum duration:** Each activity has well over 1 minute 30 seconds of data

## How It Works

1. **Data Loading:** Extract zip files, load CSV pairs, merge accelerometer and gyroscope on timestamp using merge_asof
2. **Feature Extraction:** Sliding 2-second windows with 50% overlap. 69 features per window including:
   - Time-domain: mean, variance, std, RMS, zero-crossing rate, peak-to-peak range, IQR, signal magnitude area, inter-axis correlations
   - Frequency-domain (via FFT): dominant frequency, spectral energy, spectral entropy
3. **Normalization:** Z-score standardization (fit on train, apply to both train and test)
4. **Training:** One GaussianHMM per activity with 2 hidden sub-states, trained using Baum-Welch (EM algorithm) with convergence threshold of 1e-4
5. **Classification:** Score each test window against all 4 models, pick the one with the highest normalized log-likelihood
6. **Decoding:** Viterbi algorithm recovers the most likely hidden state sequence

## Project Structure

```
Hidden-Markov-Models/
    hmm_activity_recognition.ipynb   -- main notebook with all code and analysis
    README.md                        -- this file
    datacollect/
        train/                       -- 50 zip files with training data
        test/                        -- 12 zip files with unseen test data
    sensor_data/                     -- extracted training CSVs (created by notebook)
        jumping/
        walking/
        standing/
        still/
    sensor_data_test/                -- extracted test CSVs (created by notebook)
```

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- hmmlearn

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn hmmlearn
```

## How to Run

1. Make sure the `datacollect/train/` and `datacollect/test/` folders contain the zip files
2. Open `hmm_activity_recognition.ipynb` in Jupyter or VS Code
3. Run all cells from top to bottom
4. The notebook will extract zip files, train the models, and produce all visualizations and metrics

## Outputs

The notebook generates:
- Raw sensor data plots for each activity
- Transition probability matrix heatmaps
- Emission probability visualizations (Gaussian means)
- Viterbi decoded state sequence plots
- Confusion matrix on unseen test data
- Per-class sensitivity, specificity, and accuracy table
- Metrics bar chart

## Key Concepts

- **Hidden States (Z):** The actual activities (Walking, Jumping, Standing, Still)
- **Observations (X):** Feature vectors extracted from sensor windows
- **Transition Matrix (A):** Probability of switching between activities
- **Emission Probabilities (B):** Gaussian distributions over features for each state
- **Initial Probabilities (pi):** Starting state distribution
- **Baum-Welch:** EM algorithm that learns HMM parameters from training data
- **Viterbi:** Dynamic programming algorithm that finds the most likely state sequence
