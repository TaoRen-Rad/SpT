# Spectrum Transformer (SpT)

## Overview
The Spectrum Transformer (SpT) is a novel Transformer-based neural network model designed for fast and accurate retrievals of column-averaged CO\(_2\) dry air mole fraction (XCO\(_2\)) directly from satellite-measured spectra. The model significantly reduces computational time and offers robust performance, addressing challenges such as data drift due to increasing atmospheric CO\(_2\) levels without requiring synthetic future data.

## Key Features
- **Fast and Accurate Retrievals**: Reduces computational time from minutes to milliseconds per retrieval.
- **Unbiased Generalization**: In our numerical epxeriment, the model was trained on historical OCO-2 spectra (2017-2019) and validated on data from 2020-2022, achievd RMSE as low as 1.2 ppm, with close zero 0.0 ppm ME.
- **Efficient Fine-Tuning**: Maintains accuracy with periodic fine-tuning using less than 10% of newly available data.
- **Validated Performance**: Validated against TCCON ground-based measurements, capturing seasonal and regional variations in XCO\(_2\).

## Data Preparation
The training dataset is too large to be shared online. However, you can prepare the required data as follows:
1. Visit NASA's [Earth Data website](https://disc.gsfc.nasa.gov/) to download OCO-2 Level 1 and Level 2 product data.
2. Use the provided `organize_data.py` script to process the data into a Parquet-formatted Pandas DataFrame suitable for training.

## Structure
- **`SpT` Package**: Contains all modules and functions related to the Spectrum Transformer model.
- **Training Script**: The script `train_the_model.py` handles the training process, including data loading, model initialization, and evaluation.
- **Data Organization Script**: The script `organize_data.py` processes the raw OCO-2 data into a Parquet-formatted Pandas DataFrame.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SpT.git
   cd SpT
   ```
2.	Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3.	Prepare the data as described above.
4.	Train the model:
   ```python
   python train_the_model.py
   ```

## Cite as

If you use SpT in your research, please cite as follows,

```bibtex
@article{chenTransformerBasedFastMole2025,
  title = {Transformer-Based Fast Mole Fraction of {CO}$_2$ Retrievals from Satellite-Measured Spectra},
  author = {W. Chen and T. Ren and C. Zhao and Y. Wen and Y. Gu and M. Zhou and P. Wang},
  year = {2025},
  month = mar,
  journal = {Journal of Remote Sensing},
  volume = {5},
  pages = {0470},
  doi = {10.34133/remotesensing.0470}
}

@article{chenDeterministicProbabilisticLightweight2025,
  title = {From Deterministic to Probabilistic: A Lightweight Framework for {Probabilistic Machine Learning in Trace Gas Remote Sensing},
  author = {W. Chen and T. Ren and C. Zhao},
  year = 2025,
  month = dec,
  journal = {Journal of Remote Sensing},
  volume = {5},
  pages = {0881},
  doi = {10.34133/remotesensing.0881}
}
```
## License

This project is licensed under the MIT License. See the LICENSE file for details.
