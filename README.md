# 🏏 Cricket Form Analysis with Hidden Markov Models

This project implements a complete machine learning pipeline to analyze cricket player form using Hidden Markov Models (HMM). It extracts player data from ball-by-ball JSON data, performs feature engineering, trains an HMM to identify player form states, validates the model, and predicts performance.

---

## Features

- Extraction of player innings data from Cricsheet formatted JSON zipped match files (T20, ODI, Test)
- Calculation of batting features including runs, balls faced, boundaries, dot balls, strike rates, and rolling averages
- Hidden Markov Model training to identify latent player form states like Poor, Average, and Excellent
- Model validation with metrics including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score
- Predictions on player runs with confidence intervals
- Visualizations for data distribution, form progression, and prediction errors

---

## Installation

Run the following to install required packages:
pip install hmmlearn scikit-learn pandas numpy matplotlib seaborn requests plotly



---

## Usage

1. Download Cricsheet JSON zipped archives for matches [here](https://cricsheet.org/downloads/)
2. Use the function `extract_player_data_from_json` to extract player innings by specifying player name and format
3. Perform feature engineering on the extracted data using `create_features`
4. Train the Hidden Markov Model with `train_hmm_model`
5. Validate and visualize results using summary statistics and plots

---

## Code Structure

- **Data Extraction:** Parsing match files, filtering by player and format
- **Feature Engineering:** Creating meaningful numeric features for ML
- **Model Training & Validation:** Gaussian HMM with train/test split
- **Visualization:** Use matplotlib and seaborn for performance graphs

---

## Technologies Used

- Python 3.x
- [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/) for Hidden Markov Models
- [scikit-learn](https://scikit-learn.org/stable/) for scaling and metrics
- [pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/) for data manipulation
-  [matplotlib](https://matplotlib.org/)  [seaborn](https://seaborn.pydata.org/) and  [plotly](https://plotly.com/) for plotting                              
- [requests](https://docs.python-requests.org/en/latest/) for HTTP requests if needed

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Data sourced from [Cricsheet](https://cricsheet.org)
- Support and inspiration from the open source data science and cricket analytics community

---

Feel free to contribute or raise issues for improvements!


