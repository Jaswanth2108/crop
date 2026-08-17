# Dr. Crop - Web-based ML and DL Application

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Dr.%20Crop-2ea44f?style=for-the-badge&logo=render&logoColor=white)](https://dr-crop.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Framework-Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)

Dr. Crop is a web-based Machine Learning and Deep Learning application designed to assist farmers in making informed decisions regarding crop management. This application consists of three main modules: Crop Recommendation, Fertilizer Recommendation, and Image Disease Prediction.

---

## 🚀 Live Demo

Experience Dr. Crop live in your browser:

🌐 **[Launch Dr. Crop Live App](https://dr-crop.onrender.com)**

> *Note: If hosted on a free cloud platform (e.g., Render / Hugging Face Spaces / Railway), the initial spin-up may take 15–30 seconds.*

---

## Table of Contents

- [Live Demo](#-live-demo)
- [Datasets](#datasets)
- [Modules](#modules)
  - [1. Crop Recommendation](#1-crop-recommendation)
  - [2. Fertilizer Recommendation](#2-fertilizer-recommendation)
  - [3. Image Disease Prediction](#3-image-disease-prediction)
- [Features](#features)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Datasets

- [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- [Fertilizer Recommendation Dataset](https://www.kaggle.com/datasets/amankumar1007/fertilizer-recommendation)
- [Image Disease Prediction Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Modules

### 1. Crop Recommendation

The Crop Recommendation module leverages machine learning algorithms to provide farmers with personalized crop recommendations based on various factors such as soil type, climate, and historical crop performance data. It helps farmers make optimal choices when selecting the crops to cultivate, taking into account their specific conditions.

### 2. Fertilizer Recommendation

The Fertilizer Recommendation module uses data-driven approaches to suggest the appropriate types and quantities of fertilizers for a given crop and soil condition. By analyzing soil nutrient levels and crop nutrient requirements, it helps farmers maximize crop yields while minimizing fertilizer costs and environmental impact.

### 3. Image Disease Prediction

The Image Disease Prediction module employs deep learning techniques to diagnose crop diseases from images captured by farmers. It can identify common crop diseases and provide timely recommendations for disease management. This enables farmers to take proactive measures to protect their crops from diseases, ultimately increasing crop productivity.

## Features

- User-friendly web interface for easy access and interaction.
- Data-driven recommendations for crop selection and fertilizer application.
- Deep learning-powered image analysis for disease detection.
- Personalized advice based on location-specific data.
- Historical data tracking and reporting for improved decision-making.

## Getting Started

To run Dr. Crop on your local machine, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Jaswanth2108/Crop.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Crop
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the web application:

   ```bash
   python app.py
   ```

5. Open a web browser and access Dr. Crop at [http://localhost:8000](http://localhost:8000).

## Dependencies

Dr. Crop relies on several Python libraries and frameworks, including:

- Flask
- PyTorch
- Scikit-learn
- and more (see `requirements.txt` for a complete list).

## Contributing

We welcome contributions from the community. If you'd like to contribute to the development of Dr. Crop, please fork the repository and submit pull requests with your changes. Be sure to follow our [code of conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Happy farming with Dr. Crop! 🌾🌱🚜
