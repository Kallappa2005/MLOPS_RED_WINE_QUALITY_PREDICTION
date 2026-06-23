# 🍷 Wine Quality Prediction MLOps

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-wine--quality--predictor-brightgreen)](https://wine-quality-predictor-01.onrender.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](https://github.com/Kallappa2005/MLOPS_RED_WINE_QUALITY_PREDICTION/actions)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://hub.docker.com/)

A complete **MLOps pipeline** for predicting red wine quality using machine learning. This project demonstrates end-to-end ML engineering with automated CI/CD, containerization, and cloud deployment.

This repository now also includes **DVC (Data Version Control)** setup so the data pipeline, model artifacts, and metrics can be versioned and reproduced consistently.

---

## Table of Contents

- [🎯 Live Demo](#-live-demo)
- [✨ Features](#-features)
- [🏗️ Project Architecture](#%EF%B8%8F-project-architecture)
- [🛠️ Tech Stack](#%EF%B8%8F-tech-stack)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [🎮 Usage Examples](#-usage-examples)
- [#-usage-examples](#-development)
- [📈 Performance Metrics](#-performance-metrics)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 Live Demo
**Try the app:** [https://wine-quality-predictor-01.onrender.com/](https://wine-quality-predictor-01.onrender.com/)

Enter wine characteristics and get instant quality predictions!

## ✨ Features

- 🤖 **ML Model**: ElasticNet regression for wine quality prediction
- 🐳 **Docker**: Fully containerized application 
- ☁️ **Cloud Deployed**: Live on Render with auto-scaling
- 🔄 **CI/CD Pipeline**: Automated deployment on code changes
- 📊 **ML Pipeline**: Data ingestion → Validation → Transformation → Training → Evaluation
- 🎨 **Beautiful UI**: Clean, responsive web interface
- 🚀 **Auto-Training**: Model trains automatically on deployment

## 🏗️ Project Architecture

```
📦 MLOps Pipeline
├── 📊 Data Ingestion (Wine dataset from GitHub)
├── ✅ Data Validation (Schema validation)
├── 🔄 Data Transformation (Train/test split)
├── 🤖 Model Training (ElasticNet with hyperparameters)
├── 📈 Model Evaluation (RMSE, MAE, R² metrics)
└── 🚀 Deployment (Docker + Render + CI/CD)
```

## 🛠️ Tech Stack

- **ML Framework**: scikit-learn, pandas, numpy
- **Web Framework**: Flask
- **Containerization**: Docker
- **Cloud Platform**: Render
- **CI/CD**: GitHub Actions
- **Configuration**: YAML-based config management
- **Logging**: Python logging with structured approach
- **Data Versioning**: DVC pipeline and artifact tracking

## 🚀 Quick Start

### Option 1: Use Live App (Recommended)
Just visit [https://wine-quality-predictor-01.onrender.com/](https://wine-quality-predictor-01.onrender.com/) and start predicting!

### Option 2: Local Development

**Clone the repository**
```bash
git clone https://github.com/Kallappa2005/MLOPS_RED_WINE_QUALITY_PREDICTION
cd MLOPS_RED_WINE_QUALITY_PREDICTION
```

**Create conda environment**
```bash
conda create -n wineqp python=3.8 -y
conda activate wineqp
```

**Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .
```

For notebook/type-checking tools used during development, install:

```bash
pip install -r requirements-dev.txt
```

**Initialize DVC once per clone**
```bash
dvc init
```

If DVC is already initialized in the repository, skip this step.

**Run the application**
```bash
python app.py
```

Visit `http://localhost:8080` in your browser.

## 🐳 Docker Usage

**Build and run with Docker:**
```bash
docker build -t wine-quality-app .
docker run -p 8080:8080 wine-quality-app
```

## 📊 Model Details

- **Algorithm**: ElasticNet Regression
- **Features**: 11 wine characteristics (acidity, sugar, alcohol, etc.)
- **Target**: Wine quality score (0-10)
- **Hyperparameters**: 
  - Alpha: 0.2
  - L1 Ratio: 0.1

## 🔄 MLOps Pipeline

The project follows a complete MLOps workflow:

1. **Data Pipeline**: Automated data ingestion and validation
2. **Model Pipeline**: Training, evaluation, and versioning  
3. **Deployment Pipeline**: Containerization and cloud deployment
4. **CI/CD Pipeline**: Automated testing and deployment
5. **Monitoring**: Logging and error handling

### DVC Pipeline

The project uses DVC to version and reproduce the ML workflow stages:

1. Data ingestion
2. Data validation
3. Data transformation
4. Model training
5. Model evaluation

After installing DVC, you can run:

```bash
dvc repro
```

Useful commands:

```bash
dvc dag
dvc metrics show
dvc status
```

The tracked outputs are stored under `artifacts/`, while the pipeline definition lives in `dvc.yaml`.

### dvc pipeline stages

| stage name | description | target source / inputs | generated artifacts |
| :--- | :--- | :--- | :--- |
| `data_ingestion` | downloads and extracts the raw wine dataset | `config/config.yaml` | `artifacts/data_ingestion/` |
| `data_validation` | verifies data types against project schema | `schema.yaml` | `artifacts/data_validation/` |
| `data_transformation` | handles train/test splitting and processing | `artifacts/data_ingestion/` | `artifacts/data_transformation/` |
| `model_trainer` | trains the elasticnet regression model | `params.yaml` | `artifacts/model_trainer/model.joblib` |
| `model_evaluation` | calculates rmse, mae, and r2 performance metrics | `artifacts/model_trainer/` | `artifacts/model_evaluation/metrics.json` |

## 🚀 Deployment & CI/CD

- **Automatic Deployment**: Push to `main` branch triggers deployment
- **Container Registry**: Docker images built automatically
- **Zero Downtime**: Rolling deployments on Render
- **Auto-Training**: Model retrains automatically on new deployments

## 📁 Project Structure

```
├── src/mlProject/           # Core ML package
│   ├── components/          # ML pipeline components
│   ├── config/             # Configuration management
│   ├── entity/             # Data classes and entities
│   ├── pipeline/           # Training and prediction pipelines
│   └── utils/              # Utility functions
├── config/                 # Configuration files
├── artifacts/              # Generated models and data
├── templates/              # Web UI templates
├── static/                 # CSS, JS, images
├── .github/workflows/      # CI/CD pipeline
├── dvc.yaml                # DVC pipeline definition
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
└── app.py                 # Flask web application
```

## 🎮 Usage Examples

### Web Interface
1. Visit the live app
2. Fill in wine characteristics:
   - Fixed Acidity: `7.4`
   - Volatile Acidity: `0.7`
   - Citric Acid: `0.0`
   - ... (11 features total)
3. Click "Predict" to get quality score

### API Usage
```python
# For local development
import requests

data = {
    'fixed_acidity': 7.4,
    'volatile_acidity': 0.7,
    # ... other features
}

response = requests.post('http://localhost:8080/predict', data=data)
print(response.text)
```

### production flask endpoints

| route path | http method | function | authentication / security |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | renders the web portal frontend dashboard ui | none (public access) |
| `/train` | `GET` / `POST` | triggers manual pipeline retraining sequence | gated via secure token / atomic lock |
| `/predict` | `POST` | receives wine vector attributes to compute quality matrix | none (public inference api) |

## 🔧 Development

### Adding New Features
1. Update configuration files (`config.yaml`, `params.yaml`, `schema.yaml`)
2. Modify components in `src/mlProject/components/`
3. Update pipelines in `src/mlProject/pipeline/`
4. Test locally with `python app.py`
5. Commit and push - auto-deployment handles the rest!

### Manual Model Training
Visit `/train` endpoint to manually retrain the model:
```
https://wine-quality-predictor-01.onrender.com/train
```

## 📈 Performance Metrics

The model is evaluated using:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)  
- **R²** (Coefficient of Determination)

Metrics are automatically saved in `artifacts/model_evaluation/metrics.json`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contributors

<!-- CONTRIBUTORS_START -->
<a href="https://github.com/AMAN194701"><img src="https://github.com/AMAN194701.png" width="50px" loading="lazy" title="AMAN194701" style="border-radius:50%;margin:5px;" alt="AMAN194701" /></a><a href="https://github.com/Juhi4433"><img src="https://github.com/Juhi4433.png" width="50px" loading="lazy" title="Juhi4433" style="border-radius:50%;margin:5px;" alt="Juhi4433" /></a><a href="https://github.com/LaskaaaD"><img src="https://github.com/LaskaaaD.png" width="50px" loading="lazy" title="LaskaaaD" style="border-radius:50%;margin:5px;" alt="LaskaaaD" /></a><a href="https://github.com/Prakhar54-byte"><img src="https://github.com/Prakhar54-byte.png" width="50px" loading="lazy" title="Prakhar54-byte" style="border-radius:50%;margin:5px;" alt="Prakhar54-byte" /></a><a href="https://github.com/Prateek2007-cmd"><img src="https://github.com/Prateek2007-cmd.png" width="50px" loading="lazy" title="Prateek2007-cmd" style="border-radius:50%;margin:5px;" alt="Prateek2007-cmd" /></a><a href="https://github.com/ScarsAndSource"><img src="https://github.com/ScarsAndSource.png" width="50px" loading="lazy" title="ScarsAndSource" style="border-radius:50%;margin:5px;" alt="ScarsAndSource" /></a><a href="https://github.com/Siddh2024"><img src="https://github.com/Siddh2024.png" width="50px" loading="lazy" title="Siddh2024" style="border-radius:50%;margin:5px;" alt="Siddh2024" /></a><a href="https://github.com/Yogender-verma"><img src="https://github.com/Yogender-verma.png" width="50px" loading="lazy" title="Yogender-verma" style="border-radius:50%;margin:5px;" alt="Yogender-verma" /></a><a href="https://github.com/aaniya22"><img src="https://github.com/aaniya22.png" width="50px" loading="lazy" title="aaniya22" style="border-radius:50%;margin:5px;" alt="aaniya22" /></a><a href="https://github.com/anshika1179"><img src="https://github.com/anshika1179.png" width="50px" loading="lazy" title="anshika1179" style="border-radius:50%;margin:5px;" alt="anshika1179" /></a><a href="https://github.com/ionfwsrijan"><img src="https://github.com/ionfwsrijan.png" width="50px" loading="lazy" title="ionfwsrijan" style="border-radius:50%;margin:5px;" alt="ionfwsrijan" /></a><a href="https://github.com/itsdakshjain"><img src="https://github.com/itsdakshjain.png" width="50px" loading="lazy" title="itsdakshjain" style="border-radius:50%;margin:5px;" alt="itsdakshjain" /></a><a href="https://github.com/nyxsky404"><img src="https://github.com/nyxsky404.png" width="50px" loading="lazy" title="nyxsky404" style="border-radius:50%;margin:5px;" alt="nyxsky404" /></a><a href="https://github.com/rahul616sama"><img src="https://github.com/rahul616sama.png" width="50px" loading="lazy" title="rahul616sama" style="border-radius:50%;margin:5px;" alt="rahul616sama" /></a><a href="https://github.com/saidai-bhuvanesh"><img src="https://github.com/saidai-bhuvanesh.png" width="50px" loading="lazy" title="saidai-bhuvanesh" style="border-radius:50%;margin:5px;" alt="saidai-bhuvanesh" /></a><a href="https://github.com/saurabhhhcodes"><img src="https://github.com/saurabhhhcodes.png" width="50px" loading="lazy" title="saurabhhhcodes" style="border-radius:50%;margin:5px;" alt="saurabhhhcodes" /></a>
<!-- CONTRIBUTORS_END -->


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: Wine Quality Dataset
- Inspired by MLOps best practices
- Built with modern DevOps principles

---

**⭐ Star this repo if you found it helpful!**
