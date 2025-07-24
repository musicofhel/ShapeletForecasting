# Financial Wavelet Forecasting Dashboard

A comprehensive financial analysis and forecasting platform that combines wavelet analysis, pattern recognition, and machine learning to predict market movements.

## 🚀 Quick Start

### Single Command Startup
```bash
# Development mode (with hot reload)
python app.py

# Demo mode (with sample data)
python app.py --mode demo

# Production mode (with Docker)
./start.sh prod
```

### Alternative Startup Methods
```bash
# Using the unified startup script
./start.sh dev      # Development mode
./start.sh demo     # Demo mode with sample data
./start.sh prod     # Production mode with Docker
./start.sh status   # Check service status
./start.sh stop     # Stop all services
```

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB disk space

### Dependencies Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# For Docker deployment
docker-compose --version
```

## 🎯 Features

### Core Functionality
- **Wavelet Analysis**: Advanced time-frequency analysis using PyWavelets
- **Pattern Recognition**: DTW-based pattern matching for financial sequences
- **Machine Learning**: XGBoost, LightGBM, and PyTorch models for forecasting
- **Real-time Monitoring**: Live pattern detection and alerts
- **Interactive Dashboard**: Rich visualizations with Plotly Dash

### Dashboard Components
- **Pattern Gallery**: Browse and analyze detected patterns
- **Time Series Visualization**: Interactive price charts with overlays
- **Scalogram Views**: Wavelet scalograms for frequency analysis
- **Pattern Comparison**: Side-by-side pattern analysis
- **Real-time Monitoring**: Live market data and pattern detection
- **Export Functionality**: Generate reports and export data

### API Endpoints
- `/api/predict` - Get predictions for given symbols
- `/api/patterns` - Retrieve detected patterns
- `/api/analyze` - Perform wavelet analysis
- `/api/health` - Health check endpoint

## 🏗️ Architecture

### Project Structure
```
financial_wavelet_prediction/
├── app.py                    # Single entry point
├── start.sh                  # Unified startup script
├── src/
│   ├── dashboard/            # Dash application
│   ├── wavelet_analysis/     # Wavelet processing
│   ├── models/               # ML models
│   ├── features/             # Feature engineering
│   ├── dtw/                  # DTW algorithms
│   └── api/                  # REST API
├── data/
│   ├── demo/                 # Sample datasets
│   ├── processed/            # Processed data
│   └── raw/                  # Raw market data
├── notebooks/                # Jupyter tutorials
├── tests/                    # Test suite
└── docs/                     # Documentation
```

### Technology Stack
- **Frontend**: Dash, Plotly, Bootstrap
- **Backend**: Python, FastAPI, NumPy, Pandas
- **ML/AI**: Scikit-learn, XGBoost, PyTorch, PyWavelets
- **Data**: yFinance, Pandas-TA
- **Deployment**: Docker, Docker Compose

## 🔧 Configuration

### Environment Variables
Create `.env.dashboard` file:
```bash
# Copy from example
cp .env.dashboard.example .env.dashboard

# Edit configuration
nano .env.dashboard
```

### Key Configuration Options
- `DASH_DEBUG`: Enable debug mode (true/false)
- `MODEL_DIR`: Directory for saved models
- `DATA_DIR`: Directory for data files
- `CACHE_DIR`: Directory for cache files
- `REDIS_URL`: Redis connection string (optional)

## 📊 Usage Examples

### Basic Usage
```python
# Start the dashboard
python app.py --mode demo --port 8050

# Access the dashboard
# Open browser to http://localhost:8050
```

### Advanced Usage
```python
# Production deployment
python app.py --mode prod --host 0.0.0.0 --port 80

# Custom configuration
python app.py --mode dev --debug --port 8080
```

### Docker Deployment
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Testing

### Run All Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run test suite
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Categories
- **Unit Tests**: `tests/unit/`
- **Integration Tests**: `tests/integration/`
- **Performance Tests**: `tests/performance/`

## 📈 Performance Optimization

### Caching
- **Redis**: Optional Redis caching for improved performance
- **LRU Cache**: Built-in caching for expensive computations
- **File Cache**: Persistent caching for model predictions

### Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Health Checks**: Automated service monitoring

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

#### Port Already in Use
```bash
# Kill process on port 8050
lsof -ti:8050 | xargs kill -9

# Or use different port
python app.py --port 8080
```

#### Memory Issues
```bash
# Reduce data size for demo mode
python app.py --mode demo

# Use production mode with Docker
./start.sh prod
```

### Debug Mode
Enable debug mode for detailed logging:
```bash
python app.py --debug
```

## 📚 Documentation

### API Documentation
- [API Endpoints](API_ENDPOINTS_DOCUMENTATION.md)
- [Pattern Analysis Guide](docs/advanced_features_guide.md)
- [Deployment Guide](docs/deployment_guide.md)

### Jupyter Notebooks
Explore the notebooks in the `notebooks/` directory:
- `02_wavelet_analysis_demo.ipynb` - Wavelet analysis tutorial
- `03_dtw_analysis_demo.ipynb` - DTW pattern matching
- `04_feature_engineering_demo.ipynb` - Feature engineering
- `05_model_comparison_demo.ipynb` - Model comparison
- `06_backtesting_evaluation_demo.ipynb` - Backtesting
- `07_final_project_presentation.ipynb` - Complete walkthrough

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd financial_wavelet_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start development server
python app.py --mode dev
```

### Code Style
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyWavelets**: Wavelet analysis library
- **Plotly**: Interactive visualizations
- **Dash**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **yFinance**: Financial data provider

## 📞 Support

### Getting Help
- **Issues**: Create GitHub issue for bugs
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check docs/ directory

### Contact
- **Email**: support@financialwavelet.com
- **Discord**: [Join our community](https://discord.gg/financialwavelet)

---

**Ready to start?** Run `python app.py` and open http://localhost:8050 in your browser!
