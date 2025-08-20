# Changelog

All notable changes to the PatchCore Educational Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-20

### Added
- 🔬 **Interactive PatchCore Dashboard** with real-time parameter tuning
- 📓 **Complete Jupyter Tutorial** with step-by-step anomalib integration
- 🎯 **Educational Features**:
  - Clickable test image selection
  - Fixed colormap scaling (0.0 - 8.0) for consistent comparisons
  - Real-time metrics display with detailed statistics
  - Professional dark theme optimized for analysis
- 🖼️ **Image Generation System**:
  - Parametric smiley generation with 6 variation parameters
  - 20 "good" training images with controlled variations
  - 10 mixed test images (5 anomalous + 5 good)
  - Thumbnail optimization (25px training, 60px test)
- 📊 **Advanced Visualizations**:
  - Interactive heatmaps with detailed hover information
  - Score distribution histograms
  - Patch analysis with overlap percentage calculation
  - Responsive slider controls for patch size and stride

### Technical Features
- ✅ **Fixed Colormap Scaling**: Prevents dynamic scaling for meaningful comparisons
- ✅ **Optimized Patch Analysis**: Efficient computation with real-time updates
- ✅ **Professional UI**: Dark theme with color-coded metrics
- ✅ **Cross-platform Compatibility**: Works on macOS, Linux, and Windows
- ✅ **Comprehensive Documentation**: README, requirements, and setup scripts

### Dependencies
- `dash>=3.2.0` - Web dashboard framework
- `plotly>=5.17.0` - Interactive visualizations
- `numpy>=1.24.0` - Numerical computations
- `Pillow>=10.0.0` - Image processing

### Educational Content
- **PatchCore Theory**: Complete explanation of algorithm mechanics
- **Practical Implementation**: Step-by-step coding examples
- **Parameter Understanding**: Interactive exploration of hyperparameters
- **Visual Learning**: Intuitive representations of complex concepts

## [Unreleased]

### Planned Features
- 🔄 **Multi-Algorithm Support**: VAE, AutoEncoder comparisons
- 📈 **Advanced Metrics**: ROC curves, precision-recall analysis  
- 🗂️ **Custom Dataset Support**: Upload and analyze personal images
- 🔧 **Export Functionality**: Save results and configurations
- 📚 **Extended Tutorials**: Advanced anomaly detection techniques

---

## Version History

- **v1.0.0** (2025-08-20): Initial release with complete PatchCore educational platform
- **v0.x.x** (Development): Internal development versions
