# 🏏Cricket Form Analyzer
### Balanced Multi-Player Generation 

Creates exactly 100 innings per player for 20 players (2000 total innings) with realistic distributions.

## 📝 Notes

- All splits maintain chronological order to respect time-series nature of cricket data [file:1]
- The model uses only past data to predict future performance (no data leakage) [file:1]
- Feature engineering focuses on recent form indicators and career statistics [file:1]
- Validation and test sets are kept completely separate from training [file:1]

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional feature engineering (venue-specific stats, opposition quality)
- Alternative algorithms (XGBoost, LSTM for time-series)
- Enhanced player form metrics (consistency, momentum)
- Integration of bowling and match context features

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- Data source: T20 cricket match JSON files from cricksheet.org
- Built with scikit-learn, pandas, and matplotlib
- Inspired by cricket analytics and player form prediction research

---

**Author**: [Akshat ]  
**Date**: 5 December 2025  
**Version**: 1.0



