# PCC Postoperative Risk Prediction Model

Streamlit deployment package for the PCC random forest model.

## Files

- `app.py`: Streamlit application entrypoint
- `pcc/`: PCC model files and metadata
- `requirements.txt`: pinned Python dependencies
- `.streamlit/config.toml`: Streamlit app configuration

## Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open Streamlit Community Cloud and create a new app.
3. Select the GitHub repository and branch.
4. Set the main file path to `app.py`.
5. In Advanced settings, choose Python 3.12 if available.
6. Deploy.

The model pickle expects `scikit-learn==1.6.1`; keep the pinned versions unless the model is retrained.
