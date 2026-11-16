# Gradient-Boosted Trees Workbench

An interactive Streamlit workbench that makes the steps of a small gradient-boosted regression tree visible: candidate splits, gain, leaf weights, residuals, iteration history, and baseline comparisons.

## Preview: split, residuals and the selected tree

![Gradient-boosting workbench with prediction-versus-actual and MAE-by-iteration charts, plus the selected tree's split summary.](docs/images/gradient-boosting-tree-explanation.png)

*This run uses the ten hand-authored sample rows. Train/test sets are tiny, so the visible metrics are only an illustration of the workflow and are not a generalization benchmark.*
## Learning objectives

- Inspect squared-error gradients and unit Hessians.
- Compare candidate split gains and regularized leaf weights.
- See the effect of learning-rate shrinkage and row/column subsampling.
- Separate model selection from test reporting with a validation split when enough rows are available.

## Run

    python -m pip install -r requirements.txt
    streamlit run xgboost_hand_calculation_streamlit.py

The sample dataset is editable. Choose the target, permitted features, model settings, and train/test ratio in the app.

## Evaluation design and limits

A seeded split creates training and test sets; when enough training rows are available, a separate validation portion selects the early-stopping iteration. Test rows are reserved for reporting. Small datasets remain noisy, and very small training sets disable early stopping.

This is a compact educational implementation inspired by the XGBoost objective. It is not the XGBoost package and omits production features such as histogram training, categorical splits, and native missing-value routing. The demonstration uses squared-error regression and does not establish predictive value for financial data.

- xgboost_hand_calculation_streamlit.py — tree fitting, split calculations, and interface.
- requirements.txt — app dependencies.
