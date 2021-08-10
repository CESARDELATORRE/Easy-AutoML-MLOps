from azureml.core.experiment import Experiment
from azureml.core import Dataset, Run
from sklearn.externals import joblib

from azureml.automl.core.shared.constants import MODEL_PATH

train_experiment_name = '<<train_experiment_name>>'
train_run_id = '<<train_run_id>>'
target_column_name = '<<target_column_name>>'
test_dataset_name = '<<test_dataset_name>>'

run = Run.get_context()
ws = run.experiment.workspace

# Get the AutoML run object from the experiment name and the workspace
train_experiment = Experiment(ws, train_experiment_name)
automl_run = Run(experiment=train_experiment, run_id=train_run_id)

# Download the trained model from the artifact store
automl_run.download_file(name=MODEL_PATH, output_file_path='model.pkl')

# get the input dataset by name
test_dataset = Dataset.get_by_name(ws, name=test_dataset_name)

X_test_df = test_dataset.drop_columns(columns=[target_column_name]).to_pandas_dataframe().reset_index(drop=True)
y_test_df = test_dataset.with_timestamp_columns(None).keep_columns(columns=[target_column_name]).to_pandas_dataframe()

fitted_model = joblib.load('model.pkl')

y_pred, X_trans = fitted_model.rolling_evaluation(X_test_df, y_test_df.values)

# Add predictions, actuals, and horizon relative to rolling origin to the test feature data
assign_dict = {'horizon_origin': X_trans['horizon_origin'].values, 'predicted': y_pred,
               target_column_name: y_test_df[target_column_name].values}
df_all = X_test_df.assign(**assign_dict)

file_name = 'outputs/predictions.csv'
export_csv = df_all.to_csv(file_name, header=True)

# Upload the predictions into artifacts
run.upload_file(name=file_name, path_or_stream=file_name)
