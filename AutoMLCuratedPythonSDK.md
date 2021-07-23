# AutoML Curated Python SDK

- [AutoML Curated Python SDK](#automl-curated-python-sdk)
  - [Design decisions](#design-decisions)
  - [AutoMLJob](#automljob)
  - [Task](#task)
  - [Compute](#compute)
  - [Target](#target)
  - [Dataset](#dataset)
  - [Configuration (Optional)](#configuration-optional)
  - [Featurization (Optional)](#featurization-optional)
  - [Examples:](#examples)
    - [Bare minimum required config](#bare-minimum-required-config)
    - [Verbose](#verbose)
  - [Questions](#questions)

## Design decisions
- Promote all those properties that are shown on the UI. 
    - These are the bare minimum settings that is required by AutoML to kick off an experiment. Match the simplicity that UI provides to the SDK defaults at the top level (i.e. for `AutoMLJob`)

- Rename the properties instead of overriding it (e.g. data_settings === dataset). 
    - Consider using a separate class method (aka. alternate constructor) to accept job configuration, instead of polluting the default constructor (aka. `__init__`)
    - This makes explicit the default set of properties required. If we override the default constructor, it's not clear in the contract, of what are required and optional fields. E.g., the same setting can be provided in more than one way (e.g. dataset & DataSettings), so the documentation will suggest that both are optional.
    - If the user still wants to modify some properties that are not available in the alternate constuctor, they can do so via. first creating the job using the curated sdk, then creating the properties object (i.e. using the classes from the Rest based contracts), finally setting the property on the job object.</p>
            
    - Example:
        ```python
        job = AutoMLJob.build(task="", dataset={}, target="")
        complex_forecasting_settings = ForecastingSettings(...)
        job.forecasting_settings = complex_forecasting_settings
        ```
    
    - We can also go a level deeper in the future, using alternate constructors. For example, we can use task specific constructors.
        ```python
        job = AutoMLJob.forecasting(...)
        ```

## AutoMLJob

Top level promoted properties:
- task
- compute
- target 
- dataset
- configuration (optional)
- featurization (optional)


Renamed Props [new (old)]:
- dataset (data_settings)
- featurization (featurization_settings)

## Task
One of 'classification', 'regression' or 'forecasting' (promoted from AutoMLJob.general_settings)


## Compute
Name of the compute target. "local" if running locally (promoted from AutoMLJob.compute)


## Target
Name of the target column in the dataset (promoted from AutoMLJob.data_settings)


## Dataset
Top level promoted propeties:
- train
- valid (optional)
- test (optional)


## Configuration (Optional)

Club together all other settings (such as featurization, forecasting, exit criteria) under the 'configuration' key.

Top level promoted properties
> Note: Having most common forecasting related properties in the same top-level bucket. If more are required, club them together under the forecasting bucket
- primary_metric (should default based on task type)
- model_explainability (default = True)
- blocked_models
- exit_criterion 
    - timeout_hours
    - exit_score
- validation
    - valid_percent
    - n_cross_validation
- concurrency
    - max_concurrent_trials
- time_column_name (required if task == 'forecasting')
- time_series_identifiers (required if task == 'forecasting')
- frequency
- forecast_horizon
- enable_dnn
    

Renamed Props [new (old)]:
- exit_criterion (subset of LimitSettings)


## Featurization (Optional)
- blocked_transformers (str, or List[str])
- column_purposes (Dict[str, str]. Known as 'feature type' on UI.)
- drop_columns (str, or List[str]. Known as 'Included' on UI.)
- transformer_params (Known as 'Impute with' on the UI.)


## Examples:
### Bare minimum required config
```python
automl_job = AutoMLJob.build(
    task="classification",
    compute="cpu-cluster",
    target="target_column_name",
    dataset={
        "training": azureml_dataset,
    },
)
```

### Verbose
```python
automl_job = AutoMLJob.build(
    task="forecasting",
    compute="cpu-cluster", # 'local' if not using remote
    dataset={
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    },
    target="target_column_name",
    configuration={
        "primary_metric": "normalized_root_mean_squared_error",
        "blocked_models": [],
        
    },
    featurization={
        "blocked_transformers": "LabelEncoder",
        "column_purposes": {
            "VendorName": "CategoricalHash"
        },
        "drop_columns": ["MMIN"],
        "transformer_params": {
            "Imputer":  [
                {
                    "fields": ["MMIN"],
                    "parameters": {
                        "strategy": "constant",
                        "fill_value": 0
                    }
                },
                {
                    "fields": ["CHMIN"],
                    "parameters": {
                        "strategy": "median",
                        "fill_value": null
                    }
                }
            ],
            "hash_one_hot_encoder": [
                {
                    "fields": ["ModelName"],
                    "parameters": {
                        "number_of_bits": 3
                    }
                }
            ]
        }
    }
)
```


## Questions
- UI defaults for ONNX?
