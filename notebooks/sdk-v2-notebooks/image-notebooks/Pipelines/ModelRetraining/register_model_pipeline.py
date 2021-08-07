
import argparse
import json

from azureml.core import Dataset, Model
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace
from azureml.train.automl.run import AutoMLRun
from azureml.pipeline.core import PipelineRun

parser = argparse.ArgumentParser()
parser.add_argument("--ds_name", help="input dataset name")
parser.add_argument("--model_name", help="model name to register")
parser.add_argument("--model_path",  help="model file from training run")
parser.add_argument("--metrics_path",  help="metrics file from training run")
parser.add_argument('--register_always', default=False, type=lambda x: (str(x).lower() == 'true'))
args = parser.parse_args()

print("Argument 1(ds_name): %s" % args.ds_name)
print("Argument 2(model_name): %s" % args.model_name)
print("Argument 3(model_path): %s" % args.model_path)
print("Argument 4(metrics_path): %s" % args.metrics_path)
print("Argument 5(register_always): %s" % args.register_always)

run = Run.get_context()
ws = None
if type(run) == _OfflineRun:
    ws = Workspace.from_config()
else:
    ws = run.experiment.workspace

if args.model_name == 'automl_image_model':
    print('Default model name provided. Using auto-generated model name from automl child run.')

    # Retrive the model name from automl's best child run if model name is not provided in arguments.
    pipeline_run = run.parent
    pipeline_run.__class__ = PipelineRun

    for step in pipeline_run.get_steps():
        if step.name == 'automl_module':
            automl_step_run_id = step.id
            break

    automl_run = AutoMLRun(experiment=run.experiment,
                           run_id=automl_step_run_id)

    best_child_run = automl_run.get_best_child()
    model_name = best_child_run.properties['model_name']
    print('model name obtained from the AutoML best child run is : {0}'.format(model_name))
else:
    model_name = args.model_name
    print('model name obtained from the model_name argument is : {0}'.format(model_name))

# Get the training dataset
train_ds = Dataset.get_by_name(ws, args.ds_name)
datasets = [(Dataset.Scenario.TRAINING, train_ds)]

# Get the metrics data
try:
    with open(args.metrics_path) as f:
        metrics_data = json.load(f)
except Exception as e:
    print("Could not load the metrics file.")
    raise

# Retrieve the hd run's data
hd_run_data = metrics_data[metrics_data.keys()[0]]

# Retrieve the best child run's data
hd_run_data = hd_run_data['best_child_by_primary_metric']

# Retrieve the best child run's metrics - uses the final flag in metrics data.
best_run_index = [index for index, final in enumerate(hd_run_data['final']) if final is True]

# Retrieve the best run's metric name and value
metric_value = hd_run_data['metric_value'][best_run_index[0]]
metric_name = hd_run_data['metric_name'][best_run_index[0]]
print("The best run obtained an {0} of {1}".format(metric_name, metric_value))

tags = {metric_name: metric_value}
model = None

# Get the latest registered model if any exists
try:
    model = Model(ws, model_name)
    last_train_time = model.created_time
    print("Model with name {0} already exists.".format(model_name))
    print("Model was last trained on {0}.".format(last_train_time))
except Exception as e:
    print("No model already existing with name {0}".format(model_name))

if model is None or args.register_always:
    if args.register_always:
        print("register_always switch is On. Proceeding to register the model..")
    # Register the model with the training dataset and the metrics tag. 
    model = Model.register(workspace=ws, 
                           model_path=args.model_path,
                           model_name=model_name,
                           tags=tags,
                           datasets=datasets)
    print("Registered version {0} of model {1}".format(model.version, model.name))
else:
    # Retrieve the metrics of the existing model if any.
    current_metric_val = 0
    if model.tags:
        if  metric_name in model.tags:
            current_metric_val = model.tags[metric_name]            
        if metric_value > float(current_metric_val):
            print("New model has a {0} value of {1} which is better than the existing model's value of {2}".format(metric_name, metric_value, current_metric_val))
            print("Registering the model..")
            # Register the model with the training dataset and the metrics tag. 
            model = Model.register(workspace=ws,
                                   model_path=args.model_path,
                                   model_name=model_name,
                                   tags=tags,
                                   datasets=datasets)
            print("Registered version {0} of model {1} with {2}:{3}".format(model.version, model.name, metric_name, metric_value))
        else:
            # Do not register the model as its not better than the current model.
            print("Current model has a {0} value of {1} which is better than or same as the new model's value of {2}".format(metric_name, current_metric_val, metric_value))
            print("No model registered from this run.")
    else:
        print('No metrics found for the existing model. Registering model with metrics..')
        # Register the model with the training dataset and the metrics tag. 
        model = Model.register(workspace=ws, 
                               model_path=args.model_path,
                               model_name=model_name,
                               tags=tags,
                               datasets=datasets)
        print("Registered version {0} of model {1} with {2}:{3}".format(model.version, model.name, metric_name, metric_value))
