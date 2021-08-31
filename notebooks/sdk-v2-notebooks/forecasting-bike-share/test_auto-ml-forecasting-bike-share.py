import sys

from checknotebookoutput import checkNotebookOutput
from checkexperimentresult import checkExperimentResult
from checkexperimentresult import check_experiment_model_explanation_of_best_run
from download_run_files import download_run_files

# Download files for the remote run.
download_run_files(experiment_names=['automl-bikeshareforecasting', 'automl-bikeshareforecasting_test'],
                   download_all_runs=True)

checkExperimentResult(experiment_name='automl-bikeshareforecasting',
                      expected_num_iteration='1000',
                      expected_minimum_score=0.01,
                      expected_maximum_score=0.3,
                      metric_name='normalized_root_mean_squared_error',
                      absolute_minimum_score=0.0,
                      absolute_maximum_score=1.0)

check_experiment_model_explanation_of_best_run(experiment_name='automl-bikeshareforecasting')

# Check the output cells of the notebook.
checkNotebookOutput("auto-ml-forecasting-bike-share.nbconvert.ipynb" if len(sys.argv) < 2 else sys.argv[1],
                    "warning[except]retrying[except]UserWarning: Matplotlib is building the font cache"
                    "[except]warning: a newer version of conda exists"
                    "[except]UserWarning: Starting from version 2.2.1, "
                    "the library file in distribution wheels for macOS is built by the Apple Clang"
                    "[except]The following algorithms are not compatibile with lags and rolling windows"
                    "[except]brew install libomp"
                    "[except]If 'script' has been provided here"
                    "[except]If 'arguments' has been provided here")
