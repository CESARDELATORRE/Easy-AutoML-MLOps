# AutoML E2E Workflow - Notebooks


- [AutoML E2E Workflow - Notebooks](#automl-e2e-workflow---notebooks)
- [tl;dr](#tldr)
  - [Why?](#why)
  - [What?](#what)
  - [How?](#how)
- [Overview](#overview)
- [Terminology](#terminology)
- [Data Science Workflow](#data-science-workflow)
- [How does AutoML fit within the Data Science workflow?](#how-does-automl-fit-within-the-data-science-workflow)
- [Componentization](#componentization)
- [Unified AutoML](#unified-automl)
- [Final Proposal:](#final-proposal)

# tl;dr

## Why?

While SDKv1 was a one-stop shop for all things ML, such as submitting and managing experiments or training (AutoML, RL) + tracking + logging metrics and models, SDKv2 only aims to be a thin layer of APIs to manage jobs & assets (like deployments). The training or tracking aspects are delegated to user defined libraries or scripts & MLFlow.
AutoML is a unique offering in AzureML, which contains both - control plane (experiments, models) as well as data plane (training or logging metrics) aspects. As such, the AutoML notebooks need to be re-written to use the new (curated) SDKv2, along with showcasing how we can continue to do the data plane operations using MLFlow.

Moreover, we can also take this opportunity to revamp our notebook experience, by trying to solve some of the more common problems  that existed in V1, such as code duplication, relying on notebooks for internal testing, etc.

## What?

With componentization and introduction of new areas like Vision or NLP in our offerrings, we want to be able to showcase the ability to use and connect each individual component - such as Preprocessing, Model Training, Interpretation, Test, Deploy - in a real world Data Science scenario, using a consistent set of concepts and APIs.

## How?

Showcase a more realistic story of how AutoML fits within a data scientist's experiment. These can be a comprehensive real-world experiment that showcases how AutoML fits within the workflow for that class of tasks. (Tabular - Classification/Regression/Forecasting, Vision, NLP etc.) 

For a proof of concept, I'm using Porto Seguro dataset for a classification task, since it has many nuances that a real world dataset would most likely exhibit.  Different aspects of the notebooks:

**Preprocessing**

This notebook shows the first phase of a data scientist's journey - loading, exploring and pre-processing of data. The notebook begins with a raw dataset, and ends with AutoML' s 'Featurization' step - producing a preprocessing pipeline / model. The end result of this step is generally a means through which we can a dataset which is all numbers.

In general, if we include Vision / NLP this can be an unfitted / lazily evaluated pipeline, or just a plan of how we think the input data should be preprocessed. [Unified AutoML]

**Model Training** 

This notebook takes as inputs, the outputs of the featurization step - the preprocessor and / or the preprocessed dataset (after featurization), and submits a training job to find the best model for optimizing on a given primary metric.

The best model produced follows all the contracts of an MLFlow Model. As such, the model can be loaded (or served) locally in a compatible environment as defined by the MLFlow Model, and used for predictions or further interpretations.

**Interpretation**

This notebook takes in a model produced by the training step along with the raw dataset that was used to train the model, and can produce some explanations on why the model produces the predictions that it does. 

**Test**

This notebook takes in a model and a new test dataset (which has the same features as the one that was passed to the model training step), and produces a new dataset with the predictions, along with metrics if the actual targets were also available in the dataset.

**Deploy**

Being an MLFlow Model, here we can link to the examples that AzureML is going to use to showcase deployments that use an MLFlow model. Ideally, we shouldn't have anything AutoML specific to get the model be deployed on Azure.

# Overview

In the following sections, I try to explain the motivations behind the above proposal of how I think we should structure our Notebooks.

I begin with introducing key terminologies that are required to understand some the explanations that follows. That is followed by a general overview of the workflow that Data Scientists tend to follow for solving a problem in their domain. In the sections that follow, I try to explain how AutoML fits into the pipeline, and how some of our current and future plans (such as Componentization or parallel offerings such as NLP or Vision) combine together to provide a picture of how we should be presenting our notebooks to end users.

# Terminology

- **Client**
    - Module which the end users interact with. 
    E.g. With DPv2, this is the azure-ml Python SDK with curated and DSL APIs.
    - Contains job and asset management APIs  (aka. *Control Plane Operations*)
- **Runtime**
    - Module which developers & data scientists (internal, and some external, advanced users) interact with.
    E.g. `azureml.automl.runtime` is one such module at the moment
    - Contains resource interaction APIs (aka. *Data Plane Operations*), i.e. APIs which allows us to interact with the model or data.
    - Core ML code and logic reisde here.


# Data Science Workflow

The journey of a Data Scientist begins by framing a problem statement. Once the objective is specified, the next step entails figuring out what knobs can be tweaked to get to the desired state. These knobs (also called 'independent' variables) are the things that can be controlled that get us closer to the objective (also called the 'dependent' variable). These knobs form a wide array of features that are collected from different sources, that form the initial dataset. Only then, do we get down to the details of building predictive models.

> *Fun fact: About 50% of the time is spent in data collection and pre-processing, and 80% of that is manual (even with current AutoML tools).*

Building models is an iterative process, where once we have our first model (called the 'Baseline'), we ask it certain questions, such as what features are important to make a prediction, or what happens to the dependent variable if some feature in a given row were changed. That helps us get insights about the data itself, which leads to some changes, either in the dataset or in the model (or both). The process then repeats, where we continue to develop the model, tweaking either the data or the parameters that go into the model.

To keep this feedback loop fast and efficient, the data is often subsampled down (if it is large) and the model is developed locally. Once a good set of configuration is found, training happens on the whole dataset, and then the process logically flows into the territory of how to deploy it into production (or how to submit the results to Kaggle)

The kind of questions that get asked of a model (or the data) depends on the expertise of the Data Scientist. 'Citizen' data scientists, while learning, can rely on what is provided to them (e.g. from the UI), and make decisions about the usefulness of a model, while skilled professionals generally expect to get down to the model details, either inspecting it manually or using some libraries to do it for them (like SHAP, tree interpreters etc.).

This kind of model interpretation requires an interactive communication with the model - i.e., one has the model loaded in-memory and is able to see how data interacts with it. This requires that the 'development environment' is capable of loading the given model in the first place, and that the model has well defined APIs and semantics (e.g. adhereing to scikit learn PyTorch APIs)

# How does AutoML fit within the Data Science workflow?

On a high level, there are two big bucket use cases:

- Baselines - To try out different combinations of feature engineering and models to get to the starting point - where it can be asked questions (as described above)
- Tuning - Given a best 'feature engineered' model, find the best set of parameters that should go into the model.

For both cases, it is important to be able to load a model in an environment (Jupyter Notebook), once the model is generated.

If the model is generated by AutoML, we need the automl-runtime packages to be present. In SDKv1, we  have the whole environment pre-installed when installing AutoML (client & runtime). As such, submitting an AutoML experiment (via. AutoMLConfig) and loading of the subsequent model was convenient (and can happen, say, in the same Jupyter notebook), although we did encounter a few environment incompatibility issues along the way, as reported by users.

In SDK v2, there are no separate modules per area - SDK is a thin authoring-only layer for job & asset management operations. The onus is on the user to install the libraries separately if they are to load the model locally (MLFlow and automl-runtime). 

As such, given the separation of concerns between job submission ('control plane operations') and the actual resource access ('data plane operations'), it is logical that our samples portray that reality. That is, we separate our notebooks in a way that showcases job submissions in one part, while loading the model using MLFlow in a separate part.

> ***Draft Proposal:** Have a separate notebook where we can show how to submit AutoML Jobs (the only dependency here being the common azure-ml Python SDK), and another separate notebook which given the output of the job (i.e. an AutoML generated MLFlow Model), showcases how to load and use the model for interpretation or testing.*

# Componentization

We are already out to split up the monolith Client side APIs for AutoML job submission into smaller sub-components, based on the workflow described above. E.g. Featurization, Train, Test, Explain, etc. This is beneficial for the more advanced users of AutoML, who want a little more control on the end to end workflow. 

Each of these components will have it's own set of 'control' plane operations (e.g. for configuring and submitting jobs) and 'data' plane operations (e.g. for interpreting models & predictions, transformation pipelines) possibly with a different set of runtime library (e.g. metrics, responsible ai (?)). As such, our notebooks should showcase each of these steps in a data science workflow as standalone components. 

> ***Draft Proposal:** What used to be an end to end notebook in v1, we now divide them roughly into what maps to at least one of the above described components of a data science workflow. If each component wants to showcase more features, we can go a level deeper, and link to an '[examples](https://github.com/Azure/azureml-examples/tree/main/python-sdk)' directory on Github. But the core and the most often used feature must be shown on top level.*

We can surely have a quick-starter notebook (end to end workflow), which can use the more common starter datasets such as digits, MNIST. These can be considered as the entry points for a budding data scientist. It can showcase the new way of using AutoML via. v2 SDKs. This includes the ability to use and connect each individual component - such as Preprocessing (Tabular Featurization only today, but can extend to Augmentations for DNNs), Model Training, Interpretation, Test, Deploy, but only at the control-plane level.

> *Can we do `pip install azure-ml mlflow azureml-automl-runtime` and run everything in a single notebook, like SDKv1?*
         A major premise of MLFlow is that it comes with the environment definition, that guarantees to load the model in a 'verified' environment (i.e. a fresh environment is created where you load the model). If we do the above, and have an environment pre-baked, we may be moving away from what MLFlow is meant for. Plus, it brings us again into the familiar territory of environment inconsistencies (trained the model on version X but loading it locally on Y). Moreover, with componentization, the notebooks may get even more verbose.

# Unified AutoML

With Auto-Vision and Auto-NLP soon coming up, we want to have a unified front for all areas - for both the end users (via. Client SDK / pipeline + components) as well as internal data scientists (for bringing down the time needed to get ideas or research into code). 

We are converging on a consistent set of client APIs in a common SDK for Image and Tabular (via. Pipelines and components), so that end users have a common way to kick off their experiments. A data scientist who wants apply what she's learnt in one area should be able to apply that to a different area with the least amount of resistance (such as from images to text, transfer learning quite literally (:)

Likewise, we should ensure that the runtime SDKs also converge. This is much needed, so that our internal data scientists also have a common way to add features, use common infrastructure (such as error handling) and to bring down the time needed to get ideas or research into code, with the least amount of resistance. This is also required if we are to have a successful strategy for future applications, like multimodal training.
There area libraries out there which already provide such close knit APIs:

- [FastAI](https://docs.fast.ai/)
- [AutoKeras](https://autokeras.com/tutorial/overview/)
- [PyTorch Lightning Flash](https://lightning-flash.readthedocs.io/en/stable/?badge=stable)
- [AutoGluon](https://auto.gluon.ai/stable/index.html)

These are all ML runtime libraries, with a common set of APIs that allow for code-reuse and tight integrations across different areas.

The topic on how to converge runtime SDKs requires it's own discussion and specs, so we won't be talking about them here.

> ***Draft Proposal:** Each area (Vision, NLP, Tabular), must have a common way to do operations that fall into the steps as defined by the data science workflow above. E.g. Submitting a 'Trainng' or 'Test' run on a Tabular Model  should follow the same semantics as that for 'Vision' or 'NLP', with just the configuration that changes across them.*

# Final Proposal:

Consolidating each of the draft proposals above:

For each area (Vision, NLP, Tabular) - we should have at least one comprehensive example on a real-world dataset (not MNIST, Digits, bank classification etc.), that showcases the end to end data science workflow. These notebooks must include the descriptions of the nuances in the dataset and how AutoML helps overcome them, teaching users how different pieces connect (e.g. Based on exploratory data analysis: forcing a custom transformation on a column, blocking certain linear models because they won't do well, etc.). Jupyter notebooks are great at [storytelling](https://en.wikipedia.org/wiki/Literate_programming), and we should have one to tell using AutoML. Each area can have one notebook per area of focus (Data Exploration or Preprocessing, Model Training, Interpretation etc.)

Example: [Predicting koala populations](https://www.notion.so/4027ec902e239c93eaaa8714f173bcfc) (An all-in-one notebook, but notice how they tie together the EDA with the problem they're trying to solve)

For quick-starter style notebooks, each area must have minimal divergence (i.e. the Component APIs must not differ too much) - so as to allow a user well-versed in Tabular Learning to also be able to extend her knowledge into other areas like NLP, Vision.

