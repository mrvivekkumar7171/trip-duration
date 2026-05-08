## trip-duration

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Trip Duration**
A sample ML project to build an end-to-end working app for the NYC traxi trip duration challenge. A working setup demo is fully recorded.
We are creating a service that will take a few specific inputs and prediction the trip duration. This will be hosted on internet and 
User will hit the url with data and the endpoint will return the trip duration after model inference/prediction.
https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data

Steps:
1. create the project using ccds
2. collect the data & add data in raw folder & write connectors to access data globally
3. Initialize the git & dvc & connect to GCP cloud as remote storage
4. track the data raw & model folder
5. EDA on notebook
6. write modules code in src folder
7. add code to create the app.py
8. create stages to run with dvc
9. Create a brach for the best model
10. connect GCP storage with github
11. create a self-hosted runner on GCP
12. Use CI with specific best model branch
13. CI (Setup-code, python, dvc, run pipeline, model and data access, Report via CML on model matrix, docker/containerize the image and push to AWS ECR/Docker Hub)
14. Merge best model branch with main
15. Deploy best model on EC2 via the app.py

NOTE: Make sure data related changes are tracked and commit to git and push correctly.

Trip Duration Prediction Project with end-to-end mlops

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
