# Rx-Connect
Rx-Connect is an innovative cloud-based software system designed to streamline the prescription verification process in pharmacies. By leveraging state-of-the-art AI modules, Rx-Connect provides a platform for virtual verification and pharmacy operations, enhancing patient care, increasing efficiency, safety, and cost-effectiveness while reducing preventable medical errors.

## Features
TODO: Add features here as a bullet points.


## Development Setup
### Python Environment Creation
You must use a virtual environment for an isolated installation of this project. We recommended using either [conda](https://docs.conda.io/en/latest/miniconda.html) or [virtualenv](https://pypi.org/project/virtualenv/).
### `conda` environment
Download and setup [minconda](https://docs.conda.io/en/latest/miniconda.html) to get the `conda` tool.
Once available, create and activate the environment for this project as:
```shell script
conda create -y -n ${NAME_OF_THE_PROJECT} python=3.10
conda activate ${NAME_OF_YOUR_PROJECT}
```
When active, you can de-activate the environment with `conda deactivate`.

### `virtualenv` environment
Suggestion: use [`pipx` to install `virtualenv` in your system via](https://virtualenv.pypa.io/en/latest/installation.html#via-pipx): `pipx install virtualenv`.

To create and activate your environment with `virtualenv`, execute:
```shell script
virtualenv --python 3.10 ~/.venv/${NAME_OF_YOUR_PROJECT}
source ~/.venv/${NAME_OF_YOUR_PROJECT}/bin/activate
```


### Installing Dependencies and Project Code
This project uses [pip](https://pypi.org/project/pip/) for dependency management and [setuptools](https://pypi.org/project/setuptools/) for project code management.

#### Main Dependencies
To install the project’s dependencies & code in the active environment, perform:
```
pip install -r requirements.txt && pip install -e .
```

#### Test & Dev Dependencies
To install the testing and development tools in the environment, do:
```
pip install -e ".[dev]"
```

## Running Tests and Using Dev Tools
### Testing
TODO: Describe how to run the tests
### Dev Tools
This project uses several tool to maintain high-quality code:
- `mypy` for type checking
- `flake8` for linting
- `isort` for module import organization
- `black` for general code formatting
- `pre-commit` for enforcing use of the above tools

The configuration settings for these tools are defined at the root of the `AI-LAB-RXCONNECT` repository.
****
#### `pre-commit` hooks
**NOTE**: Be sure to first install the `pre-commit` hooks defined in the `.pre-commit-config.yaml` file. To install, execute `pre-commit install` from the repository root while environment is active.

**NOTE**: All code in the project _must_ adhere to using these dev tools _before_ being committed.
****

# Contribute
TODO: Explain how other users and developers can contribute to make your code better.
