# High Performance Distributed Environment (HPDE) Cluster: A Guide for Local and Remote Training

This guide provides instructions for setting up and executing training jobs in a High Performance Distributed Environment (HPDE) Cluster. The guide is divided into sections for better readability and easy navigation.

## Preparatory Steps

Before you begin, ensure that `determined` is installed in your base environment. Additionally, establish a symbolic link between `.detignore` and `.gitignore` as follows:

```shell
ln -s .gitignore .detignore
```

The purpose of the `.detignore` file is to list the file paths that should not be packaged as part of the model definition, including data sets, checkpoints, compiled binaries, etc. The `.detignore` file uses the same syntax as `.gitignore` files, hence the symlink is created between the two files. For more information, see the Determined documentation: [Create an Experiment](https://docs.determined.ai/latest/model-dev-guide/submit-experiment.html#create-an-experiment).

## Executing a Single CPU/GPU Training Job Locally

To train a single model on a single slot (CPU or GPU computing device) locally, follow these steps:

1. **Installation of Docker:** Ensure Docker is installed on your local machine. If not, follow the instructions from the Determined AI [documentation](https://docs.determined.ai/latest/setup-cluster/deploy-cluster/on-prem/requirements.html#install-docker).

2. Make sure Docker is running in the background. For MacOS, you can start Docker by opening the Docker app. Please note that Docker on macOS does not support containers that use GPUs. Therefore, macOS Determined agents can only run CPU-based workloads.

3. **Local Training:** Initiate a local cluster by running the following command:

   ```shell
   det deploy local cluster-up
   ```

   If your local machine does not support an NVIDIA GPU, include the `--no-gpu` option:

   ```shell
   det deploy local cluster-up --no-gpu
   ```

4. **Creating an Experiment:** Navigate to the root directory of the `ai-lab-RxConnect` repository and create an experiment by specifying the configuration file:

   ```shell
   det --master localhost experiment create </path/to/config> .
   ```

   The trailing dot (.) argument uploads all of the files in the current directory as the context directory for the model. Determined will then copy the contents of the model context directory to the trial container's working directory.

   Upon successful creation of the experiment, you will see a message indicating the start of the experiment:

   ```shell
   Preparing files to send to master... XXX KB and YYY files
   Created experiment ZZZ
   ```

   **NOTE**: To automatically stream log messages for the first trial in an experiment to `stdout`, specify the configuration file and context directory as follows:

   ```shell
   det -m localhost e create </path/to/config/> . -f
   ```

   Here, `-m`, `e`, and `-f` are shorthand for `--master`, `experiment`, and `--follow`, respectively.

5. **Monitoring the Experiment via WebUI:**
   To monitor the progress of the experiment, open <http://localhost:8080/> in your browser. Use the default `determined` username; no password is required.

## Executing a GPU Training Job Remotely

To train a single model on a single slot (GPU computing device) remotely, follow these steps:

**Distributed Cluster Configuration:**
   A Determined cluster comprises a master and one or more agents. The master centralizes the management of agent resources. Modify the `resources.slots_per_trial` field to a value greater than 1 to request multiple GPU resources for the current trial. The `slots_per_trial` value must be divisible by the number of GPUs per machine.

1. **Connect to the Remote Master:**
   Set the remote IP address and port number in the `DET_MASTER` environment variable:

   ```shell
   export DET_MASTER="172.31.91.93:8080"
   ```

   Add this to your login shell's configuration file (e.g., `.bashrc` or `.zshrc`).

2. **Login:**

   ```shell
   det user login <ws[1,2,3]user[01,02,...,10]>
   ```

   A password is not required.

3. **Creating an Experiment:**
   Navigate to the root directory of the `ai-lab-RxConnect` repository and create an experiment by specifying the configuration file:

   ```shell
   det experiment create </path/to/config> .
   ```

   **NOTE:** You can also use the `-m` option to specify a remote master IP address:

   ```shell
   det -m http://172.31.91.93:8080/ experiment create </path/to/config> .
   ```

   Upon successful creation of the experiment, you will see a message indicating the start of the experiment:

   ```shell
   Preparing files to send to master... XXX KB and YYY files
   Created experiment ZZZ
   ```

4. **Monitoring the Experiment via WebUI:**
   To monitor the progress of the experiment, open <http://172.31.91.93:8080/> in your browser. Use the username (ws[1,2,3]user[01,02,..,10]); no password is required.

5. **Navigate to the Experiment:**
   Click the `Experiment` name to view the experiment’s trial display.

## Environment Variables

- **DET_MASTER:** The network address of the master of the Determined installation. The value can be overridden using the `-m` flag.

- **DET_USER** and **DET_PASS**: Specifies the current Determined user and password for use when non-interactive behaviour is required such as scripts. `det user login` is preferred for normal usage. Both **DET_USER** and **DET_PASS** must be set together to take effect. These variables can be overridden by using the `-u` flag.
