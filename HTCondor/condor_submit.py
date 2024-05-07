import argparse
import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Submission:
    """
    Submit NNTrainer Jobs to HTCondor batch systems.

    Parameters
    ----------
        working_dir : str
            The working directory for the job.
        conda_install_dir : str
            The directory where Conda is installed.
        env_name : str
            The name of the Conda environment.
        framework_dir : str
            The directory of the framework.
        config_files : List[str]
            List of configuration files.

    Attributes
    ---------
        working_dir : Path
            The working directory for the job.
        conda_install_dir : str
            The directory where Conda is installed.
        env_name : str
            The name of the Conda environment.
        framework_dir : str
            The directory of the framework.
        config_files : List[str]
            List of configuration files.
        log_dir : Path
            The directory for job logs.
        bash_script_path : Path
            The path to the bash script.
        job_args_file : Path
            The path to the job arguments file.
        sub_file_path : Path
            The path to the HTCondor submission file.
    """

    def __init__(
        self, working_dir, conda_install_dir, env_name, framework_dir, config_files
    ):
        self.working_dir = Path(working_dir)
        self.conda_install_dir = conda_install_dir
        self.env_name = env_name
        self.framework_dir = framework_dir
        self.config_files = config_files
        self.log_dir = self.working_dir / "logs"
        self.bash_script_path = self.working_dir / "train_condor.sh"
        self.job_args_file = self.working_dir / "job_args.txt"
        self.sub_file_path = self.working_dir / "train_condor.sub"

    def write_bash_script(self):
        """Writes the bash script for the job."""
        with open(self.bash_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Path to the environment we need\n")
            f.write(f'CONDA_INSTALL_DIR="{self.conda_install_dir}"\n')
            f.write('export PATH="$CONDA_INSTALL_DIR/bin:$PATH"\n\n')
            f.write("# Source the conda environment\n")
            f.write('source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"\n')
            f.write('echo "Setting up the correct environment..."\n\n')
            f.write("# Activate the environment\n")
            f.write(f"conda activate {self.env_name}\n")
            f.write('echo "Environment activated! :D"\n\n')
            f.write("# Go to the framework directory\n")
            f.write(f"cd {self.framework_dir}\n")
            f.write('echo "Checking current directory and its contents:"\n')
            f.write("pwd\n")
            f.write("ls -l\n\n")
            f.write("# Debug: Check if PyTorch is installed\n")
            f.write(
                f'$CONDA_INSTALL_DIR/envs/{self.env_name}/bin/python -c "import torch; print(torch.__version__)"\n\n'
            )
            f.write("# Train the model\n")
            f.write(
                f"$CONDA_INSTALL_DIR/envs/{self.env_name}/bin/python main.py -c $(config_file)\n"
            )
            f.write('echo "Training completed"\n')

    def write_sub_file(self):
        """Writes the HTCondor submission file."""
        with open(self.sub_file_path, "w") as f:
            f.write("universe = vanilla\n")
            f.write(f"executable = {self.bash_script_path}\n")
            f.write("getenv = True\n")
            f.write("Request_GPUs = 1\n")
            f.write("request_CPUs = 1\n")
            f.write(f"request_memory = 16 GB\n")
            f.write(f"output = {self.log_dir}/$(ClusterId).$(ProcId).out\n")
            f.write(f"error = {self.log_dir}/$(ClusterId).$(ProcId).err\n")
            f.write(f"log = {self.log_dir}/$(ClusterId).log\n")
            f.write("should_transfer_files = YES\n")
            f.write("when_to_transfer_output = ON_EXIT\n")
            f.write('+JobFlavour = "workday"\n')
            f.write('+queue="short"\n')
            f.write("arguments = $(args)\n")
            f.write(f"queue args from {self.job_args_file}\n")

    def setup_job_args(self):
        """Sets up the job arguments file."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.job_args_file, "w") as f:
            for config_file in self.config_files:
                f.write(f"config_file={config_file}\n")

    def submit(self, dry_run=False):
        """
        Submits the job to HTCondor.
        """
        if dry_run:
            logging.info("=================================")
            logging.info("Dry run mode. Jobs not submitted.")
            logging.info("=================================")
            logging.info(
                f"You can find all the relevant files in the working directory:"
            )
            logging.info(f"{self.working_dir}")
        else:
            with subprocess.Popen(
                ["condor_submit", str(self.sub_file_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as process:
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    logging.info(
                        "====================================================="
                    )
                    logging.error(
                        f"Job submission failed with error: {stderr.decode()}"
                    )
                    logging.info(
                        "====================================================="
                    )
                else:
                    logging.info("==============================================")
                    logging.info(f"Job submitted successfully: {stdout.decode()}")
                    logging.info("===============================================")


def main():
    parser = argparse.ArgumentParser(description="Generate HTCondor submission files.")
    parser.add_argument(
        "-w",
        "--working_dir",
        type=str,
        required=True,
        help="Working directory for the job.",
    )
    parser.add_argument(
        "-conda_dir",
        "--conda_install_dir",
        type=str,
        required=True,
        help="Conda installation directory.",
    )
    parser.add_argument(
        "-e", "--env_name", type=str, required=True, help="Conda environment name."
    )
    parser.add_argument(
        "-f", "--framework_dir", type=str, required=True, help="Framework directory."
    )
    parser.add_argument(
        "-c",
        "--config_files",
        nargs="+",
        required=True,
        help="List of configuration file paths.",
    )
    parser.add_argument(
        "-n",
        "--dry_run",
        action="store_true",
        help="Perform a dry run without submitting the job.",
    )

    args = parser.parse_args()

    submission = Submission(
        args.working_dir,
        args.conda_install_dir,
        args.env_name,
        args.framework_dir,
        args.config_files,
    )
    submission.setup_job_args()
    submission.write_bash_script()
    submission.write_sub_file()
    submission.submit(args.dry_run)


if __name__ == "__main__":
    main()
