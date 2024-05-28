import argparse
import os
from pathlib import Path
import logging
import jinja2
import subprocess

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# /eos/user/l/leevans/miniconda3
# /eos/user/l/leevans/NNTrainer/tth-network
# MLenv

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
        request_memory : str
            Amount of memory to request for the job.
        run_time : str
            The walltime for the job in seconds

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
        request_memory : str
            Amount of memory to request for the job.
        run_time : str
            The walltime for the job in seconds
    """

    def __init__(self, working_dir, conda_install_dir, env_name, framework_dir, config_files, request_memory, run_time):
        self.working_dir = Path(working_dir)
        self.conda_install_dir = conda_install_dir
        self.env_name = env_name
        self.framework_dir = framework_dir
        self.config_files = config_files
        self.log_dir = self.working_dir / "logs"
        self.bash_script_path = self.working_dir / "train_condor.sh"
        self.job_args_file = self.working_dir / "job_args.txt"
        self.sub_file_path = self.working_dir / "train_condor.sub"
        self.request_memory = request_memory
        self.run_time = run_time
        self.template_env = jinja2.Environment(loader=jinja2.FileSystemLoader('.'))

        logging.info("Submission configuration:")
        logging.info(f"  Working directory: {self.working_dir}")
        logging.info(f"  Conda installation directory: {self.conda_install_dir}")
        logging.info(f"  Conda environment name: {self.env_name}")
        logging.info(f"  Framework directory: {self.framework_dir}")
        logging.info(f"  Configuration files: {', '.join(self.config_files)}")
        logging.info(f"  Request memory: {self.request_memory}")
        logging.info(f"  Run time (s): {self.run_time}")

    def write_bash_script(self):
        """Writes the bash script for the job using Jinja2 templating."""
        template = self.template_env.get_template('jinja_templates/train_condor.sh.template')
        bash_script = template.render(
            conda_install_dir=self.conda_install_dir,
            env_name=self.env_name,
            framework_dir=self.framework_dir
        )
        with open(self.bash_script_path, 'w') as f:
            f.write(bash_script)

    def write_sub_file(self):
        """Writes the HTCondor submission file using Jinja2 templating."""
        template = self.template_env.get_template('jinja_templates/train_condor.sub.template')
        sub_file = template.render(
            bash_script_path=self.bash_script_path,
            log_dir=self.log_dir,
            job_args_file=self.job_args_file,
            request_memory=self.request_memory,
            run_time=self.run_time
        )
        with open(self.sub_file_path, 'w') as f:
            f.write(sub_file)

    def setup_job_args(self):
        """Sets up the job arguments file."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.job_args_file, "w") as f:
            for config_file in self.config_files:
                f.write(f"{config_file}\n")

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
        "-m",
        "--request_memory",
        type=str,
        default="24 GB",
        help="Amount of memory to request for the job. e.g. 40 GB"
        "Default is 24 GB."
    )
    parser.add_argument(
        "-r",
        "--run_time",
        type=str,
        default="43200",
        help="The run time in seconds for the job."
        "Default is 12hrs (43200s)."
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
        args.request_memory,
        args.run_time
    )
    submission.setup_job_args()
    submission.write_bash_script()
    submission.write_sub_file()
    submission.submit(args.dry_run)


if __name__ == "__main__":
    main()
