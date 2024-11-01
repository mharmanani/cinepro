import argparse
import omegaconf
import rich
import rich.pretty
import datetime
import submitit
import os

from medAI.utils.reproducibility import set_global_seed

from src.train_sam import BKSAMExperiment
from src.train_medsam import BKMedSAMExperiment
from src.train_cinepro import CineproExperiment
from src.train_semi_sl import SemiSLExperiment
from src.train_unet import UNetExperiment

class Main:
    def __init__(self, conf):
        self.args = conf

    def __call__(self):
        SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
        os.environ["TQDM_MININTERVAL"] = "30"
        os.environ["WANDB_RUN_ID"] = f"{SLURM_JOB_ID}"
        os.environ["WANDB_RESUME"] = "allow"
        CKPT_DIR = f'/checkpoint/{os.environ["USER"]}/{SLURM_JOB_ID}'

        conf.slurm_job_id = SLURM_JOB_ID
        conf.checkpoint_dir = CKPT_DIR

        if "medsam" in conf.mode:
            experiment = BKMedSAMExperiment(self.args)
        elif "cinepro" in conf.mode:
            experiment = CineproExperiment(self.args)
        elif "bksam" in conf.mode:
            experiment = BKSAMExperiment(self.args)
        elif "unet" in conf.mode:
            experiment = UNetExperiment(self.args)
        
        experiment.run()

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(Main(self.args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BK-Found')
    
    parser.add_argument('-y', '--yaml', type=str, default='medsam_base')
    parser.add_argument('-o', '--overrides', nargs="+", default=[])
    
    args = parser.parse_args()
    conf = omegaconf.OmegaConf.load(f"config/{args.yaml}.yaml")

    if conf.seed: # If seed is set, set the global seed
        set_global_seed(conf.seed)

    # Override config with command line arguments
    conf = omegaconf.OmegaConf.merge(conf, omegaconf.OmegaConf.from_dotlist(args.overrides))

    # Save config
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open (f"config/log/{conf.mode}_{timestamp}.yaml", 'w') as f:
        omegaconf.OmegaConf.save(conf, f)

    qos_to_time = {'m3': '4:00:00', 'm2': '8:00:00', 'm': '12:00:00', 'normal': '12:00:00', 'long': '24:00:00', 'deadline': '12:00:00'}

    if not conf.debug: 
        executor = submitit.AutoExecutor(folder="logs", slurm_max_num_timeout=10)
        executor.update_parameters(
            slurm_mem=conf.slurm.mem,
            slurm_gres='gpu:a40:1', 
            cpus_per_task=16,
            slurm_qos=conf.slurm.qos,
            slurm_account="deadline" if conf.slurm.qos == "deadline" else None,
            stderr_to_stdout=True,
            slurm_time = qos_to_time[conf.slurm.qos],
            slurm_name=conf.mode
        )

        job = executor.submit(Main(conf))
        print(f"Submitted job {job.job_id}")
        print(f"Logs at {job.paths.stdout}")

    else: 
        conf.data.batch_size = 1
        conf.device = 'cpu'
        if "medsam" in conf.mode:
            if "semi_sl" in conf.mode:
                experiment = SemiSLExperiment(conf).run()
            else:
                experiment = BKMedSAMExperiment(conf).run()
        elif "cinepro" in conf.mode:
            experiment = CineproExperiment(conf).run()
        elif "bksam" in conf.mode:
            if "semi_sl" in conf.mode:
                experiment = SemiSLExperiment(conf).run()
            experiment = BKSAMExperiment(conf).run()
        elif "unet" in conf.mode:
            experiment = UNetExperiment(conf).run()
        else:
            print("Invalid mode")
            exit(1)