import sys
import os
from subprocess import check_output
from textwrap import dedent


def submit(cfg, hcfg, exp_dir, log):

    overrides = list(hcfg.overrides.task)
    overrides = list(filter(lambda arg: "submit=" not in arg, overrides))
    overrides.append(f"hydra.run.dir={hcfg.runtime.output_dir}")

    match cfg.cluster.scheduler:
        case "pbs":
            cmd = get_submit_cmd_pbs(cfg, hcfg, overrides)
        case "moab":
            cmd = get_submit_cmd_moab(cfg, hcfg, overrides)
        case "slurm":
            cmd = get_submit_cmd_slurm(cfg, hcfg, overrides)
        case _:
            log.error(f'Unknown cluster scheduler "{cfg.cluster.scheduler}"')
            sys.exit()

    log.info(f"Executing in shell: {cmd}")
    jobid = None
    if not cfg.dry_run:
        jobid = check_output(cmd, shell=True, executable="/bin/bash")
        jobid = jobid.decode().strip()

    if jobid:
        log.info(f"Submitted job: {jobid}")


def get_submit_cmd_pbs(cfg, hcfg, overrides):

    ccfg = cfg.cluster
    device = cfg.device or r"\`tail -c 2 \$PBS_GPUFILE\`"
    num_gpus = ccfg.num_gpus if cfg.use_gpu else 0
    dependency = f"#PBS -W depend={ccfg.dependency}" if ccfg.dependency else ""
    cmd = dedent(
        f"""
        qsub <<EOT
        #PBS -N {cfg.run_name}
        #PBS -q {ccfg.queue}
        #PBS -l nodes={ccfg.node}:ppn={ccfg.procs or 1}:gpus={num_gpus}:{ccfg.queue}
        #PBS -l walltime={ccfg.time},mem={ccfg.mem},vmem={ccfg.vmem}
        #PBS -o {hcfg.runtime.output_dir}/pbs.log
        #PBS -j oe
        {dependency}
        cd {os.environ['AUSSIE_DIR']}
        source setup.sh
        export CUDA_VISIBLE_DEVICES={device}
        python aussie.py -cn {hcfg.job.config_name} {' '.join(overrides)} 
        exit 0
        EOT
    """
    )

    return cmd


def get_submit_cmd_moab(cfg, hcfg, overrides):

    ccfg = cfg.cluster
    num_gpus = ccfg.num_gpus if cfg.use_gpu else 0
    cmd = dedent(
        f"""
        msub <<EOT
        #MSUB -N aussie_{cfg.run_name}
        #MSUB -l nodes=1:ppn={ccfg.procs}:gpus={num_gpus}
        #MSUB -l feature={ccfg.feature},pmem={ccfg.pmem},walltime={ccfg.time}
        #MSUB -o {hcfg.runtime.output_dir}/moab.log
        #MSUB -j oe
	    cd {os.environ['AUSSIE_DIR']}
        source setup.sh
        python aussie.py {' '.join(overrides)} -cn {hcfg.job.config_name}
        exit 0
        EOT
    """
    )

    return cmd


def get_submit_cmd_slurm(cfg, hcfg, overrides):

    ccfg = cfg.cluster
    setup_cmd = f"cd {os.environ['AUSSIE_DIR']}; ./setup.sh"
    script_cmd = f"python aussie.py -cn {hcfg.job.config_name} {' '.join(overrides)}"
    num_cpus = max(cfg.num_cpus, ccfg.num_cpus)
    num_gpus = ccfg.num_gpus if cfg.use_gpu else 0
    cmd = (
        f"sbatch -p {ccfg.queue} --mem {ccfg.mem} -N 1 -c {num_cpus}"
        f" --gres=gpu:{num_gpus} -t {ccfg.time} -J aussie_{cfg.run_name}"
        f' -o {hcfg.runtime.output_dir}/slurm.log --wrap "{setup_cmd}; {script_cmd}"'
    )

    if ccfg.dependency is not None:
        cmd += f" --dependency {ccfg.dependency}"

    if ccfg.node is not None:
        cmd += f" --nodelist {ccfg.node}"

    return cmd
