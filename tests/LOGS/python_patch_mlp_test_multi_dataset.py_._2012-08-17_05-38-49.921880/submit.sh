                #!/bin/sh

                ## Reasonable default values
                # Execute the job from the current working directory.
                #PBS -d /home/gulcehre/codes/python/experiments/arcade_ds_exps/pretrained_mlp/prmlp_clean/tests

                #All jobs must be submitted with an estimated run time
                #PBS -l walltime=47:59:59

                ## Job name
                #PBS -N dbi_cc475360df7

                ## log out/err files
                # We cannot use output_file and error_file here for now.
                # We will use dbi_...out-id and dbi_...err-id instead
                #PBS -o /home/gulcehre/codes/python/experiments/arcade_ds_exps/pretrained_mlp/prmlp_clean/tests/LOGS/python_patch_mlp_test_multi_dataset.py_._2012-08-17_05-38-49.921880/dbi_cc475360df7.out
                #PBS -e /home/gulcehre/codes/python/experiments/arcade_ds_exps/pretrained_mlp/prmlp_clean/tests/LOGS/python_patch_mlp_test_multi_dataset.py_._2012-08-17_05-38-49.921880/dbi_cc475360df7.err

                ## Number of CPU (on the same node) per job
                #PBS -l nodes=1:ppn=1

                ## Execute as many jobs as needed

                #PBS -t 0-0

                ## Memory size (on the same node) per job
                #PBS -l mem=4000mb
            export OMP_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1

                ## Variable to put into the environment
                #PBS -v OMP_NUM_THREADS,GOTO_NUM_THREADS,MKL_NUM_THREADS

                ## Execute the 'launcher' script in bash
                # Bash is needed because we use its "array" data structure
                # the -l flag means it will act like a login shell,
                # and source the .profile, .bashrc, and so on
                /bin/bash -l -e /home/gulcehre/codes/python/experiments/arcade_ds_exps/pretrained_mlp/prmlp_clean/tests/LOGS/python_patch_mlp_test_multi_dataset.py_._2012-08-17_05-38-49.921880/launcher
