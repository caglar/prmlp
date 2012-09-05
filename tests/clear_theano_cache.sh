#!/bin/bash -x
hosts=(ngpu-a4-01 ngpu-a4-02 ngpu-a4-03 ngpu-a4-04 ngpu-a4-05 ngpu-a4-06 ngpu-a4-07 ngpu-a4-08 ngpu-a4-09)

for host in "${hosts[@]}";
    do
        echo $host
				ssh $host theano-cache clear;
    done


