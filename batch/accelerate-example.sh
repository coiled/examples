#!/usr/bin/env bash

#COILED n-tasks 20
#COILED vm-type g6.xlarge 
#COILED task-on-scheduler True

accelerate launch \
    --multi_gpu \
    --machine_rank $COILED_BATCH_TASK_ID \
    --main_process_ip $COILED_BATCH_SCHEDULER_ADDRESS \
    --main_process_port 12345 \
    --num_machines $COILED_BATCH_TASK_COUNT \
    --num_processes $COILED_BATCH_TASK_COUNT \
    nlp_example.py 
