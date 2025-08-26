#COILED memory 8 GiB
#COILED ntasks 10

#COILED region us-east-2

import os
print("Hello from", os.environ["COILED_ARRAY_TASK_ID"])
