import argparse
import random
import time

from azureml.core import Run

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--x1', help='first script arg')
parser.add_argument('--x2', help='second script arg')
parser.add_argument('--x3', help='third script arg')
args = parser.parse_args()

# start an Azure ML run
run = Run.get_context()

# Check arguments
random.seed()
time.sleep(1)
print(">>>Logging accuracy")
run.log('Accuracy', random.random())

print(f">>>Logging x1: {args.x1}")
run.log('x1', args.x1)

print(f">>>Logging x2 {args.x2}")
run.log('x2', args.x2)

print(f">>>Logging x3 {args.x3}")
run.log('x3', args.x3)

run.flush()