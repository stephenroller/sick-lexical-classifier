#!/bin/bash

export USE_CONDOR=1
export GLOBAL_OPTIONS=""

function run_setting {
    outputname=$1
    shift
    echo condorizer --output output/log.${outputname} "python eval.py $GLOBAL_OPTIONS --output output/$outputname $@"
}

# export GLOBAL_OPTIONS="--detailed" # uncomment for extra output

# Options for recreating results

# neutral setting, always guess baseline behavior
run_setting neu

# base features only
run_setting base --head --align --base
# wordform only
run_setting wf --head --align --wf
# asym only
run_setting asym --head --align --asym
# distributional similarity only
run_setting ds --head --align --dist
# wordnet only
run_setting wn --head --align --wn

