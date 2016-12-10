#!/bin/bash

function run_setting {
    outputname=$1
    shift
    condorizer python eval.py --output output/$outputname $@ logs/$outputname
}

run_setting allitems_lower --detailed
run_setting lexical_lower --nophrase --detailed
run_setting phrasal_lower --nolex --detailed

run_setting allitems_upper --gold --detailed
run_setting lexical_upper --nophrase --gold --detailed
run_setting phrasal_upper --nolex --gold --detailed

run_setting allitems_allfeats --head --align --memo --wf --px --dist --wn --base --asym
run_setting allitems_allfeats-memo-asym --head --align --wf --px --dist --wn --base --detailed
run_setting allitems_allfeats-memo --head --align --wf --px --dist --wn --base --asym --detailed
run_setting lexical_allfeats --head --align --memo --wf --px --dist --wn --base --asym --nophrase
run_setting lexical_allfeats-memo --head --align --wf --px --dist --wn --base --asym --nophrase --detailed
run_setting lexical_allfeats-memo-asym --head --align --wf --px --dist --wn --base --nophrase --detailed
run_setting phrasal_allfeats --head --align --memo --wf --px --dist --wn --base --asym --nolex
run_setting phrasal_allfeats-memo --head --align --wf --px --dist --wn --base --asym --nolex --detailed
run_setting phrasal_allfeats-memo-asym --head --align --wf --px --dist --wn --base --nolex --detailed

run_setting allitems_dist --head --align --dist --detailed
run_setting alltiems_wf --head --align --wf --detailed
run_setting allitems_base --head --align --base --detailed
run_setting allitems_wn --head --align --wn --detailed
run_setting allitems_px --head --align --px --detailed

#run_setting allitems_wf+dist --head --align --dist --wf --detailed
#run_setting lexical_wf+dist --head --nophrase --dist --wf --detailed
#run_setting phrasal_wf+dist --head --nolex --dist --wf --detailed

run_setting allitems_px --px --head --detailed
run_setting lexical_px --px --head --nophrase --detailed
run_setting phrasal_px --px --head --nolex --detailed

run_setting lexical_head_base --head --base --nophrase --detailed
run_setting lexical_head_dist --head --dist --nophrase --detailed
run_setting lexical_head_wn --head --wn --nophrase --detailed
run_setting lexical_head_px --head --px --nophrase --detailed
run_setting lexical_head_wf --head --wf --nophrase --detailed
run_setting lexical_memo --memo --nophrase

run_setting phrasal_align_dist --align --dist --nolex --detailed
run_setting phrasal_align_wn --align --wn --nolex --detailed
run_setting phrasal_align_wf --align --wf --nolex --detailed
run_setting phrasal_memo --memo --nolex
run_setting phrasal_align_base --align --base --nolex --detailed

run_setting lexical_head_asym --nophrase --head --asym --detailed
run_setting lexical_head_lhsvec --nophrase --head --lhsvec --detailed
run_setting lexical_head_rhsvec --nophrase --head --rhsvec --detailed
run_setting lexical_head_lhsvecrhsvec --nophrase --head --rhsvec --lhsvec --detailed
run_setting lexical_head_lhsvecrhsvecasym --nophrase --head --rhsvec --lhsvec --asym --detailed

run_setting allitems_asym --head --asym

