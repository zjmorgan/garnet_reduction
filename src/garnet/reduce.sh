#!/bin/bash

WORKFLOW="/SNS/software/scd/garnet_reduction/src/garnet/workflow.py"

echo $WORKFLOW

REDUCTION="norm"

while getopts hni FLAG; do
    case $FLAG in
        h)
            echo "reduction.json processes"
            exit 1
            ;;
        n)
            REDUCTION="norm"
            ;;
        i)
            REDUCTION="int"
            ;;
    esac
done

shift $((OPTIND-1))

if [[ $# -ne 2 ]]; then
    echo "Requires input text file and number of processes"
    exit 1
fi

INPUT=$1
PROCESSES=$2

#nsd-conda-wrap.sh scd-reduction-tools-dev --classic $INSCAPE $INPUT $PROCESSES

CONDA="/opt/anaconda/bin/activate"
if [ ! -f $CONDA ]; then
    CONDA="$HOME/miniconda3/bin/activate"
fi

rm -rf ~/.cache/fontconfig

echo $CONDA
echo $INPUT
source "${CONDA}" scd-reduction-tools-dev
echo python $WORKFLOW $INPUT $REDUCTION $PROCESSES

python $WORKFLOW $INPUT $REDUCTION $PROCESSES
