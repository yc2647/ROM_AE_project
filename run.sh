#!/bin/bash
export N_TRAINING=900
export N_BASIS=30

echo "OFFLINE STAGE"

echo "This is supposed to be run on a normal laptop"
python3 main.py --action "initialize set" --set-name "training_set" --set-cardinality $N_TRAINING #> initialize.txt
echo "You can then send the generated training set to the cluster"

echo "This is supposed to be run on a cluster, possibly in parallel and with the for loop manually split in different jobs"
# change i to anything
for ((i=0; i<N_TRAINING; i++)); do
    mpirun -n 4 python3 main.py --action "truth solve" --set-name "training_set" --set-entry $i #> truth.txt
done
echo "You can then download the generated truth solves from the cluster, and visualize them in paraview. Please make sure that you have a recent version of paraview (>= 5.7) and open the _checkpoint.xdmf files"

echo "These are supposed to be run on a normal laptop"
python3 main.py --action "perform POD" --number-basis $N_BASIS
# python3 main.py --action "train data driven" --number-basis $N_BASIS
# #echo "Usage of the ParaView plugin requires data driven training for all intermediate basis sizes"
# #for ((n=1; n<N_BASIS; n++)); do
# #    python3 main.py --action "train data driven" --number-basis $n > out.txt
# #done
#
# echo "ONLINE STAGE"
# echo "This is supposed to be run on a normal laptop"
# python3 main.py --action "reduced solve" --mu -0.35 -0.35 --number-basis $N_BASIS
# python3 main.py --action "reduced solve" --mu 0.2 0.2 --number-basis $N_BASIS
# echo "You can also run an error analysis on a testing set, by initializing the \"testing set\" file and looping over each item in the testing set. For simplicity here I only show the computation of the error for two parameter values, not in the training set. Note that error computation requires that truth and reduced solves have been already carried out in a previous run, and just reads back in previously computed solutions."
# mpirun -n 4 python3 main.py --action "truth solve" --mu -0.35 -0.35
# python3 main.py --action "reduced error" --mu -0.35 -0.35 --number-basis $N_BASIS
# mpirun -n 4 python3 main.py --action "truth solve" --mu 0.2 0.2
# python3 main.py --action "reduced error" --mu 0.2 0.2 --number-basis $N_BASIS
