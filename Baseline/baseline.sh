#!/bin/bash
echo "Starting Upper Lower Bounds Experiments"
echo "Running BackBoneNet..."
python BackBoneNet.py
echo "Running FineTuneClassifier..."
python FineTuneClassifier.py
echo "Running LowerBound..."
python LowerBound.py
echo "Finished!!"