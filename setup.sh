#!/bin/bash
# Create the dummy database for training, evaluation and test. 
# may need execution permission by doing `chmod +x setup.sh`

directories=(
    "data/train/clean"
    "data/train/noisy"
    "data/val/clean"
    "data/val/noisy"
    "data/test/clean"
    "data/test/noisy"
)

for directory in "${directories[@]}"; do
    if [ -d "$directory" ]; then
        echo "Directory $directory already exists."
	find "$directory" -type f -name "*.wav" -exec rm -f {} +
	echo "Deleted all .wav files in $directory and subdirectories"
    else
        mkdir -p "$directory"
        if [ $? -eq 0 ]; then
            echo "Directory $directory created successfully."
        else
            echo "Could not create directory $directory."
        fi
    fi
done

# Nsamples sr fmin fmax noise_ratio split_ratio
python scripts/build_tone_dataset.py 10024 65792 16000 20 6000 0.01 0.7


mkdir runs

