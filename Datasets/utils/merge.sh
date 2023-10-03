#!/bin/bash


# A utility script to merge root files.
# Requires configuration of the BASE_DIR and SUBDIRS variables.


# Base directory
BASE_DIR="1l"

# List of subdirectories
SUBDIRS=("5j3b_tt1b" "5j3b_ttB" "5j3b_ttH" "5j3b_ttbb" "5j3b_ttc" "5j3b_ttlight")

# List of unique ROOT file names to be combined
FILES=("ttH_PP8_mc16a_AFII.root" "ttH_PP8_mc16d_AFII.root" "ttH_PP8_mc16e_AFII.root")

# Loop over each unique file name
for file in "${FILES[@]}"; do
    # Create an array to store input files for the current file name
    INPUT_FILES=()
    for subdir in "${SUBDIRS[@]}"; do
        # Check if the file exists in the subdirectory
        if [[ -f "${BASE_DIR}/${subdir}/${file}" ]]; then
            INPUT_FILES+=("${BASE_DIR}/${subdir}/${file}")
        fi
    done
    # Use hadd to combine the files
    hadd "combined_${file}" "${INPUT_FILES[@]}"
    echo "Combining files for ${file}..."
done