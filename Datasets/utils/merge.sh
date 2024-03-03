#!/bin/bash

#==================#
# Merge ROOT files #
#==================#

# """
# This script merges multiple ROOT files located in subdirectories of a base directory.
#
# Usage:
#     ./merge.sh -b <basedir> [options]
#     /merge.sh -h (for help running the script)
#     for -w cmd line arg -> e.g ttH*.root
#      SUBDIRS are hard-configured in the script :/
# """

BASE_DIR=""
OUTPUT_DIR="."
PREFIX="combined_"
VERBOSE=false
WILDCARD="*.root"

# Display usage information
usage() {
    echo "Usage: $0 -b BASE_DIR -w WILDCARD [options]"
    echo "Options:"
    echo "  -b  Base directory where subdirectories are located"
    echo "  -o  Output directory for merged files (default: current directory)"
    echo "  -p  Custom prefix for output files (default: 'combined_')"
    echo "  -v  Enable verbose mode"
    echo "  -w  Wildcard pattern for input files (default: '*.root')"
    echo "  -h  Display this help and exit"
}

# Parse the command line arguments using getopts
while getopts ":b:o:p:w:vh" opt; do
    case ${opt} in
        b ) BASE_DIR=$OPTARG ;;
        o ) OUTPUT_DIR=$OPTARG ;;
        p ) PREFIX=$OPTARG ;;
        w ) WILDCARD=$OPTARG ;;
        v ) VERBOSE=true ;;
        h ) usage; exit 0 ;;
        \? ) echo "Invalid option: $OPTARG" 1>&2; usage; exit 1 ;;
        : ) echo "Invalid option: $OPTARG requires an argument" 1>&2; usage; exit 1 ;;
    esac
done
shift $((OPTIND -1))


# Check if the base directory is provided
if [[ -z "$BASE_DIR" ]]; then
    echo "Base directory (-b) is required."
    usage
    exit 1
fi

# Check if hadd is available, throw an error if not
if ! command -v hadd &> /dev/null; then
    echo "Error: hadd could not be found"
    exit 1
fi

# List of subdirectories to search for files (think of better way to supply these...)
SUBDIRS=("5j3b_discriminant_ttb" "5j3b_discriminant_ttbb" "5j3b_discriminant_ttB" "5j3b_discriminant_ttH" "5j3b_discriminant_ttc" "5j3b_discriminant_ttlight")

# Find all ROOT files in the base directory
FILES=$(find "${BASE_DIR}" -type f -name "${WILDCARD:-*.root}" | sed -E 's|.*/||' | sort -u)

# Create the output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Now, iterate over each file and merge the corresponding files from the subdirectories
for file in ${FILES}; do
    INPUT_FILES=()
    for subdir in "${SUBDIRS[@]}"; do
        if [[ -f "${BASE_DIR}/${subdir}/${file}" ]]; then
            INPUT_FILES+=("${BASE_DIR}/${subdir}/${file}")
        fi
    done
    
    if [[ ${#INPUT_FILES[@]} -gt 0 ]]; then
        OUTPUT_FILE="${OUTPUT_DIR}/${PREFIX}${file}"
        if $VERBOSE; then echo "Combining files for ${file} into ${OUTPUT_FILE}..."; fi
        hadd "${OUTPUT_FILE}" "${INPUT_FILES[@]}"
        if [[ $? -eq 0 ]]; then
            if $VERBOSE; then echo "Successfully combined ${file}"; fi
        else
            echo "Error combining ${file}"
        fi
    else
        if $VERBOSE; then echo "No files found to combine for ${file}"; fi
    fi
done

if $VERBOSE; then echo "All operations completed."; fi