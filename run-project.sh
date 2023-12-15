#!/bin/bash

# Prompt the user to enter a number
echo "Choose a photo between 1 and 5:"
read photo_number

# Construct the file name based on the input
# Assuming all images follow the format '2_<number>.jpg'
file_name="2_5${photo_number}.jpg"

# Check if the file exists
if [[ -f $file_name ]]; then
    # Run the Python script with the specified image
    python3 Otsu_Niblack.py $file_name
else
    echo "File does not exist: $file_name"
fi