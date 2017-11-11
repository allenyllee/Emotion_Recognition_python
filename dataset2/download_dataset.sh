#!/bin/bash

# download fetch_gdrive_file.sh script
# wget/curl large file from google drive - Stack Overflow 
# https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
# 
# allows you do non-interactively download large public files from gdrive 
# https://gist.github.com/allenyllee/ac42730e9c756e0aeb9b2f07424b68dd

# How do I get the raw version of a gist from github? - Stack Overflow 
# https://stackoverflow.com/questions/16589511/how-do-i-get-the-raw-version-of-a-gist-from-github
# 
# If you want the last version of a Gist document, just remove the <commit>/ from URL
#
wget https://gist.githubusercontent.com/allenyllee/ac42730e9c756e0aeb9b2f07424b68dd/raw/fetch_gdrive_file.sh -O fetch_gdrive_file.sh

URL=$1

# if file KDEF.7z doesn't exist, then download it
if ! [ -e KDEF.7z ]; then
    # download KDEF.7z
    ./fetch_gdrive_file.sh $URL KDEF.7z
else
    echo "remove KDEF.7z..."
    echo "please execute this script again"
    rm -rf KDEF.7z
fi

# if folder KDEF doesn't exist, then extract it
if ! [ -d KDEF ]; then
    # un7zip
    7z x KDEF.7z
else
    echo "remove KDEF..."
    echo "please execute this script again"
    rm -rf KDEF.7z
fi