#!/system/bin/sh

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: json_2_mask.sh
#
#   @Author: Shun Li
#
#   @Date: 2021-10-14
#
#   @Email: 2015097272@qq.com
#
#   @Description: use labelme_json_to_dataset to generate the labeled image
#
#------------------------------------------------------------------------------

rm -rf label_images/

mkdir label_images/

# use the bash in in th path containing json_images/xxxx.json
cd json_images/

# delete the folders in json_images/
direc=$(pwd)
for dir2del in $direc/* ; do
 if [ -d $dir2del ]; then
  rm -rf $dir2del 
  echo "deleting $dir2del!"
 fi
done

# get names of *.json
files=$(ls)

# generate and copy to label_images/
for file_name in $files
do
    echo "find $file_name"
    labelme_json_to_dataset $file_name

    cp -r ${file_name%%.*}"_json"/img.png ../images/${file_name%%.*}".png"
    echo "copying ${file_name%%.*}.png to ../images/"
    cp -r ${file_name%%.*}"_json"/label.png ../label_images/${file_name%%.*}".png"
    echo "copying ${file_name%%.*}.png to ../label_images/"

done
