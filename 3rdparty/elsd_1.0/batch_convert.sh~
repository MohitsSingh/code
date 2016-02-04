#!/bin/bash
BASEDIR="/home/amirro/storage/s40_elsd_output"
OUTDIR="/home/amirro/storage/s40_elsd_output/converted"
mkdir $OUTDIR 
echo $BASEDIR
for f in $BASEDIR/drinking*svg
do
	echo $f
	resFile=$(echo $f | sed "s;$BASEDIR;$OUTDIR;g" | sed "s;.pgm.svg;.png;g")
	if [ ! -e $resFile ]
	then		
		convert ${f} $resFile
	fi
done


