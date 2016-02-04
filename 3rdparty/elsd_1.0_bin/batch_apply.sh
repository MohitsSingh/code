#!/bin/bash
BASEDIR="/home/amirro/data/Stanford40/JPEGImages"
OUTDIR="/home/amirro/storage/s40_elsd_output"
echo $BASEDIR
for f in $BASEDIR/*.jpg
do
	echo $f
	pgmFile=$(echo $f | sed "s;$BASEDIR;$OUTDIR;g" | sed "s;.jpg;.pgm;g")
	resFile=$(echo $f | sed "s;$BASEDIR;$OUTDIR;g" | sed "s;.jpg;.txt;g")
#	if [ ! -e $resFile ]
#	then
		echo $f | sed "s;$BASEDIR;$OUTDIR;g" | sed "s;.jpg;.pgm;g"
		convert ${f} $pgmFile	
		elsd $pgmFile $resFile
		rm $pgmFile
#	fi
done


