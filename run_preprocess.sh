DATA_ROOT=$1 CROPS_ROOT=$2 OUT=$3
echo "Extracting bounding boxes from original videos"
python preprocessing/finding_face_region_using_real.py --video-dir $DATA_ROOT

echo "Extracting crops as pngs"
python preprocessing/extract_crops.py --video-dir $DATA_ROOT --crops-dir $CROPS_ROOT

echo "Extracting landmarks"
python preprocessing/generate_landmarks.py --video-dir $DATA_ROOT --crops-dir $CROPS_ROOT

echo "Extracting SSIM masks"
python preprocessing/generate_diffs.py --video-dir $DATA_ROOT --crops-dir $CROPS_ROOT

echo "Generate folds"
python preprocessing/generate_folds.py --root-dir $DATA_ROOT --crops-dir $CROPS_ROOT --out $OUT
