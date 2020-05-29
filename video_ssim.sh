if [ "$#" -ne 4 ]; then
	echo "Expected ./video_ssim.sh <ref path> <image path to create> <h264 bitrate> <frame rate>"
	exit
fi

mkdir -p "$2"
/usr/bin/ffmpeg -r $4 -f image2 -i "$1/0000000%03d.png" -vcodec libx264 -pix_fmt rgb8 -b:v $3M "$2/video.mp4"
/usr/bin/ffmpeg -i "$2/video.mp4" -start_number 0 "$2/0000000%03d.png"
# echo SSIM: $(python video_ssim.py "$1/" "$2/")
