import argparse

def main():
    parser = argparse.ArgumentParser(
        description = 'Extracting crops from video'
    )
    parser.add_argument('--video-dir', help="Root directory") # .video_dir
    #parser.add_argument('--crops-dir', help="Crops directory") # .crops_dir

    args = parser.parse_args()
    #os.makedirs(os.path.join(args.root_dir, args.crops_dir), exist_ok=True)

if __name__ == '__main__':
    main()
