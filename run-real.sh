cd ~/dev/instant-ngp/data/nerf/real/ && \
../../../scripts/colmap2nerf.py --video_in video.mov --video_fps 2 --out transform.json --run_colmap && \
cd ~/dev/instant-ngp && \
CUDA_VISIBLE_DEVICES=0 ./build/testbed --scene ./data/nerf/real/transform.json

