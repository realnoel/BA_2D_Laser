# Using val.py

## basic
python3 src/val.py --ckpt checkpoints/04102025_181230/best.pt

## override in_channels or modes if your ckpt config didn’t store them
python3 src/val.py --ckpt checkpoints/04102025_181230/best.pt --in-channels 4 --modes1 24 --modes2 24 --width 16 --pad 8

## save more preview images n for n pictures -1 for all pictures
python3 src/val.py --ckpt checkpoints/04102025_181230/last.pt --save-images 5