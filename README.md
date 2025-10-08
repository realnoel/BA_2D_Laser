# Using val.py

## basic
python3 src/val.py --ckpt checkpoints/04102025_181230/best.pt

## override in_channels or modes if your ckpt config didn’t store them
python3 src/val.py --ckpt checkpoints/04102025_181230/best.pt --in-channels 4 --modes1 24 --modes2 24 --width 16 --pad 8

## save more preview images n for n pictures -1 for all pictures
python3 src/val.py --ckpt checkpoints/04102025_181230/last.pt --save-images 5

## Rollout
python3 src/rollout.py --ckpt checkpoints/<checkpoint_id>/best.pt --steps 10


## Get animation or rendering

if epoch % freq_generate == 0:
                    idx_to_print = 4
                    for i in range(B):
                        sid = subset_indices[gid] if subset_indices is not None else gid
                        gid += 1
                        # Gegenüberstellung:
                        animate_side_by_side_mp4(
                            y_pred[i:i+1], y[i:i+1],
                            out_path=f"results/{timestamp}/anim_compare_id{sid:06d}_ep{epoch:04d}.mp4",
                            fps=8
                        )
                        # This is working great for returning a sequence of y with idx
                        # Wichtig: pro Sample speichern (B auf 1 reduzieren),
                        # damit nichts “vermischt” wird:
                        save_temperature_plot_sequence(
                            y_pred[i:i+1], idx_to_print,
                            path=f"results/{timestamp}",
                            name_prefix=f"seq_prediction_id{sid:06d}",
                            epoch=epoch
                        )
                        save_temperature_plot_sequence(
                            y[i:i+1], idx_to_print, 
                            path=f"results/{timestamp}",
                            name_prefix=f"seq_groundtruth_id{sid:06d}",
                            epoch=epoch
                        )