!/bin/bash -l

f=../datasets
echo "Run 大作业"
python ../src/run_scMDC.py --n_clusters 19 --ae_weight_file AE_weights_pbmc10k.pth.tar --data_file $f --save_dir atac_pbmc10k/  --embedding_file --prediction_file -el 256 128 64 -dl1 64 128 256 -dl2 64 128 256 --tau .1 --phi1 0.005 --phi2 0.005 \
    --ae_weights  ./atac_pbmc10k/2_AE_weights_pbmc10k.pth.tar