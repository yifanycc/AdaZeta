for K in 16 512; do
    # Generate k-shot splits for seeds 13,21,42,87,100 with a maximum of 1k test examples in data/k-shot-1k-test,
    # where k is the number of training/validation examples per label
#    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K --output_dir /global/cfs/cdirs/m4645/yifanycc/data
    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K
done