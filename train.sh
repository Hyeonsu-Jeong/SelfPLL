SEED=(0 1 2)
for seed in "${SEED[@]}"; do
    for dist in {0..4}; do
        python train_linear.py "$@" --dist $dist --manualSeed $seed --corruption 0.9 & 
        python train_linear.py "$@" --dist $dist --manualSeed $seed --corruption 0.8 & 
        python train_linear.py "$@" --dist $dist --manualSeed $seed --corruption 0.7 & 
        python train_linear.py "$@" --dist $dist --manualSeed $seed --corruption 0.6 
    done

    python train_linear.py "$@" --partial True --manualSeed $seed --corruption 0.9 &  
    python train_linear.py "$@" --partial True --manualSeed $seed --corruption 0.8 &  
    python train_linear.py "$@" --partial True --manualSeed $seed --corruption 0.7 &
    python train_linear.py "$@" --partial True --manualSeed $seed --corruption 0.6

    for dist in {0..4}; do
        python train_linear.py "$@" --dist $dist --manualSeed $seed --corruption 0.5 & 
        python train_linear.py "$@" --dist $dist --manualSeed $seed --corruption 0.3 & 
        python train_linear.py "$@" --dist $dist --manualSeed $seed --corruption 0.1 & 
        python train_linear.py "$@" --dist $dist --manualSeed $seed --corruption 0.0 
    done

    python train_linear.py "$@" --partial True --manualSeed $seed --corruption 0.5 &  
    python train_linear.py "$@" --partial True --manualSeed $seed --corruption 0.3 &  
    python train_linear.py "$@" --partial True --manualSeed $seed --corruption 0.1 &
    python train_linear.py "$@" --partial True --manualSeed $seed --corruption 0.0 
done