for s in 0 1 2 3 4
do
    python run.py --config config_java.yaml --test_baseline_metrics --seed $s --gpu 0
done