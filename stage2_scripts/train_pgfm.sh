current_time=$(date +"%Y-%m-%d-%H-%M")

lake_ids=$(cut -d',' -f1 ../utils/intersection_ids.csv | tail -n +2) 

for lake_id in $lake_ids; do
    echo "Processing lake_id: ${lake_id} with current_time: ${current_time}"
    python3 main_new.py --model_type fm-pg --strategy n+1 --pt_datetime 2024-07-26-16-12 --gpu 0 --label_name obs_temp --seed 40 --lake_id ${lake_id} --current_time ${current_time}
done