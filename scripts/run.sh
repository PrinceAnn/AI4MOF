python model.py --seed 123456
python active_learning.py --seed 12345678

python find_closest_samples.py \
  --all_data ../data/all_data_cleaned.csv \
  --subset ../data/low_intensity_filtered.csv \
  --selected ../Results_AL/Round1/selected_samples.csv \
  --output ../Results_AL/Round1/closest_samples.csv \
  --metric cosine