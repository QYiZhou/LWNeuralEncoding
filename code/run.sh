python3.7 ./feature_extraction/generate_features.py \
-fname='alexnet' \
-dim_rd='srp' \
-ad='feature_zoo' \
-gpu='0' \
-rp='./'

cd ./code

for roi in 'V1' 'V2' 'V3' 'V4' 'LOC' 'FFA' 'STS' 'EBA' 'PPA'
do
for wc in 1e0 1e1 1e2
do
for lay in 1e-1 1e0 1e1
do
python3.7 ./perform_encoding.py \
-gpu='0' \
-ad='feature_zoo' \
-rd='results' \
-rp='../' \
-dim_rd='srp' \
-e=500 \
-roi=$roi \
-sub=1 \
-m='hyper_tune' \
-model='alexnet' \
-wc=$wc \
-lay=$lay \
-fn=2 \
-pat=30 \
-delta=1e-3 \
-mtc='pcc' \
-cp='summary_pcc_'
done
done

python3.7 ./perform_encoding.py \
-gpu='0' \
-ad='feature_zoo' \
-rd='results' \
-rp='../' \
-dim_rd='srp' \
-e=500 \
-roi='V1' \
-sub=10 \
-m='train' \
-model='alexnet'  \
-wc=0 \
-lay=0 \
-fn=2 \
-pat=30 \
-delta=1e-3 \
-mtc='pcc' \
-cp='summary_pcc_'
done

python3.7 aver_from_csv.py -cp='summary_pcc_' \
-model='alexnet' \
-rp='../' \
-dim_rd='srp' \
-rd='results'