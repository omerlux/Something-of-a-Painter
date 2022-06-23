mkdir -p dataset/ && \

python3 utils/download_data.py && \

unzip dataset/datasets.zip -d dataset/ && \

mv dataset/all_six_datasets/* dataset && \

rm -r dataset/all_six_datasets dataset/__MACOSX
