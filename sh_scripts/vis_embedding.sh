python scripts/visualize_embedding.py \
    outputs/dino3_vitb16_mydata_margin_1000_moremore/checkpoints/999.layer.pth \
    --testdata-root testdata \
    --output outputs/dino3_vitb16_mydata_margin_1000_moremore/embedding \
    --per-cat-anom 200 --per-cat-norm 200 \
    --layer-idx 0