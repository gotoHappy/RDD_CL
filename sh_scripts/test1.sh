python scripts/diagnose.py outputs/dino3_vitb16_mydata_margin_1000_moremore/checkpoints/999.layer.pth \
  --testdata-root testdata \
  --output outputs/dino3_vitb16_mydata_margin_1000_moremore/eval_1000_wo10 \
  --use-layers 5 6 7 8 9 10
