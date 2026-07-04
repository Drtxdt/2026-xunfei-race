# RKNN PPOCR assets

Place Rockchip RKNN Model Zoo PPOCR assets here:

- `ppocrv4_det.rknn`
- `ppocrv4_rec.rknn`
- `ppocr_keys_v1.txt`

Recommended source:

- `https://github.com/airockchip/rknn_model_zoo/tree/main/examples/PPOCR`

For the factory sign task, `recognition_mode:=ppocr_rknn_rec_only` only requires
`ppocrv4_rec.rknn` and `ppocr_keys_v1.txt`.
