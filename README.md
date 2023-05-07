## TransFetch Knowledge Distillation

### Dependencies
- Python 3.8
- CUDA 11.8+
- Pytorch 2.0.0

### Running Program
- Import conda environment with `conda env create -n transfetch-env --file transfetch-env.yml`
- Change `LoadTraces_ROOT`, `OUTPUT_ROOT`, and `GPU_ID` in `run.sh`
- Set `MODEL` in `run.sh` to either "d", "r", or "v" indicating DenseNet, ResNet, or ViT
- `./run.sh`
- Export **cross-platform** conda environment if packages changed with `conda env export --from-history>transfetch-env.yml`

### Notes
- Hard coded dimension or `DIM` parameter for ResNet as 64, ResNet_tiny as 4

### Potential Bugs
- Early stop is 10 for both the training of teacher and student
