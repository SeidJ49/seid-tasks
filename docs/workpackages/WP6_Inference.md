# Work Package 6: Inference & Deployment

## Objective
Implement efficient inference pipeline and prepare model for deployment.

## Main Goals
- Create lightweight inference without Teacher model dependency
- Optimize inference speed and memory usage
- Support real-time deployment scenarios
- Provide evaluation tools for thesis validation

## Key Tasks

### 6.1 Inference Script Enhancement
- **File**: Extend/create `opencood/tools/inference.py` variant
- Support Lite-BEV model loading and inference
- Handle multi-frame temporal fusion if applicable
- Efficient memory management for inference

### 6.2 Model Export & Quantization
- Export trained model to production-ready format
- Optional quantization for edge deployment
- Support ONNX export for cross-platform compatibility
- Verification of numerical consistency after export

### 6.3 Real-time Inference Testing
- Benchmark inference speed on target hardware
- Memory profiling during inference
- Latency analysis per component
- FPS measurements for streaming scenarios

### 6.4 Evaluation Pipeline
- Support evaluation on multiple weather conditions
- Compute metrics: AP, mAP, latency
- Per-scenario performance breakdown
- Robustness metrics under adverse conditions

### 6.5 Deployment Documentation
- Inference API documentation
- Model loading and configuration guide
- Example inference scripts for users
- Performance expectations and hardware requirements

## Deliverables
- [ ] Production-ready inference script
- [ ] Model export utilities (ONNX, TorchScript)
- [ ] Inference performance benchmarks
- [ ] Real-time evaluation framework
- [ ] Deployment guide and examples

## Success Criteria
- Inference runs without Teacher model
- Inference speed meets real-time requirements (>10 Hz)
- Memory footprint suitable for embedded systems
- Evaluation metrics consistent with training
- Clear documentation for end users
