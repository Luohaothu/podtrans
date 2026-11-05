# Changelog

## 2026-11-05

- Apple MLX profiles use locally converted 8-bit `q8` weights from Fun-ASR-Nano-2512 `model.pt`. First prepare requires the approximately 2.1 GB PyTorch weights and obtains the published MLX tokenizer sidecars. Converted weights stay in the podtrans data cache. refs #4.
- MLX transcription uses a 512-token output cap and warns when a chunk reaches it without EOS. Chunking remains at 29 seconds. refs #4.
