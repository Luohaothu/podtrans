# Changelog

## 2026-11-05

- Fresh MLX q8 preparation downloads only small tokenizer and config files from the published w4a16 package, about 4.5 MB in the inspected file set. Hub allowlists exclude catalog weight shards and indexes. Obtain the approximately 2.1 GB PyTorch `model.pt` first if it is absent. refs #4.
- Apple MLX profiles use locally converted 8-bit `q8` weights from Fun-ASR-Nano-2512 `model.pt`. First prepare requires the approximately 2.1 GB PyTorch weights and obtains the published MLX tokenizer sidecars. Converted weights stay in the podtrans data cache. refs #4.
- MLX transcription uses a 512-token output cap and warns when a chunk reaches it without EOS. Chunking remains at 29 seconds. refs #4.
