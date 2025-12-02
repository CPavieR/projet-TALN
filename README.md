# projet-TALN
```python

python3 LEFAUTEPARTAGEE.py --tsv-file corpus_genitif_tsv.tsv --output-json corpus_analysis_results.json --output-csv corpus_analysis_results.csv --delay 0.1
```
```python

python3 test.py --train-json corpus_analysis_results.json --test-json test_dataset.json --use-hypernyms --use-trt --use-sst --fusion-threshold 0.5 --no-trim-rules
```

```python
python3 test.py train-save-rules --train-json corpus_analysis_results.json --out-file rules.json
```