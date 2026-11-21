# Results on the CLINC150 test set (n = 5500: 4500 in-scope, 1000 out-of-scope)

Latency: cleaning + model, mean over 200 single messages after 10 warm-up runs.

Hardware: arm | 8 logical cores | torch threads = 4 | Darwin 25.2.0 | CPU only

| Model                                   |   In-scope accuracy |   OOS recall |   Routing accuracy |   Size (MB) |   Latency mean (ms) |   Latency p95 (ms) |   Threshold |
|:----------------------------------------|--------------------:|-------------:|-------------------:|------------:|--------------------:|-------------------:|------------:|
| tf-idf + LogReg                         |              0.8258 |        0.92  |             0.8536 |       48.1  |                9.74 |               9.88 |        0.58 |
| fastText + FFN                          |              0.7633 |        0.871 |             0.8016 |       24.1  |                0.06 |               0.09 |        0.84 |
| BERT-base FP32 (teacher)                |              0.914  |        0.89  |             0.9156 |      438.42 |               34.94 |              36.32 |        0.9  |
| BERT-base INT8                          |              0.9224 |        0.854 |             0.9184 |      181.61 |               12.74 |              13.92 |        0.82 |
| BERT-small FP32 (distilled)             |              0.8889 |        0.884 |             0.8958 |      115.37 |                5.72 |               6.19 |        0.9  |
| BERT-small INT8 (distilled + quantized) |              0.89   |        0.89  |             0.898  |       76.65 |                2.97 |               3.19 |        0.9  |
