# DuPa-ASA
 Dual-Path Attention-Based Sentiment Analysis Model

**NOTIFICATION:**

_Note : Model need interface._

*Note: The README still need a further revise and implements.* 

*Note: The test parameter need to be appointed.*

**TO-DO:**

1. Combine transfer study. (BERT, XLNet)
2. **Ablation Study. (Next week)**
3. Need more comprehensive analysis of the model's sensitivity to different parameter settings.

## DataSet

IMDB

Yelp

Amazon

## Use

* To **train** the  baseline model :

```shell
python run.py --model=BLAT --embedding=random --word True
```

## Results

Without bert:

| Model       | IMDB       | Yelp-2     | Yelp-5     | Amazon     | Average |
| ----------- | ---------- | ---------- | ---------- | ---------- | ------- |
| BiLSTM      | 0.5098     | 0.7081     | 0.6316     | 0.5061     | 0.5889  |
| TextCNN     | 0.8702     | 0.9681     | 0.7735     | 0.8947     | 0.87663 |
| TextCNN_Att | 0.8644     | 0.9639     | 0.7137     | 0.8756     | 0.8544  |
| TextRCNN    | 0.7480     | 0.9267     | 0.7186     | 0.8377     | 0.8078  |
| Transformer | 0.7746     | 0.9344     | 0.7110     | 0.8093     | 0.8073  |
| DPCNN       | 0.8742     | 0.9723     | 0.7796     | 0.9058     | 0.8830  |
| Fasttext    | 0.8766     | 0.9534     | 0.7380     | 0.8517     | 0.8549  |
| Fastformer  | 0.8718     | 0.9664     | 0.7739     | 0.8813     | 0.8734  |
| **BLAT**    | **0.8630** | **0.9731** | **0.7861** | **0.9119** | 0.8835  |
| **BLAT(bert)** | **0.8952** | ****   | ****   | ****   |         |
| **BLAT(xlnet)** | **0.8952** | ****   | ****   | ****   |         |



## Citation

Please cite our paper if you use it in your work:

```shell
@inproceedings{,
   title={{}: },
   author={},
   booktitle={},
   year={}
}
```
