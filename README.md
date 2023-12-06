# DuPa-ASA
 Dual-Path Attention-Based Sentiment Analysis Model

**NOTIFICATION:**

_Note : Model need interface._

*Note: The README still need a further revise and implements.* 

*Note: The test parameter need to be appointed.*

**TO-DO:**

- [x] Combine transfer study. (BERT, XLNet) 
- [ ] **Ablation Study. (Next week)**
- [ ] Need more comprehensive analysis of the model's sensitivity to different parameter settings.

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

Without transfer learning(seed88):

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
| **BLAT**    | **0.8630** | **0.9731** | **0.7861** | **0.9119** | **0.8835** |

With transfer learning(seed 88):

| Model                 | IMDB       | Yelp-2     | Yelp-5     | Amazon     | Average    |
| --------------------  | ---------- | ---------- | ---------- | ---------- | ---------- |
| bert(bz=64, S)        | 0.8721 | 0.9562 | 0.7495 | 0.8446（.4,DP) |            |
|                 |            |            |            |            |            |
|                 |            |            |            |            |            |
| xlnet(bz=64,S)        | 0.8955 | 0.9572 | 0.7541 | 0.8601 | 0.8667 |
|                 |            |            |            |            |            |
|                 |            |            |            |            |            |
| **BLAT(bert)**        | **0.8954** | **0.9744** | **0.7915** | **0.9188** | **0.8950** |
| **BLAT(xlnet)**       | **0.8874** | **0.9748** | **0.7892** | **0.9215** | **0.8932** |
|                 |            |            |            |            |            |
|                 |            |            |            |            |            |

seed:3407,bz=128, 2080ti，单卡

| Model                 | IMDB       | Yelp-2     | Yelp-5     | Amazon     | Average    |
| --------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **BLAT**              | **0.8772** | **0.9729** | **0.7861** | **0.9127** | **0.8872** |
| **BLAT(bert)**        | **0.9000** | **0.9749** | **0.7880** | **0.9212** | **0.8960** |
| **BLAT(xlnet)**       | **0.8884** | **0.9751** | **0.7891** | **0.9238** | **0.8941** |
| **BLAT(bert-large)**  | 0.8914     | 0.9759     | 0.7915     | 0.9226     | 0.8954     |
| **BLAT(xlnet-large)** | 0.8860     | 0.9761     | 0.7872     | 0.9209     | 0.8926     |

Ablation Study: seed=88, bz=128, 单卡(M40)

| Model                     | IMDB       | Yelp-2 | Yelp-5 | Amazon | Average |
| ------------------------- | ---------- | ------ | ------ | ------ | ------- |
| **Fastformer(bert_base)** | **0.8734** | **0.** | **0.** | **0.** | **0.**  |
| **TextCNN(bert_base)**    | **0.**     | **0.** | **0.** | **0.** | **0.**  |

seed:88, bz=64, 2080ti, 多卡

| Model           | IMDB | Yelp-2 | Yelp-5 | Amazon | Average |
| --------------- | ---- | ------ | ------ | ------ | ------- |
| **bert-large**  |      |        |        |        |         |
| **xlnet-large** |      |        |        |        |         |

seed:88, bz=64, 2080ti, 多卡

| Model                  | IMDB | Yelp-2 | Yelp-5 | Amazon | Average |
| -----------------------| ---- | ------ | ------ | ------ | ------- |
| **bert-base+extract**  |      |        |        |        |         |



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
