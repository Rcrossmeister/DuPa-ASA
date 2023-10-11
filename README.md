# DuPa-ASA
 Dual-Path Attention-Based Sentiment Analysis Model

**NOTIFICATION:**

_Note : Model need interface._

*Note: The README still need a further revise and implements.* 

*Note: The test parameter need to be appointed.*

__Update:__

* *A xxx is released. 09-015-2023*

**TO-DO:**

1. Combine transfer study. (BERT, XLNet)
2. Ablation Study.
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

| Model       | IMDB       | Yelp-2     | Yelp-5     | Amazon     |
| ----------- | ---------- | ---------- | ---------- | ---------- |
| BiLSTM      | 0.5098     | 0.7081     | 0.6316     | 0.5061     |
| TextCNN     | 0.8702     | 0.9681     |            | 0.8947     |
| TextCNN_Att | 0.8644     | 0.9639     | 0.7137     |            |
| TextRCNN    | 0.7480     | 0.9267     | 0.7186     | 0.8377     |
| Transformer | 0.7746     | 0.9344     | 0.7110     | 0.8093     |
| DPCNN       |            |            |            |            |
| Fasttext    |            |            |            |            |
| Fastformer  |            |            |            |            |
| **BLAT**    | **0.8630** | **0.9731** | **0.7861** | **0.9119** |

acc(BLAT with bert on IMDB)=0.8952

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
