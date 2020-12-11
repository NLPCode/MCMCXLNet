# MCMCXLNet
This repository contains the implementation of the AAAI 2021 paper: "[**Show Me How To Revise: Improving Lexically Constrained Sentence Generation with XLNet**]".
##  Abstract
Constrained sentence generation can incorporate prior knowledge such as lexical constraints into the output. This technique has been applied to machine translation, and dialog response generation. Previous Markov Chain Monte Carlo (MCMC)-based models attempted to revise candidate sentences randomly and arbitrarily at a position drawn from a uniform distribution, resulting in many invalid operations. To overcome this challenge, we used a classifier to instruct the MCMC-based models where and how to refine the candidate sentences. In this paper, we proposed two methods to create synthetic data on which the pre-trained model is fine-tuned to obtain a reliable classifier. Next, we proposed a two-step approach, “Predict and Revise”, for constrained sentence generation. During the predict step, we leveraged the classifier to compute the learned prior for the MCMC-based models. During the revise step, we used theMCMC sampling to revise the candidate sentence, using a sampled action taken at a sampled position drawn from the learned prior for revision. We compared our proposed models with many strong baselines on two tasks, generating sentences with lexical constraints and text infilling. Experimental results have demonstrated that our proposed model performs much better than the previous work in terms of sentence fluency and diversity.
## Requirements
python 3.6  
pip install torch==1.4.0  
pip install transformers==3.0.2 
## Dataset
All our experiments are conducted on One-Billion-Word corpus (http://www.statmt.org/lm-benchmark/). We only put several sentences in the data/one-billion-words/train.txt and data/one-billion-words/test.txt. If you want to train the model from scratch, you should download the raw data from http://www.statmt.org/lm-benchmark/.

## Pretrained model checkpoints 
| Model           |  Download link
|----------------------|--------|
| XLNet-based token-level classifier| [\[link\]](https://drive.google.com/file/d/1jce_rjHM4GG4S5AFXx4W7ntPt4XxunKR/view?usp=sharing)  | 
| XLNet-based masked language model| [\[link\]](https://drive.google.com/file/d/11C6JabUpg2TQ9bCEXdnoOGUAMdUBaxgn/view?usp=sharing)  | 
| LSTM-based forward language model| [\[link\]](https://drive.google.com/file/d/1E2iye0yWxTmZwFw30h8Z7XR0A-6GaeYK/view?usp=sharing)  | 
| LSTM-based backward language model| [\[link\]](https://drive.google.com/file/d/1UPyWL9SveXBUldNITcS80UiXbfyADzkc/view?usp=sharing)  | 
| XLNet-based forward language model| [\[link\]](https://drive.google.com/file/d/1X2am3IOwfVJj2hgouRuU-igkN2ZqYbtx/view?usp=sharing)  | 
| XLNet-based backward language model| [\[link\]](https://drive.google.com/file/d/1Q6ZOl8g-p6Cne_w9hSgk1322fQmyhPi5/view?usp=sharing)  | 

Please download these checkpoints and put them into the 'checkcpoints' directory, and then decompress them with the following command:
```bash
tar -xzvf checkpoint_name.tar.gz # replace 'checkpoint_name' with the corresponding checkpoint name.
```


## Steps for using the proposed model:
We first use [One-Billion-Word](http://www.statmt.org/lm-benchmark/) corpus to create synthetic data, and then fine-tune XLNet (base-cased version) on them to get the token-level classifier. 
Next, we train forward and backward language models, and use them as the candidate generator. Finally, we refine the candidate sentence with the classifier and MCMC sampling.   If you want to use our model with the pre-trained chechpoints, you can directly go to step 4.

### Preprocessing: tokenize the raw text with XLNet (based-cased) tokenizer
```bash
cd language_models   
python xlnet_maskedlm.py --convert_data 1
```

### Step 1: create synthetic data for training the XLNet-based classifier
```bash
cd utils  
python create_synthetic_data.py --generate_mode 2
```


### Step 2: train the XLNet-based classifier
```bash
cd classifier  
python -m torch.distributed.launch --nproc_per_node=3 xlnet_classifier.py\
    --gpu 0,1,2 \
```
### Step 3: train language models

#### Train the forward LSTM-based language model
```bash
cd language_models
python lstm_lm.py --gpu 0 --dataset one-billion-words --is_forward 1
```
#### Train the backward LSTM-based language model
```bash
cd language_models
python lstm_lm.py --gpu 0 --dataset one-billion-words --is_forward 1
```

#### Train the forward XLNet-based language model
```bash
python -m torch.distributed.launch --nproc_per_node=2 xlnet_lm.py\
    --gpu 0,1 \
    --is_forward 1  \
    --train 1
```
#### Train the backward XLNet-based language model
```bash
python -m torch.distributed.launch --nproc_per_node=2 xlnet_lm.py\
    --gpu 0,1 \
    --is_forward 0  \
    --train 1
```
### Step 4: generate sentences with lexical constraints

#### Generete with LSTM-based MCMC model (L-MCMC)
```bash
cd generate  
python main.py --model_name LSTMLMGenerate --random 1 --gpu 1 --keywords 4 -sn 200
```

#### Generete with LSTM-based MCMC + classifier (L-MCMC-C)
```bash
cd generate  
python main.py --model_name LSTMLMGenerate --random 0 --gpu 1 --keywords 4 -sn 200
```

#### Generete with XLNet-based MCMC model (X-MCMC)
```bash
cd generate  
python main.py --model_name XLNetLMGenerate --random 0 --gpu 1 --keywords 4 -sn 200
```
#### Generete with XLNet-based MCMC + classifier (X-MCMC-C)
```bash
cd generate  
python main.py --model_name XLNetLMGenerate --random 0 --gpu 1 --keywords 4 -sn 200
```
## Citation
If you want to use this code in your research, you can cite our [paper](link):
```bash
@inproceedings{he2021xlentmcmc,
  title={Show Me How To Revise: Improving Lexically Constrained Sentence Generation with XLNet},
  author={He, Xingwei and Li, Victor O.K.},
  booktitle={Proceedings of AAAI},
  year={2021}
}
```

