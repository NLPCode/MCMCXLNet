
# README for MCMCXLNet
This repository contains the implementation of the AAAI 2021 paper: "[**Show Me How To Revise: Improving Lexically Constrained Sentence Generation with XLNet**]".
****
##  Abstract
Lexically constrained sentence generation allows the incorporation of prior knowledge such as lexical constraints into the output. This technique has been applied to machine translation, and dialog response generation. Previous work usually used Markov Chain Monte Carlo (MCMC) sampling to generate lexically constrained sentences, but they randomly determined the position to be edited and the action to be taken, resulting in many invalid refinements. To overcome this challenge, we used a classifier to instruct the MCMC-based models where and how to refine the candidate sentences. First, we developed two methods to create synthetic data on which the pre-trained model is fine-tuned to obtain a reliable classifier. Next, we proposed a two-step approach, “Predict and Revise”, for constrained sentence generation. During the predict step, we leveraged the classifier to compute the learned prior for the candidate sentence. During the revise step, we resorted to MCMC sampling to revise the candidate sentence by conducting a sampled action at a sampled position drawn from the learned prior. We compared our proposed models with many strong baselines on two tasks, generating sentences with lexical constraints and text infilling. Experimental results have demonstrated that our proposed model performs much better than the previous work in terms of sentence fluency and diversity. Our code, pre-trained models and Appendix are available at
https://github.com/NLPCode/MCMCXLNet.
****
## Requirements
python 3.6  
pip install torch==1.4.0  
pip install transformers==3.0.2 
****
## Dataset
All our experiments are conducted on [One-Billion-Word](http://www.statmt.org/lm-benchmark/) corpus. We only put several sentences in the data/one-billion-words/train.txt and data/one-billion-words/test.txt. If you want to train the model from scratch, you should download the raw data from http://www.statmt.org/lm-benchmark/.
****
## Pretrained model checkpoints 
| Model           |  Download link
|----------------------|--------|
| XLNet-based token-level classifier| [\[link\]](https://drive.google.com/file/d/1wyNfE_Q7-vn9s2PCWCkN_m7RscAPQrnX/view?usp=sharing)  | 
| XLNet-based masked language model| [\[link\]](https://drive.google.com/file/d/11C6JabUpg2TQ9bCEXdnoOGUAMdUBaxgn/view?usp=sharing)  | 
| LSTM-based forward language model| [\[link\]](https://drive.google.com/file/d/1E2iye0yWxTmZwFw30h8Z7XR0A-6GaeYK/view?usp=sharing)  | 
| LSTM-based backward language model| [\[link\]](https://drive.google.com/file/d/1UPyWL9SveXBUldNITcS80UiXbfyADzkc/view?usp=sharing)  | 
| XLNet-based forward language model| [\[link\]](https://drive.google.com/file/d/1X2am3IOwfVJj2hgouRuU-igkN2ZqYbtx/view?usp=sharing)  | 
| XLNet-based backward language model| [\[link\]](https://drive.google.com/file/d/1Q6ZOl8g-p6Cne_w9hSgk1322fQmyhPi5/view?usp=sharing)  | 

Please download these checkpoints and put them into the 'checkcpoints' directory, and then decompress them with the following command:
```bash
tar -xzvf checkpoint_name.tar.gz # replace 'checkpoint_name' with the corresponding checkpoint name.
```

****
## Steps for using the proposed model:
We first use [One-Billion-Word](http://www.statmt.org/lm-benchmark/) corpus to create synthetic data, and then fine-tune XLNet (base-cased version) on them to get the token-level classifier. 
Next, we train forward and backward language models, and use them as the candidate generator. Finally, we refine the candidate sentence with the classifier and MCMC sampling. If you want to use our model to generate sentences with the given keywords with the pre-trained chechpoints, you can directly go to [Step 5](#Step5).

* Pre-processing: tokenize the raw text with XLNet (based-cased) tokenizer.   
    Make sure that the directory of the dataset (e.g., "dat/one-billion-words") is empty. Then, you should prepare some sentences to construct the training set (one sentence in each line). This file is named as "train.txt". Similarly, you should prepare some sentences to construct the validation set, which is named as "test.txt" in our propgram. You should put 'train.txt' and 'test.txt' in the correspoinding dataset directory (e.g., "dat/one-billion-words"). 
 Note: the "test.txt" is the validation set mentioned in the paper. You should prepare keywords to constrct the test set mentioned in the paper. Please refer to [Step 5](#Step5) for details. 
```bash
cd language_models   
python xlnet_maskedlm.py --convert_data 1
```
* Step 1: fine-tune XLNet on the masked lm dataset
```bash
sh xlnet_maskedlm.sh
```

* Step 2: create synthetic data for training the XLNet-based classifier
```bash
cd utils  
python create_synthetic_data.py --generate_mode 2 --batch_size 100 \  
    --train_dataset_size 1000000 --test_dataset_size 100000
```


* Step 3: train the XLNet-based classifier
```bash
cd classifier  
python -m torch.distributed.launch --nproc_per_node=3 xlnet_classifier.py\
    --gpu 0,1,2 \
```
* Step 4: train language models  
    If you want to use [L-MCMC](#L-MCMC) or [L-MCMC-C](#L-MCMC-C) to generate lexically constrained sentences, you should train the forward LSTM-based language model and the backward LSTM-based language model.
    * Train the forward LSTM-based language model
    ```bash
    cd language_models
    python lstm_lm.py --gpu 0 --dataset one-billion-words --is_forward 1
    ```
    * Train the backward LSTM-based language model
    ```bash
    cd language_models
    python lstm_lm.py --gpu 0 --dataset one-billion-words --is_forward 0
    ```  
    If you want to use [X-MCMC](#X-MCMC) or [X-MCMC-C](#X-MCMC-C) to generate lexically constrained sentences, you should train the forward XLNet-based  language model and the backward XLNet-based language model.
    * Train the forward XLNet-based language model
    ```bash
    python -m torch.distributed.launch --nproc_per_node=2 xlnet_lm.py\
        --gpu 0,1 \
        --is_forward 1  \
        --train 1
    ```
    * Train the backward XLNet-based language model
    ```bash
    python -m torch.distributed.launch --nproc_per_node=2 xlnet_lm.py\
        --gpu 0,1 \
        --is_forward 0  \
        --train 1
    ```
* <span id="Step5"> Step 5: generate sentences with lexical constraints </span>  
    We show some keywords in "inputs/one-billion-words/4keywords.txt", where each line has 4 keywords. In the following, we'll generate sentences with 4 keywords.
    If you want to generate sentences with other number of keywords, you should prepare keywords and put them in the "inputs/one-billion-words/{k}keywords.txt", where '{k}' denotes the number of keywords in each line. If so, you need to change the hyperparameter "keywords" (e.g., --keywords 1, if you want to generate sentence with one keyword). 

    * <span id="L-MCMC"> Generete with LSTM-based MCMC model (L-MCMC) </span>
    ```bash
    cd generate  
    python main.py --model_name LSTMLMGenerate --random 1 --gpu 1 --keywords 4 -sn 200
    ```

    * <span id="L-MCMC-C"> Generete with LSTM-based MCMC w/ classifier (L-MCMC-C) </span>
    ```bash
    cd generate  
    python main.py --model_name LSTMLMGenerate --random 0 --gpu 1 --keywords 4 -sn 200
    ```

    * <span id="X-MCMC"> Generete with XLNet-based MCMC model (X-MCMC) </span>
    ```bash
    cd generate  
    python main.py --model_name XLNetLMGenerate --random 0 --gpu 1 --keywords 4 -sn 200
    ```
    * <span id="X-MCMC-C"> Generete with XLNet-based MCMC w/ classifier (X-MCMC-C) </span>
    ```bash
    cd generate  
    python main.py --model_name XLNetLMGenerate --random 0 --gpu 1 --keywords 4 -sn 200
    ```
****
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

