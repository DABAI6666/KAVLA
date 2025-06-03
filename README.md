
## :mag: Environment
- Pytorch `2.0.1`
```shell
conda env create -n KAVLA python=3.8
conda activate KAVLA
pip install -r requirement.txt
```
## :racehorse: Train

### Pretraining Visual Knowledge Aligner

<!-- After you have successfully downloaded the Wikipedia files and placed them in the appropriate path, you could use the following code to perform VKA pretraining. -->
Before you start the pretraining for the visual knowledge aligner, you should place the downloaded `Wikipedia-Knowledge-2M` dataset in LLaVA/playground/knowledge_data directory.

Then you can use the following scripts for pretraining.

```shell
cd Code
export PYTHONPATH=path_to_current_dir
```




#### Training VKA with LLMs 

Replace `pretrain_opt_adapter` with the save path of your pretrained EKA.

``` shell
bash scripts/decoder_model/pretrain_knowledge.sh
```


#### Fine-tune VKA on the Knowledge Data with LLM 
Change the attribute `pretrain_knowledge_params_path` to the path where the parameters extracted in the previous stage are stored.

``` shell
bash scripts/knowledge/pretrain.sh
```



#### Fine-tune VKA and FKA

Finally, we used a two-stage training method when fine-tuning FKA.

``` shell

bash scripts/knowledge_rd/rumor_detection_ft.sh
```


#### Evalution
``` shell

bash scripts/knowledge_rd/eval/rumor_detection_test.sh
```



```




