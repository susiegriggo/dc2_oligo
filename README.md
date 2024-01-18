# DC2_oligo
## **Accurate prediction of the oligomerization state of coiled-coil domains based on the representation of monomeric structures obtained with Colabfold**
![image](https://github.com/Rmadeye/dc2_oligo/assets/46814304/3cd48168-b572-489d-b8b2-5cd659182242)

DC2_oligo is a program for predicting the oligomerization state of homooligomeric coiled-coil domains. It is based on a logistic regression model trained on the 217 embeddings generated by the AlphaFold2 multimer_v3 model for a single sequence.

The input for training is a data frame `/tests/set5_homooligomers.csv` and an embedding generated by the `--save-representations` option in colabfold (.npy) files. Only embeddings are required for prediction.

Example of running AlphaFold2:
```
colabfold_batch 0_monomer.csv . --num-models 5 --model-type alphafold2_multimer_v3 --num-recycle 5 --save-single-representations
```

### **Requirements and installation** ###

1. Clone repository
```
git clone [https://github.com/labstructbioinf/dc2_oligo](https://github.com/Rmadeye/dc2_oligo/)
```

2. Create a virtual environment (using conda, for example) and install the requirements

```
conda create -n dc2_oligo
cd dc2_oligo
pip install .
```

3. Check if everything works with pytest

```bash
cd dc2_oligo
python -m pytest
```

### **Usage** ###

```bash
python predict.py --cf_results DIR --save_csv STR

 ```
 | Argument        | Description |
|:-------------:|-------------|
| **`--cf_results`** | Colabfold output directory with saved embeddings via --save-representations option |
| **`--save_csv`** | Save csv with input filename (optional) |

```bash
python predict.py --cf_results tests/data/0 --save_csv testoutput.csv
```

### **Additional information** ##

For best results, enter sequences that contain only __coiled coil domain__. You can easily identify such a domain with [__DeepCoil__](https://github.com/labstructbioinf/DeepCoil)  predictor. Please use the AlphaFold2 multimer embeddings (**alphafold2_multimer_v3**).
