## Directory Structure
- `main_torch.py`: Main python script you will run to train model and for hyperparameter search
- `model/resnet.py`: DNN model definition
- `data_gen/`: how raw datasets are processed into numpy tensors

## Medical Conditions (12) (subract 1 from ID when processing)
- A41: sneeze/cough	
- A42: staggering	
- A43: falling down	
- A44: headache
- A45: chest pain	
- A46: back pain	
- A47: neck pain	
- A48: nausea/vomiting
- A49: fan self	
- A103: yawn	
- A104: stretch oneself	
- A105: blow nose

**Note:** The repo contains tensorflow based graph neural networks, please ignore those and only use the pytorch models.
