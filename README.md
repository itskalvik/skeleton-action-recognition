## Directory Structure
- `main_torch.py`: Main python script you will run to train model and for hyperparameter search
                   Use this script to change model, optimizer and other general hyperparameters

- `model/resnet.py`: DNN model definition, The model uses a custom layer to generate spectrograms
                     and passes it on to a standard resnet 18 model.

- `data_gen/`: These scripts are used to convert NTU dataset from their native format to numpy tensors
               For now you only need to use gen_joint_data.py. 
               
### Medical Conditions (12) (subract 1 from ID when processing)
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
