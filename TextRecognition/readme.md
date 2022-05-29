# Text Recognition 
- Character Recognition
- Word Recognition (RCNN)
- Pytesseract

## Description of Files:

For Word Recognition Model:
worddatagenerator.ipynb - word image data generator (uses trdg package)
dataset.py - facilitate data loading using torch.utils.data
loss.py - CTC Loss implementation
model.py - model architecture 
converter.py - encode and decode from string to classes
trainer.py - main file to train the model

For Char Recognition Model:
emnist.py - main trainer file
EMNIST_model.py - model architecture
(Due to file size limit on github, the custom character has not been included) 

Pytesseract:
pytesseract.ipynb - call the pytesseract library
