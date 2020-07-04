# COVID-19-X-Ray-Classification
Utilizing Deep Learning to detect COVID-19 and Viral Pneumonia from x-ray images 

View file `COVID-19 X-Ray Classification.ipynb` for all the code involved in this project

Dataset can be downloaded from https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

This project also uses COVID-19 X-Rays from https://www.kaggle.com/nabeelsajid917/covid-19-x-ray-10000-images?

**Research Papers for further analysis:**
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7187882/
  - https://arxiv.org/pdf/2004.05758.pdf
  - https://arxiv.org/pdf/2003.09871.pdf

## Project Structure

Training Set contains:
  - 200 COVID-19 X-Rays 
  - 250 Viral Pneumonia X-Rays
  - 250 Normal X-Rays 
  - Total: 700 images 
  
Testing Set contains:
  - 89 COVID-19 X-Rays 
  - 300 Viral Pneumonia X-Rays
  - 300 Normal X-Rays 
  - Total: 689 images 

## Model Architecture 

![COVID Model 2](https://user-images.githubusercontent.com/43652410/84108527-68930e80-a9ee-11ea-8774-eec8af3ec103.png)


![COVID Model Arch copy](https://user-images.githubusercontent.com/43652410/86502216-d670f280-bd6e-11ea-842b-5a9d66168d5e.png)


## Results

  - Achieved 93% Accuracy on Testing Set, with F-1 Score of 0.93, using 5 Convolutional Layers and 25 Epochs. 

![COVID Model Graph](https://user-images.githubusercontent.com/43652410/83370470-824caa00-a38d-11ea-89ee-cb411d586838.png) ![COVID Model Graph 2](https://user-images.githubusercontent.com/43652410/83370483-8ed10280-a38d-11ea-9080-5ae5f11fc23c.png)

  ![accuracy](https://user-images.githubusercontent.com/43652410/83465941-8b984e00-a443-11ea-8dac-2eac3aed38b4.png)

![f1score](https://user-images.githubusercontent.com/43652410/83465716-cd74c480-a442-11ea-8768-63d91a2df320.png)


![COVID Model Confusion Matrix](https://user-images.githubusercontent.com/43652410/86201736-52c8c300-bb2e-11ea-812d-ea79d9bfdcf3.png)

  
## Findings 

It is inherently difficult to differentiate between the occurence of the two diseases from a normal x-ray. In fact, ~20 million radiology reports contain clinically significant errors, where 10% play a role in patient deaths. Deep learning offers a solution to this problem.

![xray graphic](https://user-images.githubusercontent.com/43652410/83683191-fddb7080-a5b2-11ea-8957-16e59a7e03ea.jpg)

Saliency maps can help us better understand the features in the x-rays and visualize what areas of the image are of high importance. The areas of yellow gradient have the greatest influence on the model's prediction. 

![covid saliency graphic](https://user-images.githubusercontent.com/43652410/85191833-4d49be00-b28e-11ea-8421-534dcee5b4fe.jpg)


