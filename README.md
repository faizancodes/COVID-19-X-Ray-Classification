# COVID-19-X-Ray-Classification
Utilizing Deep Learning to detect COVID-19 and Viral Pneumonia from x-ray images 

Dataset can be downloaded from https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

This project also uses COVID-19 X-Rays from https://www.kaggle.com/nabeelsajid917/covid-19-x-ray-10000-images?

**Research Papers for further analysis:**
  - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7187882/
  - https://arxiv.org/pdf/2004.05758.pdf
  - https://www.nature.com/articles/s41598-019-42557-4

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

## Results

  - Achieved 93% Accuracy on Testing Set, with F-1 Score of 0.93, using 5 Convolutional Layers and 25 Epochs. 
  
  ![COVID Model Graph](https://user-images.githubusercontent.com/43652410/83370470-824caa00-a38d-11ea-89ee-cb411d586838.png) ![COVID Model Graph 2](https://user-images.githubusercontent.com/43652410/83370483-8ed10280-a38d-11ea-9080-5ae5f11fc23c.png)
  
