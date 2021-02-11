# COVID-19 Detection Web App

Test individual x-ray images with the Deep Learning model to detect instances of COVID-19 & Viral Pneumonia

## How to Run the Code

  - Clone the repository `git clone https://github.com/faizancodes/COVID-19-X-Ray-Classification.git`
  
  - Navigate to the corresponding folder `cd COVID-19-X-Ray-Classification\webapp`
  
  - Create a virtual environment
  
       - **Windows:**
            ```
            py -3.6 -m venv env
            env\Scripts\activate
            ```
       - **Mac & Linux**
            ```
            python3.6 -m venv env
            source env/bin/activate
            ```   
            
  - Download all dependencies `pip install -r requirements.txt` 
  
  - Run the code `streamlit run webapp.py`

  - **Upload the x-ray images you want to test in the `test` folder before using the webapp**

## Example

![image](https://user-images.githubusercontent.com/43652410/107605958-a3df8000-6c02-11eb-9194-96f8c40be761.png) ![image](https://user-images.githubusercontent.com/43652410/107605985-b48ff600-6c02-11eb-9243-1c91f9aad051.png)
