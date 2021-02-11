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

    ### **Upload the x-ray images you want to test in the `test` folder before using the webapp**

## Example

![image](https://user-images.githubusercontent.com/43652410/107689023-513fab80-6c76-11eb-861a-cb8b318eb937.png) ![image](https://user-images.githubusercontent.com/43652410/107689073-60265e00-6c76-11eb-8b19-587726664b9b.png)
![image](https://user-images.githubusercontent.com/43652410/107689092-6a485c80-6c76-11eb-9603-64e655b1452b.png) ![image](https://user-images.githubusercontent.com/43652410/107689230-9237c000-6c76-11eb-9cce-d255de63b5a8.png)
