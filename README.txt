Definer：
software_engineering_show：This is the main program for visualizing the software interface, running this py file will display the software interface.
test_data：Simple data examples are provided, and the files are in txt format files. Each data is two rows, including name and data, where the data column is a set of gene sequences consisting of ACCU.
model_cerevisiae、model_musculus、model_sapiens:Models for predicting results in a visual software interface.

Install:
conda create -m Definer python==3.8.8
conda activate Definer
pip install -r requires_env.txt

Usage:
cd .\Definer\code
python software_engineering_show.py
Click the Upload File button to upload the txt file data.
Select the model used for the data at the top right.
Click Confirm to make prediction, the prediction result is displayed in Result window, click Download to download the data.
The output contains three columns of data, the name of the sequence species, the length of the sequence, and the presence or absence of the locus, with Yes indicating the presence of the locus and no indicating the absence of the locus.
