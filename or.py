from numpy import test
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir="logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str,
filemode="a")

def main(data, eta, epochs, filename, plotName):
    df = pd.DataFrame(data)
    logging.info(f"This is actual dataframe{df}")
    X,y = prepare_data(df)

    model =Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    _ = model.total_loss() # underscore _ is adummy variable , you can remove it if you want

    save_model(model, filename=filename)
    save_plot(df, plotName, model)

OR ={
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,1],
}
ETA =0.3 # 0 and 1
EPOCHS = 100
try:
    logging.info("\n >>>>>>>>> starting trainning >>>>>>")
    main(data=OR, eta=ETA, epochs=EPOCHS, filename="or.model", plotName="or.png")
    logging.info(">>>>>>>>>trainning finish successfully <<<<<<<")
except Exception as e:
    logging.exception(e)
    raise e
