"""
author: nitin tiwari
email:er.nitintiwari548@gmail.com
"""

from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_str)

def main(data, eta, epochs, filename, plotName):
    df = pd.DataFrame(data)
    logging.info(f"This is actual dataframe{df}")
    X,y = prepare_data(df)

    model =Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    _ = model.total_loss() # underscore _ is adummy variable , you can remove it if you want

    save_model(model, filename=filename)
    save_plot(df, plotName, model)


if __name__ == '__main__': 
      
    AND ={
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,0,0,1],
    }
    ETA = 0.3
    EPOCHS = 10

    main(data=AND, eta=ETA, epochs=EPOCHS, filename="and.model", plotName="and.png")
