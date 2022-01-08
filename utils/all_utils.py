import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from matplotlib.colors import ListedColormap
import os
plt.style.use("fivethirtyeight")
import logging

def prepare_data(df):
  # docstring(an illustration of method)
  """it is used to seprate features and lebels

  Args:
      df (pd.DataFrame): it is pandas data frame

  Returns:
      [tuple]: it returns the tuple of dependent variables and independent variable
  """
  logging.info("Prepairing data by segrigeting dependent and independent variables")
  x = df.drop("y",axis=1)
  y = df["y"]
  return x,y



def save_model(model,filename):
  logging.info("saving the trained model")
  model_dir = "models"
  os.makedirs(model_dir,exist_ok=True) # only create if model_dir not present
  filePath = os.path.join(model_dir,filename) # model/filePath
  joblib.dump(model,filePath)
  logging.info(f"saved the trained model{filePath}")



def save_plot(df, file_name,model):
  
  # in Pycharm generally we use this type docstring
  """
  : param df: it is a Dta Frame
  : param file-name: this is the path to save the plot
  : param model: trained model
    
    """
  def _create_base_plot(df):
    logging.info("creating the base plot")
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color = "black", linestyle = "--", linewidth=1)
    plt.axvline(x=0, color = "black", linestyle = "--", linewidth=1)

    figure = plt.gcf() #get current figure
    figure.set_size_inches(10,8)

  def _plot_decision_regions(X, y, classfier ,resolution=0.02):
    logging.info("plotting the decision regions")
    colors = ("red","blue","green","gray","cyan")
    cmap=ListedColormap(colors[ :len(np.unique(y))])

    X=X.values # X as a array

    x1=X[:, 0]
    x2=X[:, 1]
    x1_min, x1_max = x1.min() -1, x1.max() +1
    x2_min, x2_max = x2.min() -1, x2.max() +1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))


    Z=classfier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    plt.plot()

    

  X,y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir="plots"
  os.makedirs(plot_dir,exist_ok=True) # only create if model_dir not present
  plotPath = os.path.join(plot_dir,file_name) # model/filePath
  plt.savefig(plotPath)
  logging.info(f"saving the plot{plotPath}")