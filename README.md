# oneNeuron
oneNeuron | perceptron


## ADD URL -
[Git Handbook](https://guides.github.com/introduction/git-handbook/)


## Add image -
![Sample Image](oneNeuron/plots/and.png/)

## Python Code -
```python
def main(data, eta, epochs, filename, plotName):
    df = pd.DataFrame(data)
    logging.info(f"This is actual dataframe{df}")
    X,y = prepare_data(df)

    model =Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    _ = model.total_loss() # underscore _ is adummy variable , you can remove it if you want

    save_model(model, filename=filename)
    save_plot(df, plotName, model)

```
