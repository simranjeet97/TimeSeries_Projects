import utilis
models = {
    "fbprophet": utilis.FBProphetPredictor,}
    
model_name = 'fbprophet'
print(models[model_name]())