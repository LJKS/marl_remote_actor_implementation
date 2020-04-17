# Using a factory design pattern is necessary to restrict tensorflow imports to their respective subprocess call
def get_model(model_name):
    import models
    if model_name == 'Actor':
        return models.Actor
    elif model_name == 'V_MLP_model':
        return models.V_MLP_model
    elif model_name == 'MLP_model':
        return models.MLP_model
