# Using a factory design pattern is necessary to restrict tensorflow imports to their respective subprocess call
def get_model(model_name):
    """
    bindings for classes of models.py as a factory design pattern
    """
    import models
    if model_name == 'Actor':
        return models.Actor
    elif model_name == 'V_MLP_model':
        return models.V_MLP_model
    elif model_name == 'MLP_model':
        return models.MLP_model
    elif model_name == 'Dense_CNN_Model':
        return models.Dense_CNN_Model
    elif model_name == 'V_Dense_CNN_Model':
        return models.V_Dense_CNN_Model
