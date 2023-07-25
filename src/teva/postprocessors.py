
def squad_postprocessor(output_or_target, example=None, is_target=False):
    """Returns `no answer` if `output_or_target` is empty string"""
    if output_or_target:
        return output_or_target
    
    return "no answer"
