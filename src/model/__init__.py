from model.gpt2 import GPT2BasedDecryptModel


def get_decrypt_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "gpt2":
        return GPT2BasedDecryptModel
    else:
        raise ValueError("Unrecognized model")


__all__ = [
    "get_decrypt_model",
    "GPT2BasedDecryptModel",
]
