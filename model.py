from transformers import BertConfig, BertForMaskedLM

def build_model():
    """
    Builds the same small-from-scratch model used in the notebook.
    """
    config = BertConfig(
        vocab_size=30522,      # bert-base-uncased vocab
        hidden_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    return BertForMaskedLM(config)

if __name__ == "__main__":
    model = build_model()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", params)
