import argparse
import os
import torch
from maba_sparse.model import MabaSparseForCausalLM, get_101m_config


def test_checkpoint(checkpoint_path: str, prompt: str = "Hi Doctor, how are you today?"):
    print(f"Loading checkpoint from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} not found!")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint metadata: Steps={checkpoint.get('steps')}, Final Loss={checkpoint.get('final_loss')}")

    cfg = checkpoint.get("config") or get_101m_config()
    model = MabaSparseForCausalLM(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Model successfully loaded on {device} ({sum(p.numel() for p in model.parameters()):,} parameters).")

    # Encode prompt (byte-level encoding matching DialogDataset)
    input_ids = torch.tensor([[b % cfg.vocab_size for b in prompt.encode("utf-8")]], dtype=torch.long, device=device)
    print(f"\nPrompt: '{prompt}' (Input shape: {input_ids.shape})")

    with torch.no_grad():
        out_ids = model.generate(input_ids, max_new_tokens=32, temperature=0.7, top_k=40)

    decoded = "".join([chr(tok) if 32 <= tok < 127 else "?" for tok in out_ids[0].tolist()])
    print(f"Generated text: {decoded}")
    print("\nCheckpoint verification passed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/maba_dialog_checkpoint.pt")
    parser.add_argument("--prompt", type=str, default="Hi Doctor, how are you today?")
    args = parser.parse_args()
    test_checkpoint(args.checkpoint, args.prompt)
