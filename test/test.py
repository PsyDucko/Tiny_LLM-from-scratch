import torch

from tokenizer.tokenizer import ByteTokenizer
from model.transformer import Transformer


# ============================================================
# Configuration
# ============================================================

VOCAB_SIZE = 259

D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
NUM_LAYERS = 6
MAX_SEQ_LEN = 512

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = ByteTokenizer()


# ============================================================
# Load model
# ============================================================

model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "transformer.pth",
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# Translation function
# ============================================================

def translate(source_text, max_length=50):

    # --------------------------------------------------------
    # Encode source
    # --------------------------------------------------------

    src = tokenizer.encode(
        source_text,
        add_bos=True,
        add_eos=True
    )

    src = src.unsqueeze(0).to(DEVICE)
    # [1, S]


    # --------------------------------------------------------
    # Start decoder with BOS
    # --------------------------------------------------------

    tgt = torch.tensor(
        [[tokenizer.BOS]],
        dtype=torch.long,
        device=DEVICE
    )
    # [1, 1]


    # --------------------------------------------------------
    # Autoregressive generation
    # --------------------------------------------------------

    with torch.no_grad():

        for _ in range(max_length):

            # Forward pass
            logits = model(
                src,
                tgt
            )

            # logits:
            # [1, T, VOCAB_SIZE]

            # Get prediction for LAST position
            next_token_logits = logits[:, -1, :]
            # [1, VOCAB_SIZE]

            # Greedy decoding
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )
            # [1, 1]

            # Append predicted token
            tgt = torch.cat(
                [tgt, next_token],
                dim=1
            )

            # Stop if EOS generated
            if next_token.item() == tokenizer.EOS:
                break


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    result = tokenizer.decode(
        tgt[0]
    )

    return result


# ============================================================
# Test
# ============================================================

while True:

    text = input("\nEnglish: ")

    if text.lower() == "exit":
        break

    translation = translate(text)

    print("French:", translation)