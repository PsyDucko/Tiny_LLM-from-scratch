import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tokenizer.tokenizer import ByteTokenizer
from model.transformer_decoder_only import TransformerDecoderOnly


# ============================================================
# Configuration
# ============================================================

VOCAB_SIZE = 259

D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
NUM_LAYERS = 6
MAX_SEQ_LEN = 512

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = ByteTokenizer()


# ============================================================
# Load CSV
# ============================================================

texts = []

with open("cefr_leveled_texts.csv",
    "r",
    encoding="utf-8"
) as f:

    reader = csv.reader(f)

    # Skip header
    next(reader)

    for row in reader:

        if len(row) == 0:
            continue

        # ----------------------------------------------------
        # First column contains English text/story
        # ----------------------------------------------------

        text = row[0].strip()

        if text:
            texts.append(text)


print(f"Number of texts: {len(texts)}")


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):

    def __init__(
        self,
        texts,
        tokenizer,
        max_seq_len
    ):

        self.samples = []

        for text in texts:

            # ------------------------------------------------
            # Encode entire text
            # ------------------------------------------------

            tokens = tokenizer.encode(
                text,
                add_bos=True,
                add_eos=True
            )

            # ------------------------------------------------
            # Split long text into chunks
            # ------------------------------------------------

            # We need:
            #
            # input  = tokens[:-1]
            # target = tokens[1:]
            #
            # Therefore each sample can contain at most
            # MAX_SEQ_LEN tokens.
            # ------------------------------------------------

            for i in range(
                0,
                len(tokens) - 1,
                max_seq_len
            ):

                chunk = tokens[
                    i : i + max_seq_len
                ]

                # Need at least 2 tokens
                if len(chunk) < 2:
                    continue

                input_ids = chunk[:-1]
                target_ids = chunk[1:]

                self.samples.append(
                    (
                        input_ids,
                        target_ids
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx]


# ============================================================
# Create Dataset
# ============================================================

dataset = TextDataset(
    texts,
    tokenizer,
    MAX_SEQ_LEN
)

print(f"Number of training chunks: {len(dataset)}")


# ============================================================
# Collate Function
# ============================================================

def collate_fn(batch):

    inputs = []
    targets = []

    max_len = max(
        len(x[0])
        for x in batch
    )

    for input_ids, target_ids in batch:

        padding_length = max_len - len(input_ids)

        # --------------------------------------------
        # Pad input
        # --------------------------------------------

        input_ids = torch.cat([
            input_ids,
            torch.full(
                (padding_length,),
                tokenizer.PAD,
                dtype=torch.long
            )
        ])

        # --------------------------------------------
        # Pad target
        # --------------------------------------------

        target_ids = torch.cat([
            target_ids,
            torch.full(
                (padding_length,),
                tokenizer.PAD,
                dtype=torch.long
            )
        ])

        inputs.append(input_ids)
        targets.append(target_ids)

    return (
        torch.stack(inputs),
        torch.stack(targets)
    )


# ============================================================
# DataLoader
# ============================================================

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


# ============================================================
# Model
# ============================================================

model = TransformerDecoderOnly(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN
).to(DEVICE)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.PAD
)


# ============================================================
# Optimizer
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ============================================================
# Training
# ============================================================

print(f"Using device: {DEVICE}")

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        unit="batch"
    )

    for input_ids, target_ids in progress_bar:

        # ----------------------------------------------------
        # Move to GPU
        # ----------------------------------------------------

        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        #
        # Decoder-only model:
        #
        # input_ids
        #      ↓
        # Transformer
        #      ↓
        # logits
        #
        # [B, T, VOCAB_SIZE]
        #

        logits = model(input_ids)

        # ----------------------------------------------------
        # Reshape for CrossEntropyLoss
        # ----------------------------------------------------

        # logits:
        #
        # [B, T, V]
        #
        # becomes:
        #
        # [B*T, V]
        #

        logits = logits.reshape(
            -1,
            VOCAB_SIZE
        )

        # target:
        #
        # [B, T]
        #
        # becomes:
        #
        # [B*T]
        #

        targets = target_ids.reshape(-1)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            targets
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = (
        total_loss / len(dataloader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Average Loss: {average_loss:.4f}"
    )


# ============================================================
# Save Model
# ============================================================

torch.save(
    model.state_dict(),
    "transformer.pth"
)

print("Model saved to transformer.pth")