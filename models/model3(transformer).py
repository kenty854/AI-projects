import torch
from pathlib import Path

text_path = Path(__file__).parent / "datasets" / "shakespear.txt"
with text_path.open("r", encoding="utf-8") as f:
    text = f.read()

torch.set_printoptions(
    threshold=float('inf'),
    linewidth=100000000
)


#-Linear-#
batch = 64
contextWindow = 120
emb_dim = 120
Iters = 1000
#iterations per epoch
epoch = 10
lr = 1e-3
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
#------------#


#TOKENIZATION
chars = sorted(list(set(text)))
chars.append("->")
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])
dataset = torch.tensor(encode(text))


# DATA SPLIT
n = int(0.8 * len(dataset))
train_data = dataset[:n]
val_data = dataset[n:]



class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #-----------DIMENSION-----------#
        self.data = train_data
        self.vocab_size = len(chars)
        self.emb_dim = emb_dim
        self.head_count = 6
        self.head_size = 32
        self.contextWindow = contextWindow

        #-------MATRICES TO TRAIN-------#

        self.emb_mat = torch.nn.Parameter(torch.rand((len(chars), emb_dim), dtype=torch.float32)*0.02)
        self.pos_emb = torch.nn.Parameter(torch.rand((self.contextWindow , self.emb_dim), dtype = torch.float32) * 0.02)

        self.W_Q = torch.nn.Parameter(
            torch.randn(self.head_count, self.emb_dim, self.head_size) * 0.02
        )
        self.W_K = torch.nn.Parameter(
            torch.randn(self.head_count, self.emb_dim, self.head_size) * 0.02
        )
        self.W_V = torch.nn.Parameter(
            torch.randn(self.head_count, self.emb_dim, self.head_size) * 0.02
        )

        self.W_O = torch.nn.Parameter(
            torch.randn(self.head_count * self.head_size, self.emb_dim) * 0.02
        )

        self.FNN = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, 4 * self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.Linear(self.emb_dim, self.vocab_size)
        )


        path = Path(__file__).parent / "model.pt"
        if path.exists():
            self.load_state_dict(torch.load(path))

    def get_batch(self):
        start = torch.randint(0,len(self.data) - self.contextWindow, (batch,))
        q = torch.stack([self.data[s:s+self.contextWindow] for s in start])
        ans = torch.stack([self.data[s+1:s+self.contextWindow+1] for s in start])
        return q,ans

    def forward(self, x):

        batch, contextWindow = x.shape
        x = self.emb_mat[x] + self.pos_emb[:contextWindow]

        # b = batch, i = head count, c = contextWindow, e = emb_dim, h = headsize
        self.query = torch.einsum("bce, ieh -> bich", x, self.W_Q)
        self.key = torch.einsum("bce, ieh -> bich", x, self.W_K)
        self.value = torch.einsum("bce, ieh -> bich", x, self.W_V)

        attentionScores = (self.query @self.key.transpose(-2,-1)) / self.head_size**0.5
        self.attentionScores = attentionScores.masked_fill(torch.tril(torch.ones(contextWindow, contextWindow), diagonal = 0) == 0, float("-inf"))

        attentionScores = torch.softmax(self.attentionScores, dim = -1)

        Value = attentionScores @ self.value

        Value = Value.permute(0,2,1,3)
        Value = Value.reshape(batch, contextWindow, self.head_count * self.head_size)

        self.out = x + Value @ self.W_O

        self.out = self.FNN(self.out)

        return self.out


def train(model, iter, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
 
    for i in range(iter):
        x, ans = model.get_batch()
        logits = model(x)
 
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            ans.reshape(-1),
        )
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if i % 500 == 0:
            print(i, loss.item())
            print()

    yn = input("save trained model parameters (y/n): ")

    if yn.lower() == "y":
        print("saving...")
        torch.save(model.state_dict(), Path(__file__).parent / "model.pt")
    if yn.lower() == "n":
        print("unsaved")

@torch.no_grad()
def generate(model, max_length):
    model.eval()
 
    end_id = stoi["->"]
    ids = torch.tensor([[end_id]])  # start token, shape (1, 1)
 
    for _ in range(max_length):
        logits = model(ids[:, -contextWindow:])[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
 
        if next_id.item() == end_id:
            break
 
    return decode(ids[0].tolist())

def eval(model, iter):
    model.eval()
    with torch.no_grad():
        for i in range(iter):
            x, ans = model.get_batch()
            logits = model(x)
 
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                ans.reshape(-1),
            )
 
            if i % 100 == 0:
                print(i, loss.item())
                print()

model = Transformer()
pri
