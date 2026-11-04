import torch
from pathlib import Path

text_path = Path(__file__).with_name("shakespear.txt")
with text_path.open("r", encoding="utf-8") as f:
    text = f.read()

torch.set_printoptions(
    threshold=float('inf'),
    linewidth=100000000
)


#-Linear-#
batch = 4
contextWindow = 10
emb_dim = 32
Iters = 2885
#iterations per epoch
epoch = 10
lr = 1e-2
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
#------------#


#TOKENIZATION
chars = sorted(list(set(text)))
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
        self.head_count = 2
        self.head_size = 16
        self.contextWindow = contextWindow

        #-------MATRICES TO TRAIN-------#
        self.emb_mat = torch.nn.Parameter(torch.rand((len(chars), emb_dim), dtype=torch.float32)*0.02)
        self.pos_emb = torch.nn.Parameter(torch.rand((self.contextWindow , self.emb_dim), dtype = torch.float32))
        self.W_Q = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn((self.emb_dim, self.head_size), dtype=torch.float32)*0.02)
            for a in range(self.head_count)]
        )
        self.W_K = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn((self.emb_dim, self.head_size), dtype=torch.float32)*0.02)
            for a in range(self.head_count)]
        )
        self.W_V = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn((self.emb_dim, self.head_size), dtype=torch.float32)*0.02)
            for a in range(self.head_count)]
        )
        self.W_O = torch.nn.ModuleList(
            [torch.nn.Parameter(torch.randn((self.headsize * self.head_count, self.emb_dim), dtype=torch.float32)*0.02)
            for a in range(self.head_count)]
        )
    def get_batch(self):
        start = torch.randint(0,len(self.data) - self.contextWindow, (batch,))
        self.input = torch.stack([self.data[s:s+self.contextWindow] for s in start])
        self.ans = torch.stack([self.data[s+1:s+self.contextWindow+1] for s in start])
        return self.input,self.ans

    def forward(self):
        self.get_batch()

        self.input = self.emb_mat[self.input] + self.pos_emb
        # b = batch, i = head count, c = contextWindow, e = emb_dim, h = headsize
        self.query = torch.einsum("bce + ieh -> bich", self.input, self.W_Q)
        self.key = torch.einsum("bce + ieh -> bich", self.input, self.W_K)
        self.value = torch.einsum("bce + ieh -> bich", self.input, self.W_V)

        attentionScores = torch.einsum("bich + bihc -> bhcc",self.query,self.key.transpose(-2,-1)) / emb_dim**0.5
        self.attentionScores = attentionScores.masked_fill_(torch.tril(torch.ones(contextWindow, contextWindow), diagonal = 0) == 0, float("-inf"))

        attentionScores = torch.softmax(self.attentionScores, dim = -1)

        Value = attentionScores @ self.value

        self.out = self.W_O(Value)

        print(self.out.shape)


    def feedforward(self,out_mat):

        Layer1 = torch.nn.Linear


if __name__ == "__main__":
    Run = Transformer()
    Run.forward()